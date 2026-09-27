# MoE Weight Split: Base and Experts at Their Measured Rates

**Date:** 2026-09-27
**Status:** Design approved in conversation 2026-09-27 (approach B, fallback 1)

## Context

`weight_refs` (v1.12.0) give each checkpoint's measured total. The engine applies it
as one averaged bytes-per-parameter to every parameter. For a MoE checkpoint that
keeps its shared base (attention, dense layers, shared experts, embeddings) wider
than its routed experts, the average misprices both parts:

- Expert-parallel memory replicates the base and divides the experts by N. Kimi K3
  MXFP4 on 8 GPUs: the engine gives 207 GiB of weights per GPU; the base at its real
  BF16 rate gives ~272 GiB (24% under). At 64 GPUs: 51 vs ~124 GiB.
- Decode reads the base in full plus the touched experts. Batch-1 Kimi K3 reads
  ~58 GB by the average against ~137 GB real, so tokens/s is overstated ~2.3x.

Both errors point the dangerous way ("fits", "fast") on the flagship EP case.
This was finding I-4 of the weight-refs final review, documented as Domain Pitfall 9.

## Goals

1. Expert-parallel memory and every MoE decode path charge the base and the routed
   experts at their measured rates when the checkpoint allows it.
2. Totals, tensor/pipeline-parallel memory and dense models are unchanged.
3. Users can see whether a MoE split is measured or estimated.

## Non-Goals

- GGUF split measurement (no dtype summary; would need tensor-header parsing).
- Tensor-name classification. Shared experts stored in the experts' type count as
  routed; the error is small (a few billion parameters on the largest models).
- A conservative fallback for unmeasured formats (rejected: overcharges FP8 and GGUF
  bases).
- FLOPs: compute keeps following active parameters.

## Evidence (Hugging Face `?expand[]=safetensors`, 2026-09-27)

| Checkpoint | Wider than lowest type | Lowest type |
|---|---|---|
| moonshotai/Kimi-K3 | BF16 57.18B, F32 0.01B | U8 2722.74B |
| nvidia/Kimi-K3-NVFP4 | F8_E4M3 36.18B, BF16 21.00B, F32 0.01B | U8 1361.37B |
| deepseek-ai/DeepSeek-R1 | BF16 3.92B | F8_E4M3 680.57B |
| openai/gpt-oss-120b | BF16 2.17B | U8 114.66B |
| Qwen/Qwen3-235B-A22B | (single type) | BF16 235.09B |
| RedHatAI/Qwen3-235B-A22B-FP8-dynamic | BF16 1.38B | F8_E4M3 233.80B |

The U8 count is not reliable: Kimi K3 native reports logical parameters (2722.7B)
while the NVIDIA NVFP4 build reports packed bytes (1361.4B). Counts for 8-bit and
wider types are real parameters at known widths. The design therefore never uses the
lowest type's count; its bytes come from the measured file total.

## Section 1: Data and measurement

New optional field on a weight ref:

```ts
high_precision?: { params_b: number; gib: number }
```

- Sum over every dtype in the summary except the lowest-width one, when the
  summary lists more than one dtype. Widths: F64 8, F32/I32 4, BF16/F16/I16 2,
  F8_* / I8 / U8 1. The lowest-width dtype is the one with the smallest width; ties
  (e.g. F8_E4M3 and U8) resolve to the one with the larger count.
- `params_b` = count / 1e9; `gib` = sum(count x width) / 1024^3.
- Written only for MoE models (`architecture: 'moe'`) and only for safetensors refs
  whose summary has more than one dtype. GGUF refs and single-type checkpoints get
  no field.
- `refresh:models --measure <id>` prints it inside each ref; the default audit
  compares it with the live summary within 1% like `gib`.
- Data integrity (`models.test.ts`): `high_precision.gib <= gib`,
  `high_precision.params_b <= num_parameters_billion`, both positive.

## Section 2: Engine

One helper in `src/engines/quantization.ts`:

```ts
moeWeightSplit(model, format): { baseGiB: Decimal; routedGiB: Decimal; measured: boolean } | null
```

- Returns null for non-splittable models (`splitMoEParams` null).
- `total` = `calculateModelWeightVRAM(num_parameters_billion, format, model)`
  (unchanged figure). `baseGiB + routedGiB === total` always.
- With `ref.high_precision` for the format (fp16/bf16 twin rule applies), and
  `H = params_b`, `W = high_precision.gib`, `B = split.baseB`, `N = num_parameters_billion`:
  - `H >= B`: `baseGiB = W x B / H` (the base is all wide).
  - `H < B`: `lowRate = (total - W) / (N - H)`; `baseGiB = W + (B - H) x lowRate`.
  - `routedGiB = total - baseGiB`; `measured = true`.
- Without it: `baseGiB = total x B / N`, `measured = false` (today's behaviour).

A second helper, `routedTouchedFraction(model, batchSize)`, is extracted from
`calculateMoEBatchedParams`: `k/E` at batch 1, `1 - (1 - k/E)^B` above.
`calculateMoEBatchedParams` is rewritten on top of it (same results).

Consumers:

- **Expert-parallel memory** (`multi-gpu.ts` `calculateExpertParallelVRAM`):
  `weightsPerGPU = baseGiB + routedGiB / N`.
- **MoE decode bytes** (`performance.ts`): weights read per step =
  `baseGiB + routedGiB x touched`; under expert parallelism the routed term is
  divided by `gpusPerStage`. Dense models keep the existing path.
- Unchanged: totals, TP/PP memory, KV, FLOPs, prefill.

## Section 3: UI

For MoE models the weight-source line adds "base/expert split measured" or
"split estimated (average rate)". Dense models show no split text.

## Section 4: Testing

- Helper: Kimi K3 mxfp4 base ~103.2 GiB, routed ~1350.5 GiB (sum 1453.74);
  DeepSeek R1 fp8 base = BF16 bytes + the rest at the FP8 rate; a model without the
  field falls back to the fraction split with `measured: false`; parts always sum to
  the total; `routedTouchedFraction` matches `calculateMoEBatchedParams`.
- Anchors: Kimi K3 mxfp4 EP8 weights per GPU ~272 GiB (was 207); Kimi K3 mxfp4
  batch-1 decode tokens/s below the previous figure; a GGUF ref stays on the fallback.
- Guards: removing the helper from each consumer fails a test (mutation-checked).
- Auditor: pure extraction of `high_precision` from summary fixtures (Kimi native,
  NVIDIA NVFP4 tie between F8 and U8, DeepSeek FP8, single-type skip). No network in CI.

## Section 5: Delivery and docs

One PR on `feat/moe-weight-split`: auditor extraction and tests; schema field;
helpers and consumers; data via `--measure` for MoE models; UI text; docs.
Docs: CHANGELOG; `CLAUDE.md` (Pitfall 9 becomes a key pattern for the measured
split); `ARCHITECTURE.md`; guide; `refresh:models` description.
