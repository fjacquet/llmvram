# Design: Long-Context Support up to 1M+ Tokens

**Date:** 2026-09-07
**Status:** Approved (design)
**Scope:** Sequence-length caps raised to 10,485,760; activation memory bounded by
prefill chunk; TTFT replaced with a real prefill model; MoE active parameters added to
the schema and used for throughput. One unit bug fixed in the comparison store. No
framework-overhead recalibration.

## Problem

Models in `src/data/models.json` already declare context windows far beyond what the
calculator can express. Nine entries carry `context_length >= 1,000,192`, and Llama 4
Scout carries `10,485,760`. Meanwhile:

- `schemas.ts` caps `sequenceLength` at `131072` in **two** places — the inference
  schema and `TrainingInputSchema`.
- `SequenceLengthInput.tsx` has `MAX_LOG = 17` (131,072), presets stopping at 128K, and
  a hardcoded `128K` end label.

Raising the cap alone would expose three calculation errors that are invisible at 4K but
dominant at 1M.

### Verified deltas against apxml.com/tools/vram-calculator

Configuration replayed through our own engines: Qwen3.6 35B A3B, FP16 weights, INT4 KV
cache, batch 1, sequence length 1024, single 128 GB device.

| Metric | apxml | ours | delta |
|---|---|---|---|
| Total VRAM | 81.7 GB | 68.07 GB | ours −13.6 GB |
| Generation speed | ~35 tok/s | 3.8 tok/s | ours 9× too slow |
| TTFT | ~88 ms | 527 ms | ours 6× too slow |

Our breakdown at seq=1024: `W=67.06 · KV=0.01 · ACT=0.00 · FW=1.00`.

### Root causes

**1. `estimatePerformance` ignores MoE active parameters.** `performance.ts:74` computes
the memory-bound roofline as `bandwidth / (num_parameters_billion × 2)`, using the
**total** 36B. MoE decode reads only the active experts. `273e9 / (36e9 × 2) = 3.79`,
exactly our reported figure. With ~3B active the result is ~45 tok/s, consistent with
apxml's 35. This affects every MoE model at every context length.

Compounding it, `calculateMoEActiveParams` uses a flat 20% shared / 80% expert split and
returns **8.1B** for a model literally named **A3B** — 2.7× too high for fine-grained
MoE (256 experts, 8 active).

**2. TTFT has no sequence dependence.** `performance.ts:117` is
`timeToFirstToken = 1 / (tokensPerSecond × 0.5)`. A 1M-token prompt reports the same TTFT
as a 10-token prompt. For a long-context calculator this is the most visible wrongness.

**3. Activation memory scales with the full context window.**
`calculateActivationMemory` computes `batch × sequenceLength × intermediate_size × 4`.
Real engines chunk prefill; decode activations are one token wide. Measured on Qwen3.6
27B (dense, `intermediate_size = 17408`), batch 1:

| seq | current ACT |
|---|---|
| 4,096 | 0.27 GB |
| 131,072 | 8.50 GB |
| 1,048,576 | **68.00 GB** |

The 1M figure is fiction. Note this error is invisible on MoE models, whose stored
`intermediate_size` is the per-expert value (Qwen3.6 35B A3B: 512).

## Non-goals

- **Framework overhead recalibration.** `FRAMEWORK_OVERHEAD_GB` stays at a flat 1.0 GB.
  The 13.6 GB VRAM delta with apxml will therefore remain open after this work. apxml
  appears to reserve a proportional CUDA/KV pool rather than a constant, but there is no
  solid public reference for the multiplier, so calibrating it is a separate decision.
- `FLASH_ATTENTION_LONG_THRESHOLD = 8192` making 8K and 1M share a retention bucket
  (training path only).
- Changing any VRAM value at or below 8,192 tokens. See "Blast radius" below.

## Design

### A1. Activation memory bounded by the prefill chunk

`src/engines/inference.ts`, `calculateActivationMemory`:

```
activations = batch × min(sequenceLength, PREFILL_CHUNK_TOKENS)
                    × effectiveIntermediateSize × 4
```

The only change to the expression is `sequenceLength` → `min(sequenceLength, chunk)`.

`PREFILL_CHUNK_TOKENS = 8192` in `constants.ts`. Source: vLLM's default
`max_num_batched_tokens` for `UsageContext.LLM_CLASS` on GPUs below 70 GiB (16384 above,
2048 for the OpenAI API server context). The constant's doc comment carries the link.

The `× 4` factor is **kept but re-documented**. The current comment claims "FP32 bytes
per activation", which is wrong for inference. It is `2 bytes (bf16) × ~2 live buffers
per layer`. Keeping the numeric value means **no VRAM figure below 8,192 tokens
changes**, which confines test churn to the long-context cases.

Effect on Qwen3.6 27B, batch 1:

| seq | ACT before | ACT after |
|---|---|---|
| 4,096 | 0.27 GB | 0.27 GB (unchanged) |
| 131,072 | 8.50 GB | 0.53 GB |
| 1,048,576 | 68.00 GB | 0.53 GB |

### A2. TTFT becomes a prefill model

`src/engines/performance.ts` replaces line 117. Prefill is compute-bound — a different
roofline regime from the bandwidth-bound decode path above it.

```
prefillFLOPs   = 2 × activeParams × T           (linear term)
               + 2 × layers × T² × hidden_size  (causal attention term)
prefillSeconds = prefillFLOPs / (gpuFLOPS × PREFILL_MFU)
TTFT           = prefillSeconds + 1 / tokensPerSecond
```

- Attention term derives from `½ × 4 × B × T² × D × layers = 2 × layers × T² × D`; the
  ½ is causal masking.
- `PREFILL_MFU = 0.45`. Literature gives 40–60% model FLOPs utilization for prefill,
  versus 1–5% for batch-1 decode. Named constant, doc comment records the range.
- `T = sequenceLength`. **Batch size is not applied to `prefillFLOPs`**: TTFT is a
  per-request latency, and the modelled request is one sequence of `T` tokens. Batch
  continues to affect decode throughput only.
- Multi-GPU: `gpuFLOPS × numGPUs × scalingEfficiency`, matching what the decode path
  already does with bandwidth.
- **GPUs with neither `fp16_tflops` nor `fp32_tflops`:** prefill time is not computable.
  Fall back to the previous heuristic (`1 / (tokensPerSecond × 0.5)`), set
  `prefillSeconds = null` and `prefillEstimateDegraded = true`, rather than returning
  `Infinity` or `0`.

`PerformanceEstimate` gains:
- `prefillSeconds: Decimal | null` — null only in the degraded case above.
- `prefillBottleneck: 'linear' | 'attention'` — `'attention'` when the quadratic term
  strictly exceeds the linear term, `'linear'` otherwise. At 1M the quadratic term takes
  over, and surfacing that is the point of the feature.
- `prefillEstimateDegraded: boolean`.

### A3. MoE active parameters drive throughput

`performance.ts` uses `activeParams` instead of `num_parameters_billion` for:
- the memory-bound decode roofline, and
- the linear term of `prefillFLOPs`.

Weight VRAM continues to use **total** parameters. The CLAUDE.md MoE rule ("all expert
weights must fit in VRAM") is unaffected — only throughput changes.

### Interface change

`PerformanceParams` gains `sequenceLength: number`. This propagates to:

- `src/hooks/useInferenceCalculation.ts`
- `src/workers/calculation.worker.ts`
- `src/store/comparisonStore.ts`
- `src/utils/exportPptx.ts`
- `src/components/layout/ResultsPanel.tsx`

### B1. Schema field

`src/utils/schemas.ts`, `ModelSchema`, beside the existing MoE fields:

```ts
active_parameters_billion: z.number().positive().optional(),
```

Optional: dense models omit it, and `CustomModelInput` exposes no MoE fields at all (the
custom-model form in `ModelSelector.tsx` collects name, params, hidden size, layers and
heads only), so nothing is needed on the custom path.

### B2. Three-tier fallback in `calculateMoEActiveParams`

1. `active_parameters_billion` when present — used as-is.
2. Otherwise derive from stored fields:
   ```
   expertParams = layers × num_experts × 3 × hidden_size × intermediate_size / 1e9
   active       = (total − expertParams) + expertParams × (num_experts_per_token / num_experts)
   ```
   This works because our MoE entries already store the **per-expert**
   `intermediate_size`. On Qwen3.6 35B A3B it yields **4.8B**, against **8.1B** for the
   current 20/80 heuristic and **3B** actual — better, still not exact, which is why
   tier 1 exists.
3. Dense architecture or missing MoE fields → `num_parameters_billion`.

The 20/80 heuristic is deleted.

### B3. Populating 33 MoE entries

**11 encode it in the model name** — direct entry:
Gemma 4 26B **A4B**, Kimi Linear 48B **A3B**, Nemotron 3 Nano 30B **A3B**, Nemotron 3
Super 120B **A12B**, Nemotron 3 Ultra 550B **A55B**, Nemotron 3.5 Lightning 30B **A3B**,
Qwen3 235B **A22B**, Qwen3.6 35B **A3B**, Qwen3.8 2.4T **A95B**, Llama 4 Maverick and
Llama 4 Scout (both **17B** active).

**22 require verification** against the model card and `config.json` reachable from each
entry's existing `hf_url`, via the Hugging Face MCP — one at a time, no values written
from memory:
DeepSeek R1, DeepSeek V4 Flash, DeepSeek V4 Pro, GLM 4.7, GLM 4.7 Flash, GLM 5.2,
GPT OSS 20B, GPT OSS 120B, Kimi K2 Instruct, Kimi K2 Thinking, Kimi K2.5, Kimi K2.6,
Kimi K2.7 Code, Kimi K3, Ling 3.0 Tiny, Ling 3.0 Flash, MiniMax M2.1, MiniMax M2.5,
MiniMax M2.7, MiniMax M3, Mistral Large 3 675B, Mistral Small 4 119B.

This is the bulk of the data work and is sized here deliberately rather than discovered
during implementation.

`src/data/models.json` must stay sorted alphabetically by `name` after editing.

### B4. Refresh-script consistency check

`scripts/fetch-models.ts` gains the B2-tier-2 derivation as a **warning**, not a source
of truth: if the derived value diverges from a stored `active_parameters_billion` by more
than 25%, the script logs a warning. This catches a stale entry without overwriting a
hand-verified value.

### C1. Caps

Both `.max(131072)` occurrences in `src/utils/schemas.ts` — the inference schema and
`TrainingInputSchema.sequenceLength` — are raised to a shared
`MAX_SEQUENCE_LENGTH = 10_485_760`.

### C2. `SequenceLengthInput.tsx` — dynamic hint, not dynamic clamp

- `MAX_LOG` becomes derived: `log2(max(1_048_576, model.context_length ?? 0))`. Clean
  log2 = 20 for 53 models; extends to ~23.3 for Llama 4 Scout alone.
- Presets gain 256K, 512K, 1M. Presets above the selected model's native context stay
  clickable.
- `formatValue` currently renders `1048576` as `1024K tokens`. Switch to M above 1024K.
- The hardcoded `128K` end label becomes the dynamic maximum.
- **Native-context marker** on the slider track at `model.context_length`.
- **Warning badge** when `sequenceLength > context_length`: "beyond native context
  (262K) — requires RoPE scaling / YaRN".

**The user's value is never rewritten**, including on model change. Rationale: RoPE/YaRN
extension is a real workload; custom models carry no `context_length` at all; and
silently clamping on model change would destroy both the user's setting and any URL they
had shared.

`formatCtx` in `ModelSelector.tsx` already handles 1M and 10M correctly — verified, no
change.

### C3. New outputs

`ResultsPanel` shows prefill time and the dominant regime (`linear` / `attention`).

TTFT currently renders as ms via `.toFixed(1)`. At 1M that produces `45230.0 ms`.
Formatting becomes adaptive: ms below 1000, seconds above.

### C4. Unit bug in the comparison store

Found while tracing TTFT. `src/store/comparisonStore.ts:34` declares
`timeToFirstToken: number // ms`, but `src/components/layout/ResultsPanel.tsx:323` writes
`result.performance.timeToFirstToken.toNumber()` — **seconds**, with no `× 1000`.
`src/components/comparison/ComparisonColumn.tsx:300` then renders it as `ms`. Comparison
TTFT is 1000× too small: 0.53 s displays as "1 ms".

Fixed in this change. All three files are already being touched, and leaving a lying
`// ms` comment next to a new `prefillSeconds` field invites the same bug again.

### C5. Tests

- `inference.test.ts`, `inference.integration.test.ts` — only cases above 8,192 tokens
  move, thanks to the preserved `× 4`. New cases: activation plateau at 8K / 128K / 1M.
- `performance.test.ts` — substantial rewrite; the signature and formula both change.
  New cases: quadratic TTFT growth, `linear` → `attention` crossover, MoE throughput on
  active params, GPU with no FLOPS data.
- `kv-cache.test.ts` — formula unchanged; add a 1M case.
- `urlSerializer.test.ts` — round-trip `sequenceLength` at 1,048,576 and 10,485,760. The
  URL schema uses an unbounded `sl: z.number()`, so nothing breaks, but lock it down.
- `comparisonStore.test.ts` — lock the milliseconds after C4.
- Coverage targets (75% lines/functions/branches/statements on `src/engines/` and
  `src/utils/`) are maintained.

### C6. Documentation

- `CHANGELOG.md` and `README.md` — user-visible values change (MoE tok/s, TTFT,
  activations above 8K, comparison TTFT).
- `src/components/guide/GuidePage.tsx` describes TTFT in three places; the "2× slower
  than decode" characterization becomes false and must be rewritten.

## Blast radius on existing numbers

| Value | Changes? |
|---|---|
| VRAM at seq ≤ 8,192 | No |
| VRAM at seq > 8,192 (dense) | Yes — activations drop |
| VRAM at seq > 8,192 (MoE) | Negligible — per-expert `intermediate_size` |
| tok/s, dense models | No |
| tok/s, MoE models | Yes — up to ~9× faster |
| TTFT, all models | Yes — new formula |
| Comparison-view TTFT | Yes — 1000× correction |

## Open item to report, not to change

`Mistral Small 4 119B` has `num_kv_heads: 32` against `num_attention_heads: 32` — a GQA
ratio of 1.0, meaning no KV-cache reduction. Suspect for a recent Mistral. Its
`config.json` is opened anyway during B3; **report the discrepancy, do not modify the
data** without a separate decision.

## References

- [vLLM — Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization)
- [vLLM — EngineArgs batch-size defaults](https://docs.vllm.ai/en/stable/api/vllm)
- [Roofline fundamentals](https://github.com/harshuljain13/llm-inference-at-scale/blob/master/content/01_gpu_hardware/01.2_roofline_model/roofline_fundamentals.md)
- [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/pdf/2411.01783)
- [DuetServe: Harmonizing Prefill and Decode for LLM Serving](https://arxiv.org/pdf/2511.04791)
- `docs/vram-calculation-pitfalls.md`
