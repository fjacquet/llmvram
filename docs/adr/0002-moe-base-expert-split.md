# 0002. MoE base and experts priced at measured rates

**Status:** Accepted, 2026-09-27 (v1.13.0)
**Spec:** `docs/superpowers/specs/2026-09-27-moe-weight-split-design.md`

## Context

ADR 0001 gives one average bytes/param per checkpoint. MoE checkpoints keep
their shared base (attention, dense layers, embeddings) wider than their
routed experts: Kimi K3 native stores a 57.2B BF16 base and MXFP4 experts.
Expert parallelism replicates the base on every GPU and divides the experts,
and decode reads the base in full, so the average understated EP memory
(Kimi K3 EP8: 207 vs ~272 GiB per GPU) and overstated batch-1 decode ~2.3x.

## Decision

- Quantized MoE safetensors refs carry `high_precision: { params_b, gib }`:
  the tensors stored as floats of 16 bits or more (BF16, F16, F32, F64), read
  from the Hugging Face `?expand[]=safetensors` dtype summary.
- F8, U8, I8 and I32 count as quantized storage: packed int4 is stored as
  I32 and NVFP4 block scales as F8, so byte width alone cannot identify the
  base. The U8 count is unreliable (logical vs packed), so quantized bytes
  come from the measured file total, never from their counts.
- `moeWeightSplit` fills the replicated base from the wide tensors first,
  then prices the rest at the quantized rate. EP memory and MoE decode use it.
- Without the field (GGUF, 16-bit checkpoints, no ref) the single average
  applies, unchanged. No tensor-name classification (rejected: needs header
  parsing per shard; the dtype summary is already fetched).

## Consequences

- EP memory and MoE decode follow the real checkpoint for 68 refs across 27
  models; totals and TP/PP memory are unchanged.
- Shared experts stored in the experts' type count as routed (small error).
- `quantization` is required in `calculateMultiNodeVRAM` so a dropped
  argument is a compile error, not a silent return to the average.

## Sources

Hugging Face dtype summaries (2026-09-27): moonshotai/Kimi-K3, nvidia/Kimi-K3-NVFP4,
deepseek-ai/DeepSeek-R1, Qwen/Qwen3-235B-A22B-GPTQ-Int4, nvidia/Qwen3-235B-A22B-NVFP4.
