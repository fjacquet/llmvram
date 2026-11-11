# 0009. Calibrated expected speed, sparse attention, and the inference fit at 90%

**Status:** Accepted, 2026-11-11
**Spec:** `docs/superpowers/specs/2026-11-11-calibration-sparse-attention-design.md`
**Audit:** `docs/superpowers/specs/2026-11-11-engine-audit-findings.md`
**Points:** `docs/calibration/2026-11-11-vllm-efficiency-points.md`

## Context

The engine audit found the headline speed was a roofline, shown as if it were a
forecast. Measured vLLM runs reach about half of it. Three smaller things were
also wrong:

- M1: the headline tok/s and TTFT were the theoretical ceiling. Real runs sit far below it.
- M2: inference fit at `total <= vram`, but vLLM allocates only
  `gpu_memory_utilization` 0.9 of the card.
- M4: two KV sizes were wrong. Mistral Large 3 was overstated about 24.9x and
  Nemotron Ultra 2.5x.

DeepSeek-style sparse attention (DSA) was also missing. GLM 5.2 and Mistral Large 4
read at most `topk` KV positions per token, not the whole context.

## Decision

- **D1 Two figures.** The ceiling is today's roofline, unchanged. Expected is
  the ceiling scaled by a measured efficiency. Expected tok/s = ceiling x 0.47.
  Expected TTFT and prefill = ceiling / 0.469, about 2.13x longer. Rack counts
  in the next piece use expected. ADR 0005's UI freeze is lifted for this second
  figure only. No new inputs.
- **D2 Constants.** (Decode is now banded by load, ADR 0010; the flat 0.47 below is the overall median, kept for bands up to 32.) `DECODE_EFFICIENCY_BANDS` 0.47 up to batch 32, the median of n=512 accepted vLLM
  points. `PREFILL_EFFICIENCY` 0.469, the median of n=62. The statistic is the
  median. The user chose it on 2026-11-11, replacing "pessimistic end": the
  minimum (0.126 decode, 0.101 prefill) is driven by outliers and would cut the
  ceiling about 8× (decode) and 10× (prefill), against about 2× at the median.
  Spread, for the record:

  | Phase   | n   | min   | p5    | p10   | p25   | median |
  | ------- | --- | ----- | ----- | ----- | ----- | ------ |
  | Decode  | 512 | 0.126 | 0.270 | 0.315 | 0.380 | 0.470  |
  | Prefill | 62  | 0.101 | 0.143 | 0.239 | 0.344 | 0.469  |

- **D3 Which points count.** Every point is accepted or rejected in the
  calibration doc. Prefill rules:
  - reject points whose engine single-prompt prefill is under 50 ms (fixed
    request latency dominates);
  - correct queueing with the closed-loop factor 0.9 + 0.1 x (c + 1) / 2 for
    c <= 4. The harness is a closed loop (`--num-prompts 10c --max-concurrency c`).
    The burst factor (c + 1) / 2 implied over 100% of dense peak on 4 rows and a
    median of 1.020, so it was dropped;
  - keep the 4 rows with ratio above 1 (51-61% of dense peak, physical).
    Rejecting them would bias the median down.
- **D4 Fit at 90% in inference.** Inference fits when `total <= vram x 0.9`
  (`GPU_MEMORY_UTILIZATION`, helper `fitsInference`). Training stays at
  `total <= vram`, tiers literally 80/95. Inference tiers: comfortable <= 80%,
  tight <= 90%, else does not fit. The utilisation % shown stays total / vram.
  Recommendations measure the deficit, size GPU counts and pick upgrade cards
  against `vram x 0.9`. The fit bar and the multi-GPU bar mark the 90% line.
  Training draws no marker.
- **D5 KV tier.** The "vs recompute" comparison uses the ceiling prefill. That
  is the safe direction: recompute looks fastest, so the tier must beat it.
- **D6 KV fixes.** Mistral Large 3: 35,136 elements per token (sessions go up
  about 25x). Nemotron Ultra: 131,072 (it was 2.5x over).
- **D7 Sparse attention (DSA).** New optional model fields:
  `sparse_attention_topk`, `sparse_indexer_bytes_per_token`,
  `sparse_indexer_heads`, `sparse_indexer_layers`.
  - Decode reads min(ctx, topk) KV plus the full indexer cache.
  - Prefill attention FLOPs = 2 x L x H x a x (2T - a), a = min(T, topk): query
    t attends min(t, topk) keys. T^2 up to topk, 4 x L x H x topk x (T - topk/2)
    beyond. Plus indexer T^2 scoring at FP8. (2026-11-11, final review M1: was
    2 x L x T x min(T, topk) x H, which under-charged beyond topk.)
  - Memory: full KV plus the indexer cache per session. The indexer is
    replicated per TP rank.
- **D8 Which models.** GLM 5.2: topk 2048, 32 heads, 21 layers, 2772 B.
  Mistral Large 4: topk 2048, 32 heads, 44 layers, 5808 B.
- **D9 DeepSeek V4 is left on full-attention reads.** Its compressed KV and
  sliding window could not be modelled from vLLM's `deepseek_v4` backend with
  confidence. Full reads are the safe direction, per the spec source rule. Its
  long-context speed does not change. Commit 0d1f339 (Task 4) has "and DeepSeek
  V4" in its subject but adds no DeepSeek values. History is not rewritten.
- **D10 Mistral Large 4** (`mistralai-mistral-large-4-1t-a52b`): 1053.2B total,
  52.3B active. KV 36,288 = 63 KV-owning layers x 576. Nine layers share layer
  35's cache (vLLM kv_sharing plus the HF config), so the spec's 41,472 (72 x 576)
  was wrong and is corrected in the spec and plan. `weight_refs`: fp8 987.72,
  nvfp4 568.15, q8_0 1038.56 GiB. vLLM support exists only in the unmerged vLLM
  PR #64534: pending upstream vLLM support. Its DSpark draft head is counted in
  the weights, about 0.4% high. Known and minor.

## Consequences

- Headline tok/s drops to 0.47 of before; TTFT grows about 2.13x. The ceiling
  stays on screen beside it, in results, comparison and exports.
- Some configs that fit at 100% now read "does not fit" in inference.
- Mistral Large 3 and Nemotron Ultra session counts change (about 25x up, 2.5x up).
- GLM 5.2 and Mistral Large 4 long-context decode is faster. DeepSeek V4 is unchanged.

Open items:

- (Resolved by ADR 0010 gate 1, narrowed to DSA models.) The `fp8_ds_mla` KV layout is 656 B per token against the 576 charged, so FP8
  MLA KV is undercounted about 14%. This is the unsafe direction. Triage in a later piece.
  Sessions are 9-17% high when FP8 KV is selected on MLA models. This must be resolved,
  verified against vLLM source, before piece 2 divides targets by session counts.
- The indexer compute is divided by `kvShards`. That is exact only for MLA with
  1 KV head.
- One median over mixed models and GPUs hides per-config spread (decode p5 is 0.27).
- Batch-dependent decode efficiency. The AMD rows fall from about 0.63 at batch 1 to
  about 0.24 at batch 128. The flat median 0.47 is optimistic at the high concurrency
  that rack sizing uses. Resolved by ADR 0010 (bands).

## Sources

- `docs/calibration/2026-11-11-vllm-efficiency-points.md` (every point, accept or reject)
- vLLM `gpu_memory_utilization` default 0.9; vLLM `deepseek_v4`, `fp8_ds_mla` and kv_sharing code
- Audit items M1, M2, M4
