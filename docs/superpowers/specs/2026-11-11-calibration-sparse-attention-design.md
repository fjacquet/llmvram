# Calibration (expected vs ceiling), sparse attention, audit data fixes

**Date:** 2026-11-11
**Status:** Approved by the user 2026-11-11
**Series:** the piece between piece 1 (rack data, PR #78) and piece 2 (reverse rack solver).
Findings come from `docs/superpowers/specs/2026-11-11-engine-audit-findings.md`.
**Branch:** stacked on `feat/rack-precision-compute` until PR #78 merges, then rebased on main. **One PR for all of P1-P7** (user, 2026-11-11).

## Why

The user is a presales engineer. Neoclouds and enterprises hold them to tok/s, users served or
racks. The audit found the engine is a best-case roofline, 1.2x to 3x optimistic against published
measurements, so a rack count built on it would be 1.2x to 3x too small. The user has no lab, so
the correction must come from published runs. Piece 2 divides targets by these per-replica
figures, so they must be safe first.

Two more audit findings change sessions and speed: two models' KV cache is overstated (25x and
2.5x), and the frontier models now use sparse attention (DSA), which the engine prices as full
attention. Mistral Large 4 (1T, sparse attention, released 2026-09-30) is missing from the database.
The user asked to bring both into this piece ("Ajoute le, oui").

## Decisions (by the user)

- **Two figures (2026-11-11).** The headline is _expected_ = the engine's current figure divided by
  an efficiency ratio sourced from published vLLM runs, taking the pessimistic end when sources
  disagree. _Ceiling_ = today's figure, labelled "theoretical ceiling". Rack counts (piece 2) use
  expected.
- **Safe version (2026-11-10):** when figures conflict, take the safer one.
- **vLLM is the reference engine (2026-11-10).**
- **Mistral Large 4 and sparse attention are in this piece (2026-11-11).**

## Parts

### P1. Efficiency ratios (research first, then constants)

Two constants in `src/engines/constants.ts`, each with a source comment listing every data point:

- `DECODE_EFFICIENCY`: measured decode tok/s (or 1 / ITL × batch) ÷ the engine's decode tok/s
  for the same configuration.
- `PREFILL_EFFICIENCY`: the engine's prefill seconds ÷ measured TTFT at the same configuration
  (single request, or a batch whose queueing is known).

Rules for a data point:
- It must come from vLLM (any version from 0.8 on; older only if nothing newer exists, and say so).
- The GPU, model, weight format, TP/EP degree, batch or concurrency, and input/output lengths
  must be stated or derivable. A point missing any of these is listed as rejected, with the reason.
- Compute the ratio with the engine at the matched configuration, as the audit did (`npx tsx`
  script, not committed).
- **The ratio applied is the median** of the accepted points (user decision 2026-11-11, replacing "the pessimistic end"; points in `docs/calibration/2026-11-11-vllm-efficiency-points.md`).
- The audit's points are candidates: the lhl gist (vLLM 0.6.6, GPU unstated, so likely rejected),
  the Dell B200 TTFT point, and TensorRT-LLM (not vLLM: report it, don't use it).

**Too few points (the user wants everything in one PR, 2026-11-11):** the work doesn't wait for
data.
- **1 or 2 accepted vLLM points:** use the pessimistic end of those points, and label the ratio
  "provisional (n source[s])" in the constant's comment, ADR 0009 and the info tip.
- **0 accepted points:** the only case that stops the build. The builder reports what was found
  and the user chooses the value.

_Our call:_ this is the default the user saw when they approved the design.

_Our call:_ one global ratio per phase (decode, prefill), not one per GPU generation or model
size. Per-generation ratios come in only if the accepted points clearly differ by generation;
record the spread either way.

### P2. Engine outputs

`estimatePerformance` keeps every current field and meaning; those are now the _ceiling_. It adds
`expected: { tokensPerSecond, timeToFirstToken, prefillSeconds }`:
- both ratios are stored as efficiencies below 1 (_our call_, settled in review). Expected tok/s = ceiling × `DECODE_EFFICIENCY`; expected TTFT and prefill = ceiling ÷ `PREFILL_EFFICIENCY`. Expected is always slower.

`perUserTokensPerSecond` and `perUserTimeToFirstToken` get the expected inputs wherever a figure
is shown as the answer. Training is untouched.

_Our call:_ the ratios are not applied inside the roofline. They are one division at the output,
so the ceiling stays exactly today's numbers (all current tests keep passing), and changing a
ratio later moves only the expected figures.

### P3. Where the two figures show

Every place a tok/s, per-user tok/s, TTFT or prefill figure is shown:
- `VerdictBlock`, `PerformanceSection`, `CapacitySection` if it shows speed;
- the comparison view;
- the PDF and PowerPoint exports.

The expected figure is the headline. "Theoretical ceiling: X" sits smaller beside or under it.
Recommendations reason about memory only today, so this piece reaches them only through P4's fit rule; speed-based ranking belongs to piece 2. ADR 0005's freeze is lifted for this second figure
only; no new inputs.

One short info tip (existing `InfoTip`) explains: "Expected = ceiling × measured vLLM efficiency
(sources in ADR 0009)."

### P4. One rule for "fits" (audit M4)

"Does it fit" becomes `total <= vram_gb × GPU_MEMORY_UTILIZATION` (0.9, vLLM's default), the same
basis `maxConcurrentSessions` uses. **Inference mode only** (_our call_: `gpu_memory_utilization` is a vLLM serving setting, so training keeps `total <= vram_gb`). Every inference place that decides fit or utilisation colour uses one
helper. Verdicts near the limit flip from "fits" to "does not fit"; this is a breaking change
listed in the CHANGELOG. The utilisation percentage shown stays `total / vram_gb` (_our call_:
the user reads it as "share of the card"), with the 90% line marked where a bar exists.

#### P4b. KV tier recompute

`kvTierSummary`'s "vs recompute" comparison uses the _ceiling_ prefill (_our call_, the safe direction): a faster recompute makes the storage tier look less attractive, never more.

## P5. KV fixes (audit M1, M2)

- Mistral Large 3 675B: `kv_cache_elements_per_token` 35,136 (61 layers × (512 + 64), MLA;
  `params.json`).
- Nemotron Ultra 253B: 131,072 (64 attention layers of 162 × 2 × 8 KV heads × 128; `block_configs`
  in `config.json`).

Each gets a second source per the KV rule in CLAUDE.md, plus an entry in `models.test.ts` that
carries both sources.

### P6. Sparse attention (DSA) in the engine

New optional model fields, sourced from `config.json`:
- `sparse_attention_topk` (`index_topk`);
- `sparse_indexer_bytes_per_token`: the indexer's cache, summed over the layers that hold their
  own indexer. Bytes, because its dtype is FP8 plus scales; vLLM decides the real layout.

Engine effects:
- **Decode:** each step reads, per sequence, the KV of min(context, topk) tokens for the main
  attention, plus the indexer cache over the full context.
- **Prefill:** main attention FLOPs scale with T × min(T, topk) instead of T², plus the indexer's
  scoring FLOPs (T² × indexer heads × indexer head dim, at FP8 where vLLM runs it in FP8).
- **Memory:** KV stays the full context (every token is stored), plus the indexer cache per
  session.

Models it applies to: Mistral Large 4 (topk 2048; 44 of 72 layers hold their own indexer, the other
28 reuse one), GLM 5.2 (`glm_moe_dsa`, topk 2048), DeepSeek V4 Pro (topk 1024) and V4 Flash.
DeepSeek V4 also compresses KV and adds a sliding window (`compress_ratio`, SWA in vLLM's
`deepseek_v4` backend). Its _memory_ is already a measured value (`kv_cache_elements_per_token`),
and the builder must model its _reads_ from vLLM's code, not guess them.

**Source rule:** every sparse field and every formula cites vLLM's model or attention-backend code
(via Context7 or GitHub) plus the HF config. Where vLLM's layout can't be confirmed, the builder
keeps full-attention reads for that model (the safe direction) and lists it.

### P7. Mistral Large 4

Add `mistralai-mistral-large-4-1t-a52b` (id per the existing pattern):
- `num_parameters_billion` and `active_parameters_billion` from the safetensors summary or model
  card (~1T / 52B);
- 72 layers, MoE 128 experts with 4 active plus 1 shared, MLA `use_mla`;
- KV 72 × 576 = 41,472 elements per token, FP8 KV native;
- the sparse fields from P6; context 1,048,576;
- `weight_refs` measured by `npm run refresh:models -- --measure` for fp8 (native), nvfp4
  (`mistralai/Mistral-Large-4-1T-A52B-NVFP4`) and GGUF (unsloth, then ggml-org).
  Add it to the `fetch-models.ts` roster and re-sort `models.json` by name, using codepoint order
  (see the memory note on sort collation).

## Goals

1. Every speed figure the customer sees is the expected one, with its ceiling beside it, and
   every ratio traces to listed vLLM measurements.
2. Sessions and "fits" use vLLM's own memory limit.
3. Sparse-attention models are priced as vLLM runs them.
4. Mistral Large 4 is selectable with measured weights.

## Non-goals

- Piece 2 (reverse solver, replica split).
- Per-GPU-generation ratios unless the data demands them.
- Other new models (piece 3 roster refresh).
- Training, ZeRO-1/2 (audit: still wrong, out of scope).
- Engines other than vLLM.

## Anti-goals

- A ratio fitted to fewer than 3 sources without the user being told.
- Hiding the ceiling (the user wants both).
- Changing the ceiling numbers (they are today's numbers by design).

## Success criteria

- `npx vitest run`, `tsc -b --force`, `biome check`, `npm run build` green.
- Tests:
  - `expected = ceiling / ratio` for tok/s and `× ratio` for TTFT;
  - every existing ceiling pin unchanged by P1-P4 (P5 and P6 move the ceiling for Mistral Large 3, Nemotron Ultra, GLM 5.2 and DeepSeek V4 on purpose, each re-pin dated);
  - fit at 0.9 (a config at 0.95 utilisation reads "does not fit");
  - the M1/M2 KV values with two sources;
  - sparse decode reads at 128k equal to topk-bounded KV plus the indexer, for Mistral Large 4;
  - prefill attention FLOPs linear in T beyond topk;
  - Mistral Large 4 entry validated and sorted.
- ADR 0009 (calibration and sparse attention), CHANGELOG (breaking: headline speeds drop by the
  ratios; fit at 90%), README model count, in-app guide text for the two figures.

## Left to the builder

- Exact UI layout of the ceiling figure (smaller text beside or under, consistent everywhere).
- Where the fit helper lives.
- The indexer byte layout, once confirmed in vLLM.
