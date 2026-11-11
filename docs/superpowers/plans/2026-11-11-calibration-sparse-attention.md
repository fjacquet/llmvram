# Calibration, Sparse Attention and Audit Data Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every speed figure shows an _expected_ value (calibrated on published vLLM runs) with the _theoretical ceiling_ beside it. "Fits" uses vLLM's 90% memory limit. Sparse-attention models are priced as vLLM runs them. Two KV errors are fixed and Mistral Large 4 is added.

**Architecture:**
- Today's engine numbers become the _ceiling_, unchanged.
- `estimatePerformance` adds an `expected` block: ceiling × `DECODE_EFFICIENCY` for tok/s, ceiling ÷ `PREFILL_EFFICIENCY` for times.
- Sparse attention is two new optional model fields that bound attention reads and FLOPs to `topk` tokens and add the indexer cache.
- One inference fit helper replaces the scattered `total > vram` checks.

**Tech Stack:** TypeScript strict, Zod, decimal.js, React 19, Vitest, Biome 2.6.

**Spec:** `docs/superpowers/specs/2026-11-11-calibration-sparse-attention-design.md` (P1-P7). Audit evidence is in `docs/superpowers/specs/2026-11-11-engine-audit-findings.md`.

## Global Constraints

- Branch `feat/calibration-sparse-attention` (main checkout, not a worktree; stacked on PR #78). One PR for everything.
- KV, sparse and state values are known-good values from config plus a second source, never derived by hand without one. Every value in `src/utils/models.test.ts` carries its source (CLAUDE.md).
- `src/data/models.json` is sorted by `name` in **codepoint** order, not `localeCompare`.
- Never hand-edit `src/data/gpus.json`.
- Ceiling numbers are not changed by P1-P4. P5 and P6 change them only for Mistral Large 3, Nemotron Ultra, GLM 5.2, DeepSeek V4 Pro/Flash and Mistral Large 4, and each such re-pin carries a dated reason comment.
- Efficiencies are stored below 1. Expected tok/s = ceiling × `DECODE_EFFICIENCY`; expected TTFT/prefill = ceiling ÷ `PREFILL_EFFICIENCY`. Expected is never faster than the ceiling.
- The fit rule is `total <= vram_gb × GPU_MEMORY_UTILIZATION` (0.9) in **inference only**. Training keeps today's rule.
- The KV-tier "vs recompute" figure uses the **ceiling** prefill.
- vLLM is the reference engine. A calibration point counts only from vLLM ≥ 0.8 (older only if nothing newer exists, flagged as such) with GPU, model, format, TP/EP, batch/concurrency and lengths stated.
- **Calibration rules:**
  - 1-2 accepted points: use the pessimistic end and label it "provisional (n source[s])".
  - 0 accepted points: **stop and report to the user**.
- Biome: 2-space, single quotes, no semicolons, 100 columns, no `!`. Commits end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_0148C11Vkt7xNbSVsEoYPcmB`
- Stage explicit paths only. `.serena/project.yml` is the user's own uncommitted edit: never stage it.
- Verify with `npx vitest run`, `./node_modules/.bin/tsc -b --force`, `npx @biomejs/biome check .` and `npm run build`. Use direct binaries: the rtk hook can fabricate wrapped output.

## Review Focus

1. **A model with no sparse fields** must give exactly today's memory, decode and prefill. Pinned in Task 3.
2. **A sparse model whose context is ≤ topk** must equal the full-attention result, apart from the indexer cache. Pinned in Task 3.
3. **Training mode** keeps today's fit verdict and gauge tiers at the same totals. Pinned in Task 2.
4. **Offloaded inference** applies the 0.9 rule to the on-device share, and the host-capacity check is unchanged. Pinned in Task 2.
5. **A GPU with no FLOPS data** (degraded prefill) yields expected TTFT = ceiling TTFT ÷ `PREFILL_EFFICIENCY` and expected prefill `null`, never NaN. Pinned in Task 6.

---

### Task 1: KV fixes for Mistral Large 3 and Nemotron Ultra (P5)

**Files:**
- Modify: `src/data/models.json` (2 entries)
- Test: `src/utils/models.test.ts` (the `kv_cache_elements_per_token` block ~line 167 and its verified-model list ~line 224)

**Interfaces:**
- Produces: `mistralai-mistral-large-3-675b.kv_cache_elements_per_token = 35136` and `nvidia-llama-3.1-nemotron-ultra-253b.kv_cache_elements_per_token = 131072`.

- [ ] **Step 1: Write the failing tests.** Add both ids to the verified list with value and two sources each:
  - Mistral Large 3: 61 layers × (kv_lora_rank 512 + qk_rope_head_dim 64), MLA, per `params.json` at https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512. Second source: vLLM's MLA cache layout, `kv_lora_rank + qk_rope_head_dim` per layer, via Context7 `/websites/vllm_ai_en_stable`.
  - Nemotron Ultra: 64 attention layers (162 minus `no_op`/`replace_with_linear`) × 2 × 8 KV heads (128 / `n_heads_in_group` 16) × head_dim 128, per `block_configs` in https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1/resolve/main/config.json. Second source: vLLM's `DeciLM`/`nemotron_nas` model code, per-layer KV only where attention exists.
- [ ] **Step 2:** `npx vitest run src/utils/models.test.ts` FAILS (field absent).
- [ ] **Step 3:** Add the two fields to `models.json`.
- [ ] **Step 4:** Tests pass. Then run the full suite and re-pin any test that used these two models, each with `// 2026-11-11: KV from config (audit M1/M2)`.
- [ ] **Step 5:** Commit: `fix(models): known-good KV for Mistral Large 3 (MLA) and Nemotron Ultra (64 attention layers)`.

### Task 2: One inference fit rule at vLLM's 90% (P4)

**Files:**
- Modify: `src/engines/concurrency.ts`, `src/components/layout/ResultsPanel.tsx:~225-245`, `src/components/outputs/FitIndicator.tsx`, `src/components/comparison/ComparisonColumn.tsx:~118`
- Test: `src/engines/concurrency.test.ts`, `src/components/outputs/FitIndicator.test.tsx` (create if absent), the existing `ResultsPanel` tests

**Interfaces:**
- Produces:
  - `export function fitsInference(totalGB: number, vramGB: number): boolean` in `concurrency.ts` (`totalGB <= vramGB * GPU_MEMORY_UTILIZATION`).
  - `FitIndicator` gets an optional prop `limitPercent?: number`, default 100.
  - Inference callers pass `GPU_MEMORY_UTILIZATION * 100` (90); training passes nothing.
  - Gauge tiers: "Fits Comfortably" ≤ `limitPercent − 10`, "Tight Fit" ≤ `limitPercent`, else "Does Not Fit". At the 100 default that keeps a tight fit up to 100, so **check** that this preserves today's training tiers (≤80 / ≤95). If it doesn't, keep the training tiers literally as today and apply the new tiers only when `limitPercent` is given.

- [ ] **Step 1: Write the failing tests:**
  - `fitsInference(250, 278)` is true; `fitsInference(251, 278)` is false (278 × 0.9 = 250.2).
  - FitIndicator with `limitPercent={90}` at 92% reads "Does Not Fit".
  - Training FitIndicator at 92% keeps today's label.
  - ResultsPanel inference on a configuration at 95% utilisation reads "does not fit".
  - Offloaded inference applies the rule to `onDevice.total`.
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Replace the three `greaterThan(selectedGPU.vram_gb)` device checks in ResultsPanel's inference path with `!fitsInference(...)`. Pass `limitPercent` to inference FitIndicators. In ComparisonColumn, colour against 90 for inference snapshots (the snapshot already carries `fits`). Leave the host-capacity check alone.
- [ ] **Step 4:** Tests pass, then the full suite. Re-pin verdict tests that flip, with `// 2026-11-11: fit at vLLM 0.9 (audit M4)`.
- [ ] **Step 5:** Commit: `fix(fit): inference fits at vLLM gpu_memory_utilization 0.9, same rule as sessions`.

### Task 3: Sparse attention in the engine (P6, engine only)

**Files:**
- Modify: `src/utils/schemas.ts` (ModelSchema), `src/engines/kv-cache.ts`, `src/engines/performance.ts:~252` (decode attention term) and `:~356` (prefill attention FLOPs), plus the decode KV-bytes read (step 1 of `estimatePerformance`)
- Test: `src/engines/kv-cache.test.ts`, `src/engines/performance.test.ts`

**Interfaces:**
- Produces:
  - `Model.sparse_attention_topk?: number` (int, positive).
  - `Model.sparse_indexer_bytes_per_token?: number` (int, positive; summed over the layers that own an indexer). Zod refine: the two are set together.
  - `export function calculateSparseIndexerVRAM(model: Model, sequenceLength: number, sessions: number): Decimal` in `kv-cache.ts` (GiB). It returns 0 without the fields, and is added to the session KV total wherever linear state is added today.
  - `export function attendedTokens(model: Model, contextTokens: number): number`: `min(context, topk)`, or `context` without sparse fields.

**Engine rules (spec P6):**
- **Decode KV read per sequence:** the main KV at `attendedTokens(context)` tokens, plus the full indexer cache (`sparse_indexer_bytes_per_token × context`).
- **Decode attention FLOPs per token:** `4 × layers × attendedTokens(context) × hidden`, plus indexer scoring. Indexer scoring is `2 × context × indexer_heads × indexer_head_dim` per indexer layer, and it needs two more optional fields: `sparse_indexer_heads` and `sparse_indexer_layers`. Before coding, check vLLM's DSA indexer (Context7 `/websites/vllm_ai_en_stable`, or GitHub) for whether it scores the full context. Add the two fields only if it does; otherwise leave indexer FLOPs out and say so in a comment. Task 4 relies on your finding.
- **Prefill attention FLOPs:** replace `2 × layers × T² × hidden` with `2 × layers × T × attendedTokens(T) × hidden`. That is exact when T ≤ topk and linear in T beyond it.

- [ ] **Step 1: Write the failing tests.** Use a fixture built from a real model row (spread `deepseek-v4-pro` or `zai-org-glm-5.2` plus fields), never a literal:
  - A model without sparse fields gives the same memory, prefillSeconds and tokensPerSecond as before the change (`toBeCloseTo` 9). This is Review Focus 1.
  - At context 1024 with topk 2048, results equal full attention, except that memory and decode reads include the indexer cache. This is Review Focus 2.
  - At 128k with topk 2048, prefill attention FLOPs are `2 × L × T × 2048 × hidden`, and doubling T doubles them.
  - At 128k, decode KV bytes per sequence equal the KV at 2048 tokens plus `indexer_bytes × 131072`.
  - The schema rejects topk without indexer bytes.
- [ ] **Step 2:** Run the tests and confirm they fail.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Tests pass, then the full suite (no data uses the fields yet, so nothing else moves).
- [ ] **Step 5:** Commit: `feat(engine): sparse attention (DSA) bounds attention reads and FLOPs to topk, counts the indexer cache`.

### Task 4: Sparse values for GLM 5.2 and DeepSeek V4 Pro/Flash (P6, data)

**Files:**
- Modify: `src/data/models.json` (3 entries)
- Test: `src/utils/models.test.ts` (a new sparse block, sources per value), re-pins elsewhere

**Interfaces:**
- Consumes: Task 3 fields.

- [ ] **Step 1: Research, before code.** For each model, take `index_topk`, the indexer heads and head dim, and which layers own an indexer, from `config.json`:
  - GLM 5.2 `glm_moe_dsa`: topk 2048;
  - DeepSeek V4 Pro: topk 1024;
  - DeepSeek V4 Flash: its config.

  Then take the indexer cache bytes per token per layer from vLLM's code: the indexer KV dtype (FP8 plus scale) and the `deepseek_v4` sparse backend (compress_ratio, SWA). Use Context7 `/websites/vllm_ai_en_stable` (at most 3 queries per question) or GitHub `vllm-project/vllm`.
  - **DeepSeek V4:** if its reads (compressed KV + SWA + topk) cannot be confirmed in vLLM's code, leave its sparse fields absent. Full reads are the safe direction. List it in the commit and report.
  - **Indexer FLOPs:** confirm whether the indexer scores the full context (this decides Task 3's indexer-FLOPs fields).
- [ ] **Step 2: Write the failing tests** with the found values and their two sources.
- [ ] **Step 3:** Add the fields.
- [ ] **Step 4:** The full suite. Re-pin moved tests for these models with `// 2026-11-11: sparse attention (DSA)`.
- [ ] **Step 5:** Commit: `feat(models): sparse attention values for GLM 5.2 and DeepSeek V4`.

### Task 5: Mistral Large 4 (P7)

**Files:**
- Modify: `scripts/fetch-models.ts` (roster), `src/data/models.json` (new entry, re-sorted), `src/utils/models.test.ts`

**Interfaces:**
- Produces: model id `mistralai-mistral-large-4-1t-a52b`.

- [ ] **Step 1: Write the failing tests:**
  - The entry validates.
  - `kv_cache_elements_per_token` is 41,472 (72 × (512 + 64)). Sources: `params.json` `kv_lora_rank`/`qk_rope_head_dim`, plus the vLLM MLA layout.
  - `sparse_attention_topk` is 2048, and the indexer is owned by 44 of the 72 layers (`indexer_types` "full" count). Indexer bytes per Task 4's vLLM layout.
  - `use_mla` is true; MoE has 128 experts, 4 active, 1 shared.
  - `context_length` is 1,048,576.
  - Its `weight_refs` include fp8 and nvfp4.
  - The array stays sorted by name in codepoint order.
- [ ] **Step 2:** Run and confirm the failures.
- [ ] **Step 3: Add the entry.**
  - Parameter counts: from the model card or the safetensors summary. If they are packed, use `refresh:models --measure` output, not the raw total.
  - `weight_refs`: `npx tsx scripts/fetch-models.ts --measure mistralai-mistral-large-4-1t-a52b` for fp8 (native `mistralai/Mistral-Large-4-1T-A52B`), nvfp4 (`mistralai/Mistral-Large-4-1T-A52B-NVFP4`) and GGUF (`unsloth/Mistral-Large-4-1T-A52B-GGUF`, else ggml-org). Add `high_precision` via `--split` if the script supports it for this model.
- [ ] **Step 4:** Tests pass; the full suite.
- [ ] **Step 5:** Commit: `feat(models): add Mistral Large 4 1T-A52B (MLA, sparse attention, FP8/NVFP4/GGUF measured)`.

### Task 6: Calibration ratios and expected outputs (P1, P2)

**Files:**
- Modify: `src/engines/constants.ts`, `src/engines/types.ts` (`PerformanceEstimate`), `src/engines/performance.ts`, `src/workers/calculation.worker.ts:~75,~263` and `src/hooks/useInferenceCalculation.ts:~77,~88` (serialise/deserialise)
- Test: `src/engines/performance.test.ts`, the worker/hook tests if present

**Interfaces:**
- Produces:
  - `DECODE_EFFICIENCY: Decimal` and `PREFILL_EFFICIENCY: Decimal` (both in (0, 1]) in `constants.ts`.
  - `PerformanceEstimate.expected: { tokensPerSecond: Decimal; timeToFirstToken: Decimal; prefillSeconds: Decimal | null }`.

- [ ] **Step 1: Research, before code.** Collect published vLLM measurements and compute engine ÷ measured at matched configurations with a throwaway `scripts/.calib-*.ts` script. Delete it after; never commit it. Search InferenceX, vLLM blog and releases, NVIDIA and AMD developer blogs, Red Hat, cloud-provider benchmarks, and arXiv.
  - Record every candidate in the report as accepted or rejected, with the reason.
  - Decode: efficiency = measured tok/s ÷ engine tok/s. Prefill: engine prefill ÷ measured TTFT.
  - Take the minimum of the accepted points.
  - **0 accepted points for either phase: STOP and report BLOCKED with the candidates.** The user chooses.
  - 1-2 points: "provisional (n source[s])" in the constant's comment.
- [ ] **Step 2: Write the failing tests:**
  - `expected.tokensPerSecond = ceiling × DECODE_EFFICIENCY`.
  - `expected.timeToFirstToken = ceiling ÷ PREFILL_EFFICIENCY`.
  - `expected.prefillSeconds = ceiling ÷ PREFILL_EFFICIENCY`, or null when the ceiling is null (Review Focus 5).
  - Expected is never faster than the ceiling.
  - Every ceiling field is unchanged (the existing pins).
  - Worker round-trip preserves `expected`.
- [ ] **Step 3:** Implement. Apply one division or multiplication at the end of `estimatePerformance`; the roofline stays untouched. Each constant's comment lists every accepted point (source URL, config, measured, engine, ratio) and states "pessimistic end of n points".
- [ ] **Step 4:** Tests pass; the full suite.
- [ ] **Step 5:** Commit: `feat(performance): expected tok/s and TTFT from published vLLM efficiency, ceiling unchanged`.

### Task 7: Show expected with the ceiling beside it (P3, P4b)

**Files:**
- Modify: `src/components/outputs/VerdictBlock.tsx:~30,~38`, `src/components/outputs/PerformanceSection.tsx:~38-110`, `src/store/comparisonStore.ts:~36` (add `ceilingTokensPerSecond`, `ceilingTimeToFirstToken`), `src/components/layout/ResultsPanel.tsx:~316,~649`, `src/components/comparison/ComparisonColumn.tsx:~310-315`, `src/utils/exportPptx.ts:~474-500,~579`, `src/components/guide/GuidePage.tsx` (one paragraph)
- Test: the existing component tests for these files, plus `exportPptx` tests

**Interfaces:**
- Consumes: `PerformanceEstimate.expected` (Task 6).

- [ ] **Step 1: Write the failing tests:**
  - The verdict shows expected tok/s as the headline and "Theoretical ceiling: <ceiling> tokens/sec" as secondary text. Same for TTFT.
  - Per-user tok/s and TTFT use expected inputs.
  - The PPTX rows "Decode Speed" and TTFT carry expected plus "(ceiling X)".
  - The comparison column shows expected, and the diff map compares expected.
  - CapacitySection's "vs recompute" still passes the **ceiling** `prefillSeconds` (P4b; ResultsPanel ~266 unchanged).
- [ ] **Step 2:** Run and confirm the failures.
- [ ] **Step 3:** Implement. Use one existing `InfoTip` next to the expected figure: "Expected = theoretical ceiling × measured vLLM efficiency (sources: ADR 0009)". Add the word "provisional" if Task 6 used fewer than 3 points. The guide paragraph explains the two figures. The PDF export captures the DOM, so it needs no extra change; check it expands correctly.
- [ ] **Step 4:** Tests pass; the full suite; `npm run build`.
- [ ] **Step 5:** Commit: `feat(ui): expected speed headline with theoretical ceiling beside it, in results, comparison and exports`.

### Task 8: Docs

**Files:**
- Create: `docs/adr/0009-calibrated-expected-speed-and-sparse-attention.md` (plus a row in `docs/adr/README.md`)
- Modify: `CLAUDE.md` (Key Patterns: expected vs ceiling, fit at 0.9 in inference, sparse fields), `CHANGELOG.md` `## [Unreleased]`, `README.md` (model count 54 → 55, "two figures"), `ARCHITECTURE.md` (performance outputs), memory `MEMORY.md` counts line (outside the repo)

- [ ] **Step 1: ADR 0009.** Context: the audit headline. Decisions: two figures, efficiency direction, every calibration point accepted or rejected, fit at 0.9 inference-only, KV-tier recompute on ceiling, the sparse attention rules, and which models have sparse values (and why DeepSeek V4 doesn't, if Task 4 left it out).
- [ ] **Step 2: CHANGELOG.** Breaking:
  - headline speeds drop by the efficiencies (give the numbers);
  - fit at 90%;
  - Mistral Large 3 and Nemotron Ultra sessions up about 25× and 2.5×;
  - GLM 5.2 and DeepSeek V4 long-context speed changes.

  Added: Mistral Large 4, sparse attention, expected vs ceiling.
- [ ] **Step 3:** The full verification set (vitest, tsc, biome, build). Paste the counts in the commit body.
- [ ] **Step 4:** Commit: `docs: ADR 0009, CHANGELOG, README, CLAUDE.md for calibration and sparse attention`.
