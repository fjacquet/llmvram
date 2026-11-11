# Roster Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add the seven latest models of six families to `models.json` (55 → 62) with measured weights and two-source KV values.

**Architecture:** data work through the existing audit tooling (`scripts/fetch-models.ts`, `scripts/model-audit.ts`, `scripts/hf.ts`). Each model is one task. No engine change is planned; a model needing one is reported, not built.

**Tech Stack:** TypeScript, Zod (`src/utils/schemas.ts`), Vitest, `tsx` scripts, Hugging Face, vLLM source (Context7 or GitHub).

**Spec:** `docs/superpowers/specs/2026-11-11-roster-refresh-design.md`. Approved 2026-11-11. Read it; it is the authority.

## Global Constraints

- **Branch:** `feat/roster-refresh-2026-11`, in the main checkout. Never use a worktree. Stacked on `feat/sizing-mode`.
- **Verify values before writing them** (global rule): Hugging Face `config.json` and the safetensors summary, plus vLLM's model code via Context7 or GitHub. Cite sources in `models.test.ts`.
- **Per-model rules:** the spec's "Per-model rules" section applies to every task.
- **Add, never remove.** Ids never change. `models.json` stays sorted by `name` in codepoint order (`a < b` on code units, not `localeCompare`). Re-sort after each insert.
- **Never write `models.json` from a script.** `refresh:models` prints; you paste or edit by hand, then re-sort.
- **KV values:** `kv_cache_elements_per_token`, `kv_sliding_*` and `linear_state_bytes_per_session` need two sources each, in `models.test.ts`.
- **Safe direction:** where a layout cannot be confirmed, charge full-attention reads and full KV, and say so.
- **Stop rule:** if a model needs an engine change (a new attention type the fields cannot express), do not build it. Report NEEDS_CONTEXT with the evidence.
- **Biome:** 2-space indent, single quotes, no semicolons, 100 columns, no `!`.
- **Commits:** write each message to a file and use `git commit -F`. End every message with:
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_0148C11Vkt7xNbSVsEoYPcmB`
- **Staging:** stage explicit paths only. Never stage `.serena/project.yml`.
- **Verify** with `npx vitest run`, `./node_modules/.bin/tsc -b --force`, `npx @biomejs/biome check .` and `npm run build`. Use direct binaries, not rtk.

## Review Focus

1. **A multimodal repo** (Kimi K3.1 Pro, GLM 5.3 Flash, DeepSeek V4.1, LongCat 2.5) whose transformer fields sit under `text_config`: the parameter and head counts must come from the text model, not the vision tower.
2. **MoE active parameters** must be the routed-plus-shared per-token count, and `moeWeightSplit` must work with the measured `high_precision`.
3. **Context length** stated in config versus what vLLM serves.
4. **A model with sliding or linear layers** (Ling hybrid, GLM 5.3 Flash): sessions must not be overstated.
5. **A model with no instruct or non-Base release** (ERNIE 5.5): the name says "Base".

---

## The model-entry procedure (used by Tasks 1-6)

Each model task follows these steps. The task text gives the repo, id, name and what to watch for.

- [ ] **Step 1: Recon.** Read `config.json` (and `text_config`), the safetensors summary and the model card. Use `scripts/hf.ts` helpers or fetch the raw files. Record parameters (total and active), architecture, layers, heads, KV heads, sizes, experts, context, license.
- [ ] **Step 2: KV.** Work out `kv_cache_elements_per_token` (and the sliding or linear-state fields) from vLLM's model implementation and cache spec. Find a second source: the HF cache shapes in transformers, or a published figure. If sparse attention is configured, source the four `sparse_*` fields as in ADR 0009 (GLM 5.2 is the pattern). If the two sources disagree, report NEEDS_CONTEXT.
- [ ] **Step 3: Weights.** Run `npm run refresh:models -- --measure <id>` after adding a draft entry, and `--split` for the MoE `high_precision` values. Add refs for the native format and each reference format that exists.
- [ ] **Step 4: Entry.** Add the entry to `src/data/models.json`, re-sort, add the id to `MODEL_IDS` in `scripts/fetch-models.ts`, and add the `models.test.ts` entry with both KV sources.
- [ ] **Step 5: Tests.** `models.test.ts` (KV with both sources, ordering) and a sizing sanity check: the model sizes on one server GPU and one rack GPU without NaN. Run the full suite, `tsc -b --force`, `biome check .` and `npm run build`.
- [ ] **Step 6: Commit** `feat(data): add <Model name>`, with the source URLs in the body.

### Task 1: GLM 5.3 and GLM 5.3 Flash

**Repos:** `zai-org/GLM-5.3` (`glm_moe_dsa`, sparse attention) and `zai-org/GLM-5.3-Flash` (`glm5_next`).
**Ids:** `zai-org-glm-5.3` and `zai-org-glm-5.3-flash`. **Names:** "GLM 5.3" and "GLM 5.3 Flash".
**Watch for:** GLM 5.3 follows the GLM 5.2 pattern, so copy its sparse-field sourcing. For FP8 KV, `kvElementBytes` already charges 656 B per token per layer for models with `use_mla` and `sparse_attention_topk`. GLM 5.3 Flash is a new architecture; if it is hybrid (linear or sliding layers), use the sliding or linear-state fields and apply the stop rule if they cannot express it.

### Task 2: Kimi K3.1 Pro

**Repo:** `moonshotai/Kimi-K3.1-Pro` (`kimi_k31`, 8-bit compressed-tensors, multimodal). **Id:** `moonshotai-kimi-k3.1-pro`. **Name:** "Kimi K3.1 Pro".
**Watch for:** compare with the existing `moonshotai-kimi-k3` entry (2779.9B) for what changed. The text config may be under `text_config`. Check MLA, expert counts and context.

### Task 3: DeepSeek V4.1 Flash

**Repo:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`deepseek_v41`, multimodal, fp8). **Id:** `deepseek-ai-deepseek-v4.1-flash-1017`. **Name:** "DeepSeek V4.1 Flash".
**Watch for:** the existing `deepseek-v4-flash` entry (284B) deliberately has no sparse fields, because its compressed KV and sliding window could not be confirmed against vLLM's `deepseek_v4` backend (ADR 0009). Check whether V4.1 changes that layout. Keep full-attention reads unless vLLM's source confirms the layout.

### Task 4: Ling 3.1 Flash

**Repo:** `inclusionAI/Ling-3.1-flash-1019` (`bailing_hybrid`, fp8, custom code). **Id:** `inclusionai-ling-3.1-flash`. **Name:** "Ling 3.1 Flash".
**Watch for:** the existing `inclusionai-ling-3.0-flash` (127.5B) shows the pattern. The hybrid layer mix (full vs linear or sliding layers) drives KV; use the HF config for the layer types and vLLM's `bailing` model code. The DSpark draft repo is a separate model, not part of the weights.

### Task 5: ERNIE 5.5 Flash Base

**Repo:** `baidu/ERNIE-5.5-Flash-Base` (`ernie5_5_moe`, fp8; also a `-BF16` repo). **Id:** `baidu-ernie-5.5-flash-base`. **Name:** "ERNIE 5.5 Flash Base".
**Watch for:** only a Base release exists; check again for a non-Base repo before settling. License is "other"; record it. Use the BF16 repo as the bf16 weight ref if it exists.

### Task 6: LongCat 2.5

**Repo:** `meituan-longcat/LongCat-2.5` (`pro_v`, multimodal MoE, custom code; a `-W4A4` repo exists). **Id:** `meituan-longcat-longcat-2.5`. **Name:** "LongCat 2.5".
**Watch for:** `pro_v` is a new architecture. If vLLM has no model class for it, the KV cannot be sourced from vLLM; use the HF config plus a published figure, and apply the stop rule if two sources are not available.

### Task 7: Docs and counts

**Files:** `CHANGELOG.md` (`## [Unreleased]`, Added), `README.md` (model count 62), `CLAUDE.md` and `MEMORY` counts (the controller updates the memory file), `docs/adr/README.md` only if an ADR is warranted (none planned).

- [ ] **Step 1:** CHANGELOG lists the added models, with the sources of the notable KV decisions, and any model skipped with the reason.
- [ ] **Step 2:** README count and any model list.
- [ ] **Step 3:** Full verification; put the counts in the commit body. Commit `docs: roster refresh, 62 models`.
