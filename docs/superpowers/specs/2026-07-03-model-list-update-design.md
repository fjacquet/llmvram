# Design: Current-Generation Model List Refresh (July 2026)

**Date:** 2026-07-03
**Status:** Draft — awaiting user review
**Scope:** Data-driven refresh of the model database — add the current-generation
flagship LLMs and retire clearly-obsolete entries. No engine changes.

## Goal

The curated list in `src/data/models.json` is a full generation behind the July-2026
open-weight landscape. This refresh:
1. Adds the current-gen flagships (Gemma 4, Qwen3.6, DeepSeek V4, GLM 5.2, Kimi K2
   Thinking / Kimi Linear, Mistral Medium 3.5, MiniMax M3).
2. Retires clearly-obsolete entries (2023–early-2024 models) while keeping recent
   previous-gen models for comparison.
3. Keeps every entry accurate (all config fields read from live HuggingFace
   `config.json` on 2026-07-03).
4. Refreshes the fetch pipeline (`scripts/fetch-models.ts` `MODEL_IDS`).

## Non-goals

- **No engine changes.** Several current flagships use exotic attention (MLA,
  DeepSeek Sparse Attention, lightning attention, linear attention, mamba-hybrid).
  They are added with **nominal** `num_kv_heads` from config, exactly matching how the
  existing DeepSeek R1 / Kimi K2 entries are stored. Accurate KV-cache math for these
  is a separate, later engine task.
- **No exhaustive sweep.** Curated flagships (~12 core + a few optional), not every
  size/variant.
- **No new schema fields.** All entries fit the existing `ModelSchema`.

## Context / constraints (verified in the codebase)

- `src/data/models.json` (currently 65 models) is the source of truth imported by the
  app (`ModelSelector.tsx`, `uiStore.ts`, `models.test.ts`). `models-fetched.json` is
  only scratch output of the fetch script.
- `scripts/fetch-models.ts` fetches `config.json` per repo; its param-count estimator
  is explicitly "rough" and mishandles MoE. So `num_parameters_billion` is hand-curated
  from the model's true safetensors parameter total (via HF repo overview). For MoE this
  is the **total** count (all experts must fit in VRAM).
- KV-cache engine (`src/engines/kv-cache.ts`) uses the standard GQA formula
  `numKVHeads = model.num_kv_heads ?? model.num_attention_heads`; no MLA/linear/sparse
  special-casing. Existing MLA entries already use nominal head counts and accept this.
- Entries must be **sorted alphabetically by `name`** after edits.
- `id` convention: `{org}-{model-slug-lowercased}` (e.g. `qwen-qwen3.6-27b`).
- Multimodal models (Gemma 4, Qwen3.6, MiniMax M3, Mistral 3) store the transformer
  fields under `text_config`; values below are from there.

## Models to add — 13 core

All fields below were read from each model's live `config.json` and repo overview on
2026-07-03. `hf_url` = `https://huggingface.co/<repo>`. For MoE, `num_parameters_billion`
is the TOTAL parameter count. Where a model has no dense `intermediate_size` (pure MoE),
`moe_intermediate_size` is stored in the `intermediate_size` field (noted per row); this
only affects the minor activation term, not weight VRAM.

### Gemma 4 (Google) — replaces Gemma 3

| Field | Gemma 4 31B | Gemma 4 12B | Gemma 4 26B A4B |
|---|---|---|---|
| id | google-gemma-4-31b-it | google-gemma-4-12b-it | google-gemma-4-26b-a4b-it |
| repo | google/gemma-4-31B-it | google/gemma-4-12B-it | google/gemma-4-26B-A4B-it |
| architecture | dense | dense | moe |
| num_parameters_billion | 32.7 | 12.0 | 26.5 |
| hidden_size | 5376 | 3840 | 2816 |
| num_hidden_layers | 60 | 48 | 30 |
| num_attention_heads | 32 | 16 | 16 |
| num_kv_heads | 16 | 8 | 8 |
| intermediate_size | 21504 | 15360 | 2112 (dense; moe_inter 704) |
| num_experts / per_token | — | — | 128 / 8 |
| context_length | 262144 | 262144 | 262144 |
| license | apache-2.0 | apache-2.0 | apache-2.0 |

### Qwen3.6 (Alibaba) — replaces Qwen3 / Qwen2.5

| Field | Qwen3.6 27B | Qwen3.6 35B A3B |
|---|---|---|
| id | qwen-qwen3.6-27b | qwen-qwen3.6-35b-a3b |
| repo | Qwen/Qwen3.6-27B | Qwen/Qwen3.6-35B-A3B |
| architecture | dense | moe |
| num_parameters_billion | 27.8 | 36.0 |
| hidden_size | 5120 | 2048 |
| num_hidden_layers | 64 | 40 |
| num_attention_heads | 24 | 16 |
| num_kv_heads | 4 | 2 |
| intermediate_size | 17408 | 512 (moe_inter; no dense FFN) |
| num_experts / per_token | — | 256 / 8 |
| context_length | 262144 | 262144 |
| license | apache-2.0 | apache-2.0 |

### DeepSeek V4 (MLA) — replaces DeepSeek R1

| Field | DeepSeek V4 Flash | DeepSeek V4 Pro |
|---|---|---|
| id | deepseek-v4-flash | deepseek-v4-pro |
| repo | deepseek-ai/DeepSeek-V4-Flash | deepseek-ai/DeepSeek-V4-Pro |
| architecture | moe | moe |
| num_parameters_billion | 158.1 | 861.6 |
| hidden_size | 4096 | 7168 |
| num_hidden_layers | 43 | 61 |
| num_attention_heads | 64 | 128 |
| num_kv_heads | 1 (MLA nominal) | 1 (MLA nominal) |
| intermediate_size | 2048 (moe_inter) | 3072 (moe_inter) |
| num_experts / per_token | 256 / 6 | 384 / 6 |
| context_length | 1048576 | 1048576 |
| license | MIT | MIT |

### GLM 5.2 (Z.ai, sparse attention) — replaces GLM 4.7

| Field | GLM 5.2 |
|---|---|
| id | zai-org-glm-5.2 |
| repo | zai-org/GLM-5.2 |
| architecture | moe |
| num_parameters_billion | 753.3 |
| hidden_size | 6144 |
| num_hidden_layers | 78 |
| num_attention_heads | 64 |
| num_kv_heads | 64 (nominal; DeepSeek Sparse Attention in reality) |
| intermediate_size | 12288 |
| num_experts / per_token | 256 / 8 |
| context_length | 1048576 |
| license | MIT |

### Kimi (Moonshot) — newer than the current Kimi K2 entries

| Field | Kimi K2 Thinking | Kimi Linear 48B A3B |
|---|---|---|
| id | moonshotai-kimi-k2-thinking | moonshotai-kimi-linear-48b-a3b |
| repo | moonshotai/Kimi-K2-Thinking | moonshotai/Kimi-Linear-48B-A3B-Instruct |
| architecture | moe | moe |
| num_parameters_billion | 1058.1 | 49.1 |
| hidden_size | 7168 | 2304 |
| num_hidden_layers | 61 | 27 |
| num_attention_heads | 64 | 32 |
| num_kv_heads | 64 (MLA nominal) | 32 (linear attn — big KV overestimate) |
| intermediate_size | 18432 | 9216 |
| num_experts / per_token | 384 / 8 | 256 / 8 |
| context_length | 262144 | 1048576 |
| license | other (Modified MIT) | MIT |

### Nemotron 3 Ultra (NVIDIA, mamba/MoE/attention hybrid) — completes the Nemotron 3 tier

The list already stores two `nemotron_h` hybrids (Nemotron 3 Nano 30B A3B, Super 120B
A12B) with nominal `num_kv_heads` and full layer counts. Ultra 550B follows the same
convention — the flagship sibling of the existing Super 120B.

| Field | Nemotron 3 Ultra 550B A55B |
|---|---|
| id | nvidia-nemotron-3-ultra-550b-a55b |
| repo | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 |
| architecture | moe |
| num_parameters_billion | 560.5 |
| hidden_size | 8192 |
| num_hidden_layers | 128 |
| num_attention_heads | 64 |
| num_kv_heads | 2 (nominal; ~3/4 of layers are Mamba, so KV-cache is overestimated — same as the existing Super 120B entry) |
| intermediate_size | 5120 |
| num_experts / per_token | 512 / 22 |
| context_length | 262144 |
| license | other (NVIDIA Open Model License) |

### Mistral & MiniMax

| Field | Mistral Medium 3.5 128B | MiniMax M3 |
|---|---|---|
| id | mistralai-mistral-medium-3.5-128b | minimax-m3 |
| repo | mistralai/Mistral-Medium-3.5-128B | MiniMaxAI/MiniMax-M3 |
| architecture | dense | moe |
| num_parameters_billion | 127.7 | 427.0 |
| hidden_size | 12288 | 6144 |
| num_hidden_layers | 88 | 60 |
| num_attention_heads | 96 | 64 |
| num_kv_heads | 8 | 4 (lightning attn hybrid) |
| intermediate_size | 28672 | 12288 |
| num_experts / per_token | — | 128 / 4 |
| context_length | 262144 | 1048576 |
| license | other | other |

## Optional additions (decide during review)

- **Phi-4 14.7B** (`microsoft/phi-4`) — the only current Phi (no Phi-5). Config to verify
  before adding.
- **Gemma 4 E4B** (`google/gemma-4-E4B-it`) — nano/on-device variant.
- **LiquidAI LFM2.5 230M** (`LiquidAI/LFM2.5-230M`) — tiny edge, conv-hybrid arch.

## Models to retire (clearly obsolete: 2023 – early 2024)

LLaMA 2 7B / 13B / 70B; Mistral 7B v0.1 / v0.3; Mixtral 8x7B v0.1 / 8x7B Instruct v0.1 /
8x22B v0.1; Gemma 2B; Gemma 7B; Gemma 2 9B; Phi 3 Mini 4K / Small 8K / Medium 4K;
Phi 3.5 Mini Instruct; MPT 30B; Falcon 40B; Yi 34B; LLaMA 3 8B; DeepSeek Coder 33B;
DeepSeek V2 Lite; DeepSeek Coder V2 Lite; Command-R; Command-R Plus;
Nemotron Llama 3.1 Nano 8B; Nemotron Llama 3.3 Super 49B (superseded by the Nemotron 3
Nano/Super entries). (26 entries.)

**Kept (recent previous-gen, still widely run):** Llama 3.1/3.2/3.3, Llama 4 Scout /
Maverick, Mistral Large 3 / Small 4, Devstral 2, Magistral, Ministral, Nemotron 3 Nano /
Super, Nemotron Llama Ultra 253B (dense large-model baseline; see decision 5), GPT-OSS
20B/120B, Kimi K2 Instruct / K2.5, MiniMax M2.1/2.5/2.7, DeepSeek R1 (kept as the
recognizable reasoning baseline).

**Net count:** 65 − 26 retired + 13 added = **52 models** (more if optional ones added).

## Implementation changes

1. **`src/data/models.json`** — remove the 26 retired entries, add the 13 core (+ any
   approved optional), then re-sort the whole array alphabetically by `name`.
   Also update `src/utils/models.test.ts`: its hard-coded family-presence tests assert
   LLaMA 2 / Mixtral / Command-R / Phi are present — those families are retired, so
   re-point them to current families and update the MoE-pitfall + GQA fixtures.
2. **`scripts/fetch-models.ts`** — replace obsolete repo ids in `MODEL_IDS` with the new
   current-gen repos, grouped by family with comments. Add a comment that multimodal
   models expose fields under `text_config` and gated/custom-code models still need manual
   curation (the script reads top-level config only).
3. **Verification** — the added entries are already verified against live config.json;
   record the source date in the CHANGELOG entry.
4. **`CHANGELOG.md`** — new version entry listing additions + retirements.
5. **`README.md`** — update the model-count figure/badge.
6. **Memory** — update `MEMORY.md` current counts and note the generational refresh.

## Testing / validation

- `npx vitest run src/utils/models.test.ts` — schema validity + alphabetical-sort
  invariant.
- `npm run typecheck` and `npm run lint` — must pass.
- Manual sanity: load the app, select a dense (Gemma 4 31B), a standard MoE
  (Qwen3.6 35B A3B), and an exotic-attention one (Kimi Linear), confirm the VRAM
  breakdown renders with no NaN/error.

## Open decisions for reviewer

1. **Optional models** — add any of Phi-4, Gemma 4 E4B, LiquidAI LFM2.5?
5. **Nemotron Llama Ultra 253B** — keep (dense large baseline, recommended) or retire now
   that Nemotron 3 Ultra 550B is in?
2. **Retirement list** — is retiring all 22 right, or keep any (e.g. Mixtral 8x7B, Llama 2
   70B, Phi-3) as recognizable baselines?
3. **DeepSeek R1** — keep (as proposed) or retire now that DeepSeek V4 is in?
4. **Exotic-attention entries** — OK to add now with nominal `kv_heads` (consistent with
   existing list), accepting approximate KV-cache, or hold the linear/mamba ones
   (Kimi Linear, Nemotron Ultra) until the engine supports them?
