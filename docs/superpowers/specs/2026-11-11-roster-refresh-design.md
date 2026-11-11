# Piece 3: model roster refresh (November 2026)

**Date:** 2026-11-11
**Status:** Approved by the user 2026-11-11 ("prend les derniers"; subagent-driven, Sonnet subagents)
**Series:** piece 3, after Sizing (PR #80). **Branch:** `feat/roster-refresh-2026-11`, stacked on `feat/sizing-mode`. The user merges everything at the end.

## Why

Sizing is only as good as its models. Seven recent releases are missing from `models.json` (55 entries), and customers ask about them.

## Decisions (by the user, 2026-11-11)

Take the latest release of each family. All six candidates were found on Hugging Face on 2026-11-11:

| Family              | Repo to add                            | Note                                                              |
| ------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| Kimi K3.1 Pro       | `moonshotai/Kimi-K3.1-Pro`             | `kimi_k31`, 8-bit compressed-tensors                              |
| DeepSeek V4.1 Flash | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | the later of two revisions (10 Sep, 17 Oct)                       |
| GLM 5.3             | `zai-org/GLM-5.3`                      | `glm_moe_dsa` (sparse attention)                                  |
| GLM 5.3 Flash       | `zai-org/GLM-5.3-Flash`                | `glm5_next`; added because the roster keeps both sizes of GLM 4.7 |
| Ling 3.1 Flash      | `inclusionAI/Ling-3.1-flash-1019`      | the later of two revisions; `bailing_hybrid`                      |
| ERNIE 5.5 Flash     | `baidu/ERNIE-5.5-Flash-Base`           | only a Base release exists; named "ERNIE 5.5 Flash Base"          |
| LongCat 2.5         | `meituan-longcat/LongCat-2.5`          | multimodal, `pro_v`, MoE                                          |

## Our calls

- **Add, never remove.** Ids are never renamed and a shared link may name an older model, so older entries stay. The roster grows from 55 to 62.
- **Ids follow the existing pattern** (lowercased `org-name`): for example `zai-org-glm-5.3`, `moonshotai-kimi-k3.1-pro`, `deepseek-ai-deepseek-v4.1-flash-1017`. Check the pattern in `models.json`.
- **Models whose architecture the engine cannot price are listed, not guessed.** Where a field cannot be sourced twice, the model is added without it only if the engine's safe direction applies (full-attention reads, full KV); otherwise the model is skipped and reported.

## Per-model rules (CLAUDE.md, "Domain Pitfalls" and "Key Patterns")

- Parameters (total and active), layers, heads, KV heads, hidden and intermediate sizes, experts and context come from the HF `config.json` and the safetensors summary. Multimodal repos nest the text fields in `text_config`.
- `kv_cache_elements_per_token`, `kv_sliding_elements_per_token` plus `kv_sliding_window`, and `linear_state_bytes_per_session` are known-good values: what vLLM allocates, confirmed by a second source (vLLM's model or cache-spec code, or a published figure). Never derive them by hand without one. Every value in `models.test.ts` carries both sources.
- MLA: `use_mla: true`, and the latent KV elements are per layer 512 + 64 for the DeepSeek-style layout. Check each model.
- Sparse attention (`index_topk` in config): the four `sparse_*` fields, sourced from vLLM's model code and the HF config, as for GLM 5.2 (ADR 0009). Where vLLM's layout cannot be confirmed, leave the fields off (full reads, the safe direction) and say so in the commit message.
- Weights are measured per format: `npm run refresh:models -- --measure <id>` for the native format and the usual references (RedHatAI or nvidia FP8/NVFP4, unsloth then ggml-org or bartowski GGUF, most-downloaded AWQ/GPTQ). MoE refs carry `high_precision` (`--split`).
- `models.json` is sorted by `name` in codepoint order, not `localeCompare`.
- Add each id to the roster in `scripts/fetch-models.ts`.

## Goals

1. Each added model is selectable, validates through Zod, and sizes correctly in Inference and Sizing.
2. Every KV value has two sources, listed in `models.test.ts`.
3. README, CHANGELOG and the counts say 62 models.

## Non-goals

- Removing older models. Engine changes for a new attention type (report and stop; the controller rules).
- Per-model efficiency calibration.

## Success criteria

- `npx vitest run`, `tsc -b --force`, `biome check .` and `npm run build` are green.
- `npm run refresh:models` reports no drift for the new entries.
- `models.test.ts` has an entry per new model with both sources, and `models.json` is sorted.

## Addition: publisher country of origin (user, 2026-11-11)

Some customers are sensitive to a model's origin and fear bias. The user decided: **display the origin, nothing more**. No filter and no statement about bias.
- New optional field `origin_country` (ISO 3166-1 alpha-2) on each model, filled by hand from the publisher's headquarters, with the source per publisher in `models.test.ts`.
- Shown as a code ("FR") with the country name on hover (_our call_: a code, since flag emoji render as letters on Windows).
- The Guide says what it is not: the publisher's headquarters, not where a model was trained or what data it saw, and not a measure of bias.
