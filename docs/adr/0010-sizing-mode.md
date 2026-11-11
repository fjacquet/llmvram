# 0010. Sizing mode, decode efficiency by load band, and FP8 DSA KV

**Status:** Accepted, 2026-11-11
**Spec:** `docs/superpowers/specs/2026-11-11-rack-sizing-mode-design.md` (including the amendment "KV storage tier in Sizing")
**Points:** `docs/calibration/2026-11-11-vllm-efficiency-points.md`

## Context

Presales needs the inverse question: given a model and N simultaneous users, how
many racks or servers? ADR 0009 left two items open before that could be built:
the FP8 MLA KV size and a flat decode efficiency that is optimistic at the
concurrency rack sizing uses.

## Decision

- **D1 User decisions.**
  - Mode C: a third mode, Sizing, beside Inference and Training.
  - Rack GPUs (`isRackGPU`: `max_gpus_per_node` > 8) are selectable only in Sizing.
  - Users = simultaneous sessions.
  - No speed floor: the result shows the expected speed, with its ceiling, and does not size to a target.
  - Decode efficiency by load band (D4).
- **D2 Gate 1, `fp8_ds_mla`.**
  - vLLM charges 656 B per token per layer: 512 FP8 + 4 x 4 B float32 scales + 64 BF16 RoPE. Source: `vllm/v1/attention/backends/mla/flashmla_sparse.py`.
  - It applies only to DSA (sparse MLA) models. `vllm/model_executor/models/config.py`, `verify_and_update_config`, maps fp8 to `fp8_ds_mla` only when `dsa_config` is set. Here that is `use_mla` plus `sparse_attention_topk`: GLM 5.2 and Mistral Large 4.
  - Other MLA models (DeepSeek R1, Kimi, Mistral Large 3) stay at 1 B per element (576 B per layer).
  - Exception: on SM10x with FlashInfer sparse, vLLM may use per-tensor FP8 at 576. We charge 656, the safe direction.
  - ADR 0009's open item "fp8_ds_mla, all MLA, 9-17%" is resolved, and narrowed to the DSA models.
- **D3 Gate 2, Recommendations.** GPU counts round up to the next valid TP degree (`validTPDegrees`). R1 Distill 70B (64 heads) at 32k on H100 gets 4x, not 3x.
- **D4 Gate 3, decode efficiency by band** (user decision, 2026-11-11), in every mode:

  | Batch  | Efficiency        | Source median (n)                 |
  | ------ | ----------------- | --------------------------------- |
  | <= 32  | 0.47              | 2-8 0.481 (203), 9-32 0.472 (188) |
  | 33-128 | 0.425             | 0.425 (103)                       |
  | > 128  | 0.25, provisional | 0.252 (1)                         |

  `PREFILL_EFFICIENCY` 0.469 is unchanged. The last band rests on one point and
  moves when more are measured.
- **D5 Solver** (`sizeDeployment`, composes the existing engines).
  - Candidates are `validTPDegrees` only.
  - Servers: tp <= `max_gpus_per_node` with 1 node, plus a full-server tp x nodes 2..8 (pipeline stages). Racks: TP only within 72 GPUs.
  - Fit at 1 session via `fitsInference`. Sessions per replica come from `maxConcurrentSessions`, capped at users. Speed is taken at batch = sessions.
  - Units: a rack or server holds `floor(max / tp)` replicas; a multi-server replica takes `replicas x nodes` servers.
  - Tie-break, in order: units, total GPUs, replica size. Idle GPUs are shown.
  - Activations are sized at the real batch (users, capped), KV per session, so a count never assumes room the activations take (final review C2).
  - Examples (fp16, 8k context, 2000 users): deepseek-v4-pro on GB300 is TP16, 4 per rack, 8 idle. kimi-k3 on GB200 is TP48, 1 per rack, 24 idle, 8 racks.
- **D6 KV tier in Sizing** (user decision; it also keeps the host-grace tier reachable).
  - Open sessions come from `kvTierSummary.sessionsHeld`.
  - Decode batch = min(hot, ceil(open x activeShare)), only when sessions are actually parked (open > hot). An overloaded tier falls back to HBM only.
  - `capacityTB` is per unit (server or rack). A replica gets capacity / replicas per unit. A multi-server replica gets one server's capacity (conservative).
  - Example: GLM 5.2 fp8 32k on GB300 with 2000 users needs 4 racks without a tier and 1 rack with host-grace at 25% active share. **This assumption drives the count.** The UI and the PPTX show it.
- **D7 Mode rules.** R15 restores a shared link with a rack GPU in Inference or Training as Sizing. Leaving Sizing with a rack GPU clears the GPU, with a toast. Training on NVL72 is no longer offered.
- **D8 Comparison.** Sizing snapshots show a dash for the weights, KV and activations rows, label Total VRAM "per replica" and speed "Per-user", and are never diffed against inference columns.
- **D9 Non-goals.** Speculative decoding, multi-model racks, price and power, PD disaggregation, EP across a rack, training in Sizing.

## Consequences

- Expected decode falls at batch > 32. Counts are set by memory (90% vLLM limit); the speed shown is expected, with the theoretical ceiling beside it.
- FP8 KV on GLM 5.2 and Mistral Large 4 holds 656/576 more bytes: about 10-12% fewer sessions.
- Rack GPUs disappear from Inference and Training (breaking); old links with them open in Sizing.
- Recommendations may advise one more GPU where the count was not a valid TP degree.

Open items:
- The > 128 band has one point; a decode batch above 256 needs `--max-num-seqs` raised (soft note shown).
- EP across a rack is not modelled, so MoE racks size by TP and pipeline only.
- `slidingWindowKV` and the dense fallback bypass `kvElementBytes` (unreachable today).
- The results skeleton flickers on keystrokes.
- "Does not fit" cannot be exported.
- The decode-speed step at batch 129 follows the banding (user decision).
- A custom GPU without bandwidth shows 0 tok/s.

## Sources

- vLLM `flashmla_sparse.py`, `config.py` (`verify_and_update_config`)
- `docs/calibration/2026-11-11-vllm-efficiency-points.md`
- ADR 0009 (calibration, open items)

## Amendment 2026-11-11: mode split

**Decision (user).** Machines are split by mode. Inference and Training offer discrete machines only (server GPUs: HGX, MI355X, cards). Sizing offers integrated full racks only (`isRackGPU`: GB200, GB300, Vera Rubin NVL72). This reverses the spec line "Sizing also offers every server GPU".

- `gpuAllowed(mode, gpu)` is `(mode === 'sizing') === isRackGPU(gpu)`; the selector lists only that.
- Entering Sizing with a server GPU (mode toggle, GPU picker, or a shared link with `m=sizing`) swaps in the first rack GPU of the catalogue (`rackGPUCorrection`, currently GB300 NVL72), with the usual "Adjusted for ..." toast or "Shared link adjusted" on restore. It runs in `commit()` before `normalizeConfig`, since the GPU is not a rule field, so it cannot loop with R15: after it, mode and GPU agree.
- Leaving Sizing with a rack GPU still clears it; R15 still sends rack-GPU links in Inference/Training to Sizing.

**Consequences.**

- Server sizing (units = servers, multi-server replicas) is unreachable from the UI. The solver keeps its server branch and tests; remove them once confirmed unused.
- With rack GPUs only, R3 (host-grace without a Grace link) cannot fire in Sizing: every rack GPU has one.
- Breaking: old Sizing links with a server GPU open with a rack GPU.
