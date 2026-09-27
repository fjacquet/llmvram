# v2.0: Configuration Rules, GPU Topology, Simplified UI

**Date:** 2026-09-27
**Status:** Design approved in conversation 2026-09-27; spec pending review
**ADRs:** 0004 (rule set), 0005 (audience and UI scope), 0006 (GPU topology)
**Delivery:** one PR, released as v2.0.0 (plain merge and tag on the user's word)

## Context

Three verification passes (code reproduction with the real store and engines,
vendor sources per card) found:

- Input dependency rules enforced in 9 places, each covering some entry
  paths only. Reproduced: expert-parallel restored from a shared link onto a
  dense model (engine error); training mode keeping vLLM/TGI; CPU-RAM offload
  and PCIe KV tiers offered on unified-memory GPUs; 500 offloaded layers on an
  80-layer model; training settings written into links but never restored;
  `interconnectOverride` never serialized.
- Wrong GPU topology data: bridge-only NVLink cards modelled as 8-way NVLink,
  one H200 entry mixing two products, DGX Spark modelled as 2 GPUs per node,
  A100 SXM priced at NVLink-4 bandwidth, H100 PCIe FP16 at the sparse figure.
- Tensor-parallel degrees vLLM refuses to run (e.g. Llama 70B on 3, 5, 6, 7
  GPUs) computed as if valid.
- A UI with 8 always-visible inputs, 9 conditional groups, up to 12 result
  blocks, nothing collapsible (audience decision: ADR 0005).

## Goals

1. No impossible combination can be selected or restored; every correction is
   shown to the user.
2. GPU data describes each card's real topology, sourced per card.
3. The default screen shows the essential inputs and a verdict; the rest is
   one click away and never hides an active non-default setting.
4. Results for configurations that were already valid and whose GPU data was
   correct are unchanged.

## Non-Goals

- Counting replicas (the rename in Section 5 is a label, not a model).
- The fabric prefill efficiency floor (`EFFICIENCY_FLOOR`, `FABRIC_REFERENCE_GBPS`
  in `src/engines/fabric.ts`): no sourced model yet; changing it would move
  existing 100/400 GbE results. Deferred; small clusters get a W3 warning.
- AMD MI300X/MI325X per-degree mesh bandwidth (896 GB/s, priced at 1075).
  Deferred follow-up.
- Linear-attention head divisibility for hybrid models (data lacks linear
  head counts). Documented limitation of R14.
- New inputs or options beyond the 200GbE fabric preset and the H200 NVL entry.

## Section 1: Rule table (`src/engines/config-rules.ts`)

A pure module: `normalizeConfig(config, model, gpu) -> { config, corrections }`
and `allowedOptions(config, model, gpu)`, both driven by one rule table. Each
rule: id, applies-when, allowed values, correction, notice text. Engine
throws remain as a backstop only.

### Hard rules (corrected, with notice)

| Id | Rule | Correction | Notice (English UI text) |
|---|---|---|---|
| R1 | numGPUs in [1, gpu.max_gpus_per_node] | clamp | "GPU count set to {n}: {gpu} supports at most {n} per server." |
| R2 | expert-parallel only if `splitMoEParams(model) !== null` | -> tensor-parallel | "Strategy set to tensor parallel: {model} is not a MoE model." |
| R3 | kvTier host-grace only if `graceLinkGBps(gpu.id) !== null` | -> none | "KV tier turned off: {gpu} has no Grace host memory." |
| R4 | clampKVTier bounds; offloadHostCapacityGB > 0 or null | clamp / null | "KV tier setting adjusted to its allowed range." |
| R5 | interconnectOverride in gpu.interconnect_options, else null | -> null | "Interconnect reset to {default}: not available on {gpu}." |
| R6 | unified_memory GPU: no cpu-ram offload; kvTier not host-pcie/host-grace; no cpuOffloadOptimizer | offloadingEnabled=false (if target cpu-ram); tier none; optimizer offload false | "Offloading turned off: {gpu} has unified memory, RAM is the same pool." / "KV tier turned off: {gpu} has no separate host memory." / "CPU optimizer offload turned off: unified memory." |
| R7 | training mode + preset in {vllm, tgi} | preset -> none | "Framework preset cleared: {preset} is inference-only." |
| R8 | cpuOffloadOptimizer only if preset.supportsCpuOffload (zero1/2/3) | -> false | "CPU optimizer offload turned off: needs a DeepSpeed ZeRO preset." |
| R9 | offloadLayers in [0, model.num_hidden_layers]; offloadPercentage in [0,100] | clamp | "Offloaded layers set to {n}: {model} has {n} layers." |
| R10 | numNodes 1-8; batchSize >= 1; concurrentUsers >= 1; sequenceLength 512-10,485,760; loraRank >= 1; gradientAccumulationSteps >= 1 | clamp | "{field} set to {value}: outside the allowed range." |
| R12 | kvCacheOffload and kvTier != none | tier -> none | "KV tier turned off: KV cache offload already keeps all KV off the GPU." |
| R13 | interconnect_options only offered when max_gpus_per_node > 1 | (data + UI) | none |
| R14 | strategy tensor-parallel: numGPUs must satisfy vLLM (`heads % t == 0`, and `t % kv == 0` if t >= kv else `kv % t == 0`) | numGPUs -> nearest valid degree <= current (TP-1 always valid) | "GPU count set to {t}: vLLM can't split {model}'s {h} attention heads across {n} GPUs. Use pipeline parallel for {n}." |

Action intent (not a rule): picking vLLM/TGI sets mode = inference before
normalizing. Training presets are kept when switching to inference (inert;
preserved for the round trip). `none` is valid in both modes; do not derive
this from `FRAMEWORK_PRESETS.none.mode`.

### Soft rules (warning only, value kept)

| Id | Condition | Warning |
|---|---|---|
| W1 | sequenceLength > model context | existing RoPE/YaRN warning |
| W2 | TP degree > interconnect recommendedMaxTPDegree | existing |
| W3 | multi-node unified-memory or single-GPU-per-node clusters | "Small clusters: DGX Spark up to 4 units over 200 GbE, DGX Station up to 2; prefill over slow fabrics is approximate." |
| W4 | custom GPU without FLOPS | memory-bound fallback note |
| W5 | quantization format vs GPU generation | support note |
| W6 | pipeline stages > num_hidden_layers | "{n} pipeline stages exceed {model}'s {L} layers; some stages would be empty." |
| W8 | expert-parallel and num_experts % t != 0 | "{E} experts don't split evenly across {t} GPUs." |

(W7, "invalid TP degree", became hard rule R14 by decision.)

### Inert inputs (hidden, kept, ignored; never reset)

Training mode: offloading, sharding, interconnect, weight/KV quantization,
concurrent users, servers/fabric, numGPUs > 1 without a ZeRO preset.
numGPUs = 1: sharding. numNodes = 1: fabric.

### Dependency graph and ordering (from pass 3, extended for R14)

Edges: R7 -> R8; R1 -> R14 (R14 runs after R1); R2 -> R14 (model change can
switch the strategy to TP, then R14 snaps numGPUs). R3/R6/R12 all write
tier = none; R6/R8 both write optimizer offload = false (same values, no
conflict). No cycle: the mode/preset coupling is action intent, not a rule.
Every correction is a constant or a clamp toward model/GPU data, which no
rule writes; `normalizeConfig` reaches a fixpoint in at most 3 passes in any
order and is idempotent. The implementation loops to a fixpoint (max 4
passes, asserted in tests).

### Valid TP degrees (all 54 models, computed from models.json)

Up to 8 GPUs: {1,2,4,8} for 51 models; google-gemma-3-1b {1,2,4};
zai-org-glm-4.7-flash {1,2,4,5}; moonshotai-kimi-k3 {1,2,3,4,6,8}. No model
is TP-1-only. Up to 72 (NVL72): most models {1,2,4,8,16,32,64}; 40-head
models add 40; 48-head models {1,2,4,8,16,24,48}; 96-head models
{1,2,4,8,16,24,32,48}, Kimi K3 (kv 96) also 3, 6, 12; 24-head models
{1,2,4,8,12,24}. MLA entries store kv heads = heads, so the rule
reduces to the heads check, matching vLLM's MLA path. Custom models: kv heads
default to heads (no constraint beyond heads).

### Notices

One toast per user action (sonner), title "Adjusted for {GPU/model/mode}" and
one line per correction; on shared-link restore the title is "Shared link
adjusted". Soft warnings render inline where they exist today.

## Section 2: Store and URL integration

- Every store action builds the next config and calls `normalizeConfig`;
  corrections are emitted to the notice channel. The scattered checks
  (`clampGPUCount` calls, `resetTierForGPU`, the strategy reset in
  `setSelectedModel`, `startsWith('deepspeed-')`) are replaced by the table.
- URL restore builds the whole config first, then normalizes once. `fp` is
  restored raw (no auto-optimizations, which would overwrite `to`, `gc`,
  `fa`). Restored keys: all currently serialized, including ga, gc, fa, fp, co.
  New serialized key: interconnectOverride.
- Update the `clampGPUCount` "silent by design" comment and the CLAUDE.md
  "GPU count clamping is silent" bullet: corrections are now always shown.

## Section 3: GPU data (edited in `scripts/fetch-gpus.ts`, regenerated)

Schema (`GPUSchema`, custom-GPU factory, URL custom-GPU schema):
- `unified_memory?: boolean` (true for all apple-silicon entries and nvidia-gb10).
- `nvlink_bridge?: { type: InterconnectType; size: number }`.
- New `INTERCONNECT_SPECS['nvlink-3']` (600 GB/s).
- Engine: `resolveInterconnect(gpu, tpDegree)` uses `nvlink_bridge.type` when
  tpDegree <= bridge size, else `gpu.interconnect`.

Per-card changes (sources in the table below):

| Id | Change |
|---|---|
| nvidia-h100-80gb-pcie | interconnect pcie-5; nvlink_bridge {nvlink-3, 2}; fp16_tflops 989 -> 756 (1,513 is sparse) |
| nvidia-a100-80gb-pcie | interconnect pcie-4; nvlink_bridge {nvlink-3, 2} |
| nvidia-a100-80gb-sxm | interconnect nvlink-3 (600 GB/s, was priced at 900) |
| nvidia-h200-141gb | SXM only (HGX 4/8 NVSwitch); drop interconnect_options |
| nvidia-h200-nvl-141gb (new) | H200 NVL: pcie-5; nvlink_bridge {nvlink-4, 4}; fp16_tflops 835; max 8 |
| nvidia-gb10 | max_gpus_per_node 1; interconnect none; drop interconnect_options; unified_memory |
| nvidia-gb300-desktop-252gb | interconnect none (single GPU) |
| nvidia-l40s, nvidia-rtx-6000-ada | explicit pcie-4 |
| apple-silicon entries | unified_memory |
| fabric: ethernet-200g (new preset) | portGBps 25 (200 Gb/s ConnectX-7) |
| framework: tgi | label "TGI (archived)", kept |

Ids never change. Old `nvidia-h200-141gb` links now resolve to the SXM
product; a sender who meant NVL cannot be detected (stated in the CHANGELOG,
not a notice).

Sources (per card, from pass 3): H100 datasheet
(megware nvidia-h100-datasheet.pdf), Lenovo LP1732; A100
(nvidia.com/en-us/data-center/a100/); PNY H200 NVL datasheet; HPE 2/4-way
bridge PSN1014857028PLEN / PSN1014856854VNEN; NVIDIA HGX AI Factory
components; DGX Spark and DGX Station product pages; NVIDIA Sync cluster
assistant; Lenovo LP1812 (L40S); RTX 6000 Ada datasheet; vLLM
config/model.py, model_executor/layers/linear.py (TP divisibility).

### Computed impact (pass 3, verbatim)

- H100 PCIe, Llama 3.1 70B 8k, decode tok/s: fp8 bs1 TP-2 -9.9%, TP-4
  -16.8%, TP-8 -25.7%; fp8 bs32 -5.1 / -9.4 / -16.1%; fp16 bs1/bs32
  -5.7/-3.7, -10.3/-6.9, -17.2/-12.3%. A100 PCIe: -3.6 to -25.2%. TTFT +18%
  (H100), +41% (A100). Memory unchanged. W2 now fires at TP-8 on pcie-5 and
  TP-4/8 on pcie-4.
- 2x GB10 (was TP-2 over a fictitious NVLink-5), Llama 3.1 70B fp8 bs1:
  7.2 tok/s -> 3.6 (clamped to 1). Correct 2-node setup: ~3.4 tok/s; TTFT at
  100G fabric 194 s (fabric floor, deferred; W3 shown).
- L40S / RTX 6000 Ada: decode unchanged, TTFT about +20%.
- Unified-memory offload to cpu-ram (M3 Ultra, Llama 3.1 70B Q4_K_M, 30%):
  3.4 tok/s (fictitious slowdown) -> 17.7 tok/s (offloading off).
- No change: R2, R3, R4, R8, R9, R10 for already-valid configs.

## Section 4: UI (ADR 0005)

- Inputs: "Essential" always visible (mode, model, GPU + GPUs per replica +
  servers, format, context, concurrent users). "Advanced" in a native
  `<details>` (batch, KV precision, strategy, fabric, interconnect variant,
  offloading, KV tier), opened automatically when any advanced value differs
  from its default, with "N settings changed" in its summary.
- The strategy selector is visible whenever `gpu.max_gpus_per_node > 1` (not
  only when numGPUs > 1), so pipeline parallel stays reachable before R14
  snaps a TP degree.
- The GPU-count slider offers only values allowed by `allowedOptions`
  (R1, R14); invalid degrees are not selectable.
- Options not allowed (R3, R6, R12, R13) are not offered.
- Label "GPUs per server" -> "GPUs per replica (in one server)"; with
  numNodes > 1 the summary reads "{n} GPUs per server x {m} servers per
  replica". Same rename in the PPTX "Number of GPUs" row.
- Results: verdict block (fit gauge, decode speed, first-token delay, max
  sessions); warnings always visible; "Details" collapsed (charts, breakdown
  table, multi-GPU chart, weight source, tier summary, per-user metrics, GiB
  note). The PDF export expands everything before capture, then restores.
- Training mode hides inert inputs (Section 1).
- The GPUCountSelector summary labels expert-parallel correctly (was
  "pipeline parallel").

## Section 5: User guide

`src/components/guide/GuidePage.tsx`:
- New "What's new in 2.0" section: layout, rules, data corrections per card,
  what changes for existing links.
- New "Why a setting changed": every hard rule in plain words, the notice,
  and what to do (e.g. pipeline parallel for 6 GPUs).
- GPU Selection / Hardware: topology per card (NVSwitch 4/8, bridges 2/4,
  NVL72, unified memory), "GPUs per replica", valid TP degrees.
- Offloading / KV tier: unified-memory behaviour; host capacity.
- Results: verdict block, details, PDF export.
- URL Sharing: links open corrected with a notice; training settings and the
  interconnect variant now travel.

## Section 6: Testing

- Rules module: per rule, three paths (change the field, change its
  dependency, restore a link) as a table; properties: idempotent, order
  independent (permute restore order), fixpoint within 4 passes; R14 against
  the valid-degree table above; R14 + R2 cascade on model change.
- Store: real store (persist mocked as pass-through); notices emitted once
  per action with all corrections.
- URL: round-trip for every serialized key including the new ones; hostile
  links (dense + expert-parallel, sl < 512, bs 0) open corrected with the
  "Shared link adjusted" notice.
- Data: per-card assertions for the changed fields; `resolveInterconnect`
  bridge behaviour (TP-2 on H100 PCIe uses nvlink-3, TP-4 uses pcie-5);
  impact anchors from Section 3 (H100 PCIe TP-8 fp8 bs1 ~-25.7%).
- UI: composition tests for InputPanel and ResultsPanel written before the
  layout change (visible-by-default set, auto-open on non-default, warnings
  visible, strategy reachable at max > 1); PDF export expands details.
- No test fixture is a hand-written GPU/model literal (CLAUDE.md).

## Section 7: Delivery and docs

Task order in the single PR: (1) rules module + tests; (2) store integration
and URL restore; (3) GPU data, nvlink-3, resolveInterconnect; (4) 200GbE
preset + W3; (5) allowedOptions in the UI + notices; (6) composition tests,
then the UI simplification; (7) guide, CHANGELOG, ADRs, CLAUDE.md.

Docs: CHANGELOG `## [2.0.0]` with a "Breaking changes" subsection (layout,
label rename, corrected links, per-card number changes, H200 id now SXM);
CLAUDE.md (rules module key pattern; clamping no longer silent; numGPUs =
parallel degree of one replica; nvlink_bridge / unified_memory);
ARCHITECTURE.md; ADR 0006 moved to Accepted with the decisions taken.
