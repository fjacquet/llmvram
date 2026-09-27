# v2.0: Configuration Rules, GPU Topology, Simplified UI

**Date:** 2026-09-27
**Status:** Design approved in conversation 2026-09-27 (incl. multi-node model replacement and TTFT relabel); spec pending review
**ADRs:** 0004 (rule set), 0005 (audience and UI scope), 0006 (GPU topology), 0007 (multi-node prefill/decode model)
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
- AMD MI300X/MI325X per-degree mesh bandwidth (896 GB/s, priced at 1075).
  Deferred follow-up.
- Linear-attention head divisibility for hybrid models (data lacks linear
  head counts). Documented limitation of R14.
- New inputs or options beyond the 200GbE fabric preset and the H200 NVL entry.

## Section 1: Rule table (`src/engines/config-rules.ts`)

A pure module: `normalizeConfig(config, model, gpu) -> { config, corrections }`
and `allowedOptions(config, model, gpu)`, both driven by one rule table. Each
rule: id, the modes in which its fields are live, applies-when, allowed
values, correction, notice text. `normalizeConfig` applies only the rules
live in the current mode, so a field hidden as inert is never corrected and
never produces a notice. A mode switch is itself a trigger: rules that become
live run then, with one notice. Engine throws remain as a backstop only.

Live modes: R1, R4, R9 (offload fields), R10 in both modes; R2, R3, R5, R12,
R14 in inference only (strategy, KV tier and interconnect are inert in
training); R6 offload/tier parts in inference, optimizer part in training;
R7, R8 in training.

### Hard rules (corrected, with notice)

| Id | Rule | Correction | Notice (English UI text) |
|---|---|---|---|
| R1 | numGPUs in [1, gpu.max_gpus_per_node] | clamp | "GPU count set to {n}: {gpu} supports at most {n} per server." |
| R2 | expert-parallel only if `splitMoEParams(model) !== null` | -> tensor-parallel | "Strategy set to tensor parallel: {model} is not a MoE model." |
| R3 | kvTier host-grace only if `graceLinkGBps(gpu.id) !== null` | -> none | "KV tier turned off: {gpu} has no Grace host memory." |
| R4 | clampKVTier bounds; offloadHostCapacityGB > 0 or null | clamp / null | "KV tier setting adjusted to its allowed range." / "Host capacity reset to the default: it must be a positive number." |
| R5 | interconnectOverride in gpu.interconnect_options, else null | -> null | "Interconnect reset to {default}: not available on {gpu}." |
| R6 | `gpu.unified_memory === true` (the only source; never `interconnect === 'unified'` or the tier): no cpu-ram offload; kvTier not host-pcie/host-grace; no cpuOffloadOptimizer | offloadingEnabled=false (if target cpu-ram); tier none; optimizer offload false | "Offloading turned off: {gpu} has unified memory, RAM is the same pool." / "KV tier turned off: {gpu} has no separate host memory." / "CPU optimizer offload turned off: unified memory." |
| R6 (action intent) | Enabling offloading on a unified-memory GPU while `offloadTarget === 'cpu-ram'` | -> `offloadTarget: 'nvme'` (not `offloadingEnabled=false`; the toggle stays on) | "Offload target set to NVMe: {gpu} has unified memory, RAM is the same pool." A shared link encoding the same combination is not an action intent and takes the plain R6 path above (offloading off, not NVMe). Product owner approved 2026-09-27. |
| R7 | training mode + preset in {vllm, tgi} | preset -> none | "Framework preset cleared: {preset} is inference-only." |
| R8 | cpuOffloadOptimizer only if preset.supportsCpuOffload (zero1/2/3) | -> false | "CPU optimizer offload turned off: needs a DeepSpeed ZeRO preset." |
| R9 | offloadLayers in [0, model.num_hidden_layers]; offloadPercentage in [0,100] | clamp | "Offloaded layers set to {n}: {model} has {n} layers." / "Offload percentage set to {p}%: outside the allowed range." |
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
| W3 | multi-node unified-memory or single-GPU-per-node clusters | "Small clusters: DGX Spark up to 4 units over 200 GbE, DGX Station up to 2; use the 200GbE fabric preset." |
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
switch the strategy to TP, then R14 snaps numGPUs). R14 triggers: model,
numGPUs, strategy (pipeline -> tensor-parallel at 6 GPUs must snap) and a
mode switch to inference. R3/R6/R12 all write
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
- Engine: the existing `resolveInterconnect(gpu)` in `multi-gpu.ts` becomes
  `resolveInterconnect(gpu, groupSize)`: `nvlink_bridge.type` when groupSize <=
  bridge size, else `gpu.interconnect`. Every consumer passes its group size
  and uses the resolved value: TP all-reduce (performance.ts), EP all-to-all
  (group = numGPUs, so a 2-card bridge falls back to PCIe at EP-4+), W2 and
  `validateInterconnect` (multi-gpu.ts), the interconnect badge
  (ShardingStrategySelector), MultiGPUBreakdownChart, the PPTX export, the
  worker and the sync hook. The badge must never show NVLink while the maths
  uses PCIe.
- Bridge display label by bandwidth: "NVLink bridge — 600 GB/s" (H100/A100
  PCIe; the H100 bridge is NVLink 4 at 600 GB/s), "NVLink bridge — 900 GB/s"
  (H200 NVL). The data types the H100/A100 PCIe bridge as the 600 GB/s
  `nvlink-3` bucket (the schema has no separate "NVLink 4 bridge at 600 GB/s"
  tier), which is what drives the "NVLink bridge — 600 GB/s" label; the
  bandwidth shown is correct either way.

Per-card changes (sources in the table below):

| Id | Change |
|---|---|
| nvidia-h100-80gb-pcie | interconnect pcie-5; nvlink_bridge {nvlink-3, 2}; fp16_tflops 989 -> 756 (1,513 is sparse) |
| nvidia-a100-80gb-pcie | interconnect pcie-4; nvlink_bridge {nvlink-3, 2} |
| nvidia-a100-80gb-sxm | interconnect nvlink-3 (600 GB/s, was priced at 900) |
| nvidia-h200-141gb | SXM only (HGX 4/8 NVSwitch); drop interconnect_options |
| nvidia-h200-nvl-141gb (new) | H200 NVL, NVIDIA group in `fetch-gpus.ts`: every required GPUSchema field copied from the PNY H200 NVL datasheet (vram 141, bandwidth, fp32), fp16_tflops 835 dense (datasheet 1,671 with sparsity), pcie-5, nvlink_bridge {nvlink-4, 4}, max 8, tier datacenter. Update the GPU count in README/CLAUDE.md memory (27 -> 28). |
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
  100G fabric 13.3 s after Section 3b (was 195 s).
- L40S / RTX 6000 Ada: decode unchanged, TTFT about +20%.
- Unified-memory offload to cpu-ram (M3 Ultra, Llama 3.1 70B Q4_K_M, 30%):
  3.4 tok/s (fictitious slowdown) -> 17.7 tok/s (offloading off).
- No change: R2, R3, R4, R8, R9, R10 for already-valid configs.

## Section 3b: Multi-node prefill and decode model (ADR 0007)

Replaces the unsourced efficiency heuristic in `src/engines/fabric.ts`
(`fabricPrefillEfficiency`, `fabricDecodeEfficiency`, `FABRIC_REFERENCE_GBPS`,
`EFFICIENCY_FLOOR`, `PP_BASE_EFFICIENCY`, added in 9f33ac3, never measured).

- Stage-boundary transfer ("hop"):
  `hop(tokens) = tokens x hidden x 2 tensors x 2 B / (eta x port x gpusPerNode) + latency`
  (vLLM sends `hidden_states` and `residual`; each TP rank sends a 1/tp slice
  and the receiver all-gathers, so per-node aggregate bandwidth applies).
- Prefill: compute over the pipeline with the GPipe bubble, microbatches
  `M = ceil(B x T / C)`, C = vLLM `max_num_batched_tokens` default (16384 for
  >= 160 GB GPUs, 8192 for >= 70 GB non-A100, else 2048); speedup
  `N x M / (M + N - 1)`; plus `(N - 1) x hop(T)`. At B = 1 and T <= C one
  request walks the stages serially (no cross-node speedup).
- Decode: existing `B / (B + stages - 1)` kept; the inter-node efficiency
  multiplier is dropped; `(N - 1) / stages x hop(B)` added per step
  (conservative: vLLM sends asynchronously).
- Constants, labelled honestly in code and docs:
  - eta GB10 (no GPUDirect RDMA) = 0.37 - SECONDARY source (one measurement:
    NCCL send/recv ~9 GB/s vs 24.6 GB/s RDMA on a Spark 200G link).
  - eta HGX = 0.8 - ASSUMPTION (hint only: GH200 all_reduce 45.4/50 GB/s).
  - latency = 10 us RDMA floor (arXiv 2511.15076); vLLM metadata likely
    50-200 us, still < 1% of a decode step.
  - Justification for shipping them: results are insensitive to eta and
    latency (doubling port speed changes decode by < 1%, matching vllm#6610:
    21.0 tok/s at 400G vs 21.1 at 800G on 2x GH200 PP=2).
- `classFactor` becomes `effectiveFraction` (eta); `MultiGPUVRAMBreakdown`
  carries `interNodeGBps` instead of the two efficiencies. Consumers:
  multi-node.ts, multi-gpu.ts, performance.ts, types.ts,
  MultiGPUBreakdownChart, the worker and the sync hook.
- TTFT meaning: for B > 1 the figure is the prefill of a burst of B prompts
  divided by B. Label "Time to first token" -> "Prefill per request
  (amortized over batch B)" in the UI and PPTX; the guide explains that in a
  real burst the last request waits about B times longer.
- New fabric preset `ethernet-200g` (portGBps 25).

Impact (research pass, 8k context; TTFT s before -> after):
- 2x GB10 70B B1 100G 195 -> 13.3; 4x GB10 70B B32 100G 71 -> 3.7;
  4x GB10 405B B32 100G 395 -> 20.
- HGX H100x8 70B B32: 2 nodes 100G 0.47 -> 0.20; 4 nodes 100G 0.25 -> 0.11.
- HGX B1 (single request): 2 nodes 800G+ +16 to +26% slower; 4 nodes 200G+
  +40 to +115% slower (the old cross-node speedup did not exist in vLLM).
- Decode: GB10 100-800G +5 to +9%; HGX 100G +4-5%; else < 5%.
- KV-tier "resume faster than recompute": no flip in 100 configs; tightest
  70B B200x8 4-node B32 1.6T (resume 0.037 s vs recompute 0.048 s).

Sources: vLLM `config/vllm.py` (max_concurrent_batches = pp_size),
`v1/engine/core.py` (batch queue), `v1/core/sched/scheduler.py` (chunked
prefill rescheduling), `engine/arg_utils.py` (max_num_batched_tokens),
`distributed/parallel_state.py` (send_tensor_dict), `models/llama.py`;
GPipe arXiv 1811.06965; Megatron arXiv 2104.04473; vllm#6610; vllm#41685;
multimodalflow DGX Spark dual-node NCCL RDMA; arXiv 2511.15076.

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
- Two reset buttons (product owner decision, 2026-09-27):
  - "Reset advanced settings" inside the Advanced section: batch, KV
    precision, strategy, fabric, interconnect variant, offloading and KV tier
    back to their store defaults; model, GPU, GPU count, servers, format,
    context and concurrent users kept.
  - "Reset all" in the header: back to the initial empty state, URL hash
    cleared.
  - Both go through `normalizeConfig` and show one notice listing what was
    reset (always-warn rule).

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
  impact anchors pinned to absolute tok/s computed in the test from the corrected data (H100 PCIe TP-8 fp8 bs1 is the -25.7% case); training ZeRO-3 on 6 GPUs with Llama 3.1 70B keeps 6 GPUs and emits no notice (R14 inference-only).
- UI: composition tests for InputPanel and ResultsPanel written before the
  layout change (visible-by-default set, auto-open on non-default, warnings
  visible, strategy reachable at max > 1); PDF export expands details.
- Multi-node (Section 3b): doubling portGBps changes decode tok/s < 1%;
  2-node B1 TTFT = single-node prefill + (N-1) x hop exactly; `fabricHopSeconds`
  268,435,456 B at 25 x 0.37 GB/s + 10 us = 29.03 ms; 2x GB10 70B 8k TTFT
  within 1% of one GB10; B32 applies M = ceil(B x T / C); single-node
  passthrough unchanged; KV-tier verdict test on the tightest config.
- No test fixture is a hand-written GPU/model literal (CLAUDE.md).

## Section 7: Delivery and docs

Task order in the single PR: (1) rules module + tests; (2) store integration
and URL restore; (3) GPU data, nvlink-3, resolveInterconnect; (4) multi-node
model (Section 3b) + 200GbE preset + TTFT relabel; (5) allowedOptions in the UI + notices; (6) composition tests,
then the UI simplification; (7) guide, CHANGELOG, ADRs, CLAUDE.md.

Docs: CHANGELOG `## [2.0.0]` with a "Breaking changes" subsection (layout,
label renames, corrected links, per-card number changes, H200 id now SXM,
multi-node TTFT changes from Section 3b);
CLAUDE.md (rules module key pattern; clamping no longer silent; numGPUs =
parallel degree of one replica; nvlink_bridge / unified_memory);
ARCHITECTURE.md; ADR 0006 moved to Accepted with the decisions taken.

## Appendix: per-card audit (pass 3, one source per card)

"Today" = interconnect / interconnect_options / max_gpus_per_node before v2.

| id | Product / form factor | Scale-up link | Sold as (GPUs) | NVLink group | Today | Verdict | Source |
|---|---|---|---|---|---|---|---|
| nvidia-h100-80gb-pcie | H100 PCIe card | 2-way bridge, 600 GB/s | 1-8 | 2 | nvlink-4 / - / 8 | WRONG: pcie-5 + bridge; fp16 756 | NVIDIA H100 datasheet; Lenovo LP1732 |
| nvidia-h100-80gb-sxm | HGX H100 | NVSwitch 900 GB/s | 4, 8 | 4, 8 | nvlink-4 / - / 8 | OK | H100 datasheet; NVIDIA HGX AI Factory RA |
| nvidia-h200-141gb | mixed SXM + NVL | SXM NVSwitch; NVL 2/4-way bridge 900 GB/s | SXM 4, 8; NVL up to 8 | SXM 8; NVL 2, 4 | nvlink-4 / [nvlink-4, pcie-5] / 8 | WRONG: split (id = SXM; new NVL id) | PNY H200 NVL datasheet; HPE PSN1014857028PLEN / PSN1014856854VNEN |
| nvidia-b200-192gb | HGX/DGX B200 | NVSwitch 1.8 TB/s | 8 (4: UNVERIFIED) | 8 | nvlink-5 / - / 8 | OK | NVIDIA DGX B200; HGX AI Factory RA |
| nvidia-gb300-288gb | HGX B300 | NVSwitch 1.8 TB/s | 8 | 8 | nvlink-5 / - / 8 | OK (secondary source only) | pantheon.run HGX B300 specs |
| nvidia-gb300-nvl72 | NVL72 rack | rack NVLink domain | 72 | up to 72 | nvlink-5 / - / 72 | OK | NVIDIA GB300 NVL72 |
| nvidia-a100-80gb-pcie | A100 PCIe card | 2-way bridge, 600 GB/s | 1-8 | 2 | nvlink / - / 8 | WRONG: pcie-4 + bridge | NVIDIA A100 page |
| nvidia-a100-80gb-sxm | HGX A100 | NVSwitch 600 GB/s | 4, 8, 16 | 8 | nvlink (priced 900) / - / 8 | WRONG bandwidth: nvlink-3 | NVIDIA A100 page |
| nvidia-l40s | PCIe card | none | 1-8 (UNVERIFIED) | 1 | none / - / 8 | WRONG: pcie-4 | Lenovo LP1812 |
| nvidia-rtx-pro-6000-server | PCIe card | none | up to 8 | 1 | none / - / 8 | OK (explicit pcie-5 preferred) | NVIDIA RTX PRO 6000 Server; RTX PRO Server |
| nvidia-rtx-6000-ada | workstation PCIe | none | UNVERIFIED | 1 | none / - / 8 | WRONG: pcie-4 | NVIDIA RTX 6000 Ada datasheet |
| nvidia-rtx-5090 | GeForce PCIe 5 | none | UNVERIFIED | 1 | none / - / 8 | OK | NVIDIA GeForce compare |
| nvidia-rtx-4090 | GeForce PCIe 4 | none | UNVERIFIED | 1 | none / - / 8 | OK | NVIDIA GeForce compare |
| nvidia-rtx-3090 | GeForce PCIe 4 | 2-way bridge | UNVERIFIED | 2 | none / - / 8 | OK, conservative | NVIDIA GeForce compare |
| nvidia-gb300-desktop-252gb | DGX Station, 1 GPU | NVLink-C2C to CPU only | 1 | 1 | nvlink-5 / - / 1 | set none | NVIDIA DGX Station |
| nvidia-gb10 | DGX Spark | none; ConnectX-7 200 Gb/s, up to 4 units | 1 | 1 | nvlink-5 / [nvlink-5, pcie-5] / 2 | WRONG: max 1, none | NVIDIA DGX Spark |
| amd-mi355x | OAM on UBB | IF full mesh, 1075 GB/s | 8 | 8 | infinity-fabric / - / 8 | OK | ROCm MI350; AMD MI355X brochure |
| amd-mi350x | OAM on UBB | IF mesh, 1075 GB/s | 8 | 8 | infinity-fabric / - / 8 | OK | AMD MI350X brochure |
| amd-mi325x | OAM on UBB | IF mesh, 896 GB/s | 8 | 8 | infinity-fabric (priced 1075) / - / 8 | bandwidth 20% high (deferred) | AMD MI325X platform datasheet |
| amd-mi300x | OAM on UBB | IF mesh, 896 GB/s | 8 | 8 | infinity-fabric (priced 1075) / - / 8 | bandwidth 20% high (deferred) | AMD MI300X platform datasheet |
| apple-* (8 rows) | unified-memory chip | none; Thunderbolt clusters (TB5 RDMA, macOS 26.2) | 1 | 1 | unified / - / 1 | OK | Apple newsroom spec_url per row |

Exact ids are taken from `gpus.json` during implementation; rows above use
the audit's names.
