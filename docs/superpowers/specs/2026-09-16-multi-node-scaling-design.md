# Multi-Node Scaling + AMD Instinct MI350/MI325 Series

**Date:** 2026-09-16
**Status:** Approved, pending implementation plan

## Context

Two requests, resolved into one design because the second depends on the first
being correct:

1. Add AMD Instinct MI355X to the GPU database (extended to MI350X and MI325X).
2. Support scaling a configuration across multiple identical servers.

The calculator currently models a single node of 1–8 GPUs. Frontier models on
288 GB accelerators still exceed one node at long context, so a node dimension
is the natural next axis. Along the way, adding AMD hardware surfaced an
existing defect in how AMD interconnect is resolved, and the multi-node work
surfaced a second defect in the pipeline-parallel KV cache term. Both are fixed
here because both change the numbers this feature reports.

## Goals

- Three AMD Instinct rows: MI355X, MI350X, MI325X.
- A truthful intra-node interconnect model for AMD hardware.
- Inference across N identical servers: tensor parallel inside a node, pipeline
  parallel across nodes, with a user-selectable scale-out fabric.

## Non-Goals

- Multi-node **training**. `training.ts`, `lora.ts` and `deepspeed.ts` stay
  single-node. Cross-node FSDP/ZeRO-3 gradient and optimizer sharding is a
  distinct memory model and gets its own spec if wanted.
- Heterogeneous nodes. "Multiple identical servers" is the whole scope; mixed
  GPU types per node are out.
- Tensor parallelism across a node boundary. Not offered — see §3.

## Findings in Existing Code

These are defects in code this feature builds on. Both are fixed as part of the
work, and both change assertions in existing tests.

### Finding A — pipeline-parallel holds the full KV cache on every stage

`kv-cache.ts:73` multiplies by `num_hidden_layers`, so `singleGPU.kvCache` is
the total across **all** layers. Pipeline parallelism splits layers across
stages, so each stage holds KV for only its own layers. But `multi-gpu.ts:147`
does:

```ts
// KV cache NOT divided (each GPU needs full cache for its layers)
const kvCachePerGPU = singleGPU.kvCache
```

The comment conflates "full cache for the sequence" (true) with "full cache for
all layers" (false). At 8-way PP this overstates the KV term by 8×.

This is tolerable today because PP is a minority path. It is not tolerable once
PP composes with TP: at 4 nodes × 8 GPUs, weights shrink 32× while KV does not
shrink at all, so per-GPU total becomes KV-dominated and long-context multi-node
configurations report "does not fit" when real deployments fit comfortably. On
288 GB MI355X cards this is precisely the regime users will be in.

**Fix:** `kvCachePerGPU = singleGPU.kvCache.div(numStages)`.

**Test impact:** `multi-gpu.test.ts` lines 247, 297–301 and 339 assert the
current behavior and are updated.

### Finding B — one scaling number cannot serve both rooflines

`performance.ts` applies `multiGPUResult.scalingEfficiency` at two sites:

- line ~104 — decode tokens/sec
- line ~155 — prefill effective FLOPS

Across a node boundary these diverge sharply. Decode ships one token's
activations per stage handoff (batch × hidden × 2 bytes — kilobytes), so the hop
is latency-bound at roughly 10 µs against a 10–20 ms decode step: near-free.
Prefill ships full-sequence activations (batch × seqlen × hidden × 2 bytes) and
is genuinely bandwidth-bound.

A single flat inter-node efficiency penalizes decode and flatters prefill.

**Fix:** two inter-node efficiencies, applied at their respective sites. The
combined `scalingEfficiency` field is retained as the decode-path product so
`performance.ts`'s decode site needs no signature change.

## Design

### §1 — GPU data: three AMD Instinct rows

`src/data/gpus.json` is **generated**. Source of truth is the `GPUS: GPU[]`
literal in `scripts/fetch-gpus.ts`; rows go there, adjacent to `amd-mi300x`
(line 193), then `npm run refresh:gpus` regenerates the JSON. The file is
grouped by vendor, not sorted alphabetically — the `models.json` sort rule does
not apply.

| field | MI355X | MI350X | MI325X |
|---|---|---|---|
| `id` | `amd-mi355x` | `amd-mi350x` | `amd-mi325x` |
| `name` | AMD Instinct MI355X | AMD Instinct MI350X | AMD Instinct MI325X |
| `manufacturer` | `amd` | `amd` | `amd` |
| `vram_gb` | 288 | 288 | 256 |
| `memory_bandwidth_gbps` | 8000 | 8000 | 6000 |
| `memory_type` | HBM3E | HBM3E | HBM3E |
| `bus_width` | 8192 | 8192 | 8192 |
| `fp16_tflops` | 2516 | 2300 | 1307 |
| `fp32_tflops` | 157 | 144 | 163 |
| `tdp_watts` | 1400 | 1000 | 1000 |
| `interconnect` | `infinity-fabric` | `infinity-fabric` | `infinity-fabric` |
| `tier` | `datacenter` | `datacenter` | `datacenter` |

`fp16_tflops` is **dense**, matching the database's existing convention (H100
989, B200 4500, MI300X 1307). AMD's headline "4.6/5.0 PFLOPS FP16" figures are
with sparsity and must not be used. MI325X reuses MI300X compute — same CDNA 3
core, more memory — consistent with the existing MI300X row.

MI355X (1400 W, liquid) and MI350X (1000 W, air) share die and memory; the
MI355X's ~9% throughput advantage comes from clocks.

**Verification required at implementation:** every numeric above is confirmed
against AMD's official MI350-series datasheet PDF before the row is written.
Figures here are from secondary sources cross-checked against each other.

### §2 — Intra-node interconnect: real Infinity Fabric

`multi-gpu.ts:312` currently resolves `'infinity-fabric'` to `'pcie-5'`
(128 GB/s, 0.78 efficiency, recommended max TP degree 4). An 8-OAM node with
full-mesh xGMI is nothing like PCIe-5. The consequence today is inflated
per-GPU VRAM for every AMD card plus a spurious "TP with 8 GPUs may degrade"
warning from `ShardingStrategySelector.tsx:41`.

Infinity Fabric / xGMI is AMD's **scale-up** interconnect — direct GPU-to-GPU
peer access inside one chassis, the same layer as NVLink and equally
vendor-specific. The enum already carries vendor-specific members (`nvlink-4`,
`nvlink-5`), so `'infinity-fabric'` fits the established pattern.

Changes:

- `types.ts:155` — `InterconnectType` gains `'infinity-fabric'` (5 members → 6).
  Every exhaustive `Record<InterconnectType, …>` must gain the case; the
  compiler enumerates them.
- `constants.ts:199` — new `INTERCONNECT_SPECS` entry.
- `multi-gpu.ts:312` — return `'infinity-fabric'` instead of `'pcie-5'`.

```ts
'infinity-fabric': {
  type: 'infinity-fabric',
  bandwidthGBps: 1075,
  recommendedMaxTPDegree: 8,
  tpScalingEfficiency: 0.93, // 1075 GB/s — between NVLink-4 and NVLink-5
},
```

**Units.** 1075 GB/s is AMD's aggregate per-GPU GPU-to-GPU figure for an 8-way
fully-connected MI350X platform, **bidirectional**. The existing table is
bidirectional throughout (NVLink-5's 1800 is NVIDIA's bidirectional per-GPU
number). A widely-quoted 538 GB/s figure is the unidirectional half of the same
number and must not be used here — mixing the two would halve AMD's apparent
capability.

**Efficiency derivation.** Interpolated log-linearly between the two adjacent
anchors in the existing table: NVLink-4 (900 GB/s → 0.92) and NVLink-5
(1800 GB/s → 0.97), i.e. +0.05 per doubling.
`0.92 + 0.05 · log₂(1075/900) = 0.933` → **0.93**. This is an interpolation,
not a measurement, consistent with how the rest of `INTERCONNECT_SPECS` was
constructed. The derivation goes in the code comment.

**Blast radius.** PCIe-5 is unaffected in both its other roles: the direct
mapping at `multi-gpu.ts:303` (for GPUs whose field is literally `'pcie-5'`,
e.g. H200's `interconnect_options`) and the tier fallback at `multi-gpu.ts:316`
(datacenter GPUs with `interconnect: 'none'` or undefined — L40S, RTX PRO 6000).
Nothing is removed from `INTERCONNECT_SPECS`.

**Accepted side effect.** MI300X results change too — they are currently
pessimistic. MI300X is 4th-gen Infinity Fabric and MI350-series is 5th; a single
shared entry at 1075 GB/s slightly flatters MI300X. Splitting into
`infinity-fabric-4` / `infinity-fabric-5` was considered and rejected: the
gap between generations is noise next to the 128-vs-1075 error being corrected,
and two more enum members widen every exhaustive record for little gain.

### §3 — Inter-node fabric: a separate tier

Scale-out (node to node) is a different layer from scale-up and gets its own
table. It is deliberately **not** merged into `INTERCONNECT_SPECS`: different
physics, different efficiency semantics, and the two coexist in one cluster.

New in `constants.ts`:

```ts
export type FabricType =
  | 'ethernet-1600g'
  | 'ethernet-800g'
  | 'infiniband-xdr'
  | 'infiniband-ndr'
  | 'ethernet-400g'
  | 'ethernet-100g'
  | 'custom'
```

**Port speed is not node bandwidth.** The standard AI-node build is one NIC per
GPU, and collective libraries (RCCL/NCCL) stripe a pipeline stage handoff across
all of them. An 8-GPU node with 800GbE NICs therefore has
8 × 100 GB/s = 800 GB/s of scale-out bandwidth, not 100. Per the topology
decision below, `nicsPerNode` is **derived as `gpusPerNode`**; the fabric
dropdown label displays the resulting per-node aggregate so the assumption is
visible rather than hidden.

Per-node aggregate unidirectional bandwidth:

```
B = portGBps × gpusPerNode
```

**Efficiency formulas.** Let `L = max(0, log₂(1600 / B))`, the number of
halvings below a 1600 GB/s per-node reference.

```
prefillEfficiency = PP_BASE × (1 − 0.06·L − 0.020·L²) × fabricClassFactor
decodeEfficiency  = 0.99 − 0.01·L
```

with `PP_BASE = 0.95` (a pipeline stage boundary is never free even at
unlimited bandwidth) and `fabricClassFactor = 1.02` for InfiniBand, `1.00` for
Ethernet/RoCEv2 — credit-based flow control avoids the drop-and-recover tail
that PFC/ECN incurs under incast. Both efficiencies are clamped to `[0.05, 1.0]`.

The quadratic term encodes that degradation is superlinear: the first halving of
bandwidth costs little, the fourth is severe. The decode formula is near-flat by
design, per Finding B.

Resulting table at `gpusPerNode = 8`:

| fabric | GB/s/port | B (GB/s/node) | L | prefill | decode |
|---|---|---|---|---|---|
| 1.6TbE SONiC / RoCEv2 | 200 | 1600 | 0 | 0.95 | 0.99 |
| InfiniBand XDR 800G | 100 | 800 | 1 | 0.89 | 0.98 |
| 800GbE SONiC / RoCEv2 | 100 | 800 | 1 | 0.87 | 0.98 |
| InfiniBand NDR 400G | 50 | 400 | 2 | 0.78 | 0.97 |
| 400GbE RoCEv2 | 50 | 400 | 2 | 0.76 | 0.97 |
| 100GbE | 12.5 | 100 | 4 | 0.42 | 0.95 |

The table is **computed from the formulas, not hand-entered**, so the custom
entry and the presets cannot drift apart.

Ethernet presets are grounded in current hardware: Broadcom Tomahawk 6
(102.4 Tb/s, 128 × 800G or 64 × 1.6T) has shipped since October 2025 with
hardened SONiC available, and IEEE 802.3dj finalizes 1.6T optics mid-2026.

**Custom fabric.** A Zod schema in `src/utils/schemas.ts` mirroring the existing
`CustomGPUInput` pattern:

```ts
export const CustomFabricSchema = z.object({
  name: z.string().min(1),
  port_gbps: z.number().positive().max(10_000),
})
```

Efficiencies derive from the same formulas. Per the NIC decision, NIC count is
not exposed here either; the custom entry takes port speed only.

**Pipeline bubble.** Pipeline parallelism also loses throughput to fill/drain
independent of bandwidth:

```
bubbleEfficiency = M / (M + numNodes − 1),  M = max(batchSize, numNodes)
```

Omitting this flatters deep pipelines badly (4 nodes at batch 1 is a ~40%
throughput loss from bubbles alone). The `M` assumption — microbatch count
tracks batch size, floored at the stage count — is a heuristic and is documented
as such in the code comment.

**Disclosure.** Every efficiency figure in §2 and §3 is derived from bandwidth
by the documented formulas. None is a measured benchmark. This matches how
`INTERCONNECT_SPECS` was already constructed and must be stated in the code
comments; the UI should not present these as vendor-sourced numbers.

### §4 — Engine: compose, do not extend

Three structures were considered:

- **(a) Compose.** A new `calculateMultiNodeVRAM` derives a per-stage
  `InferenceVRAMBreakdown`, then delegates the intra-node split to the existing
  `calculateMultiGPUVRAM`.
- **(b) Extend** `calculateMultiGPUVRAM` with `numNodes` and fabric parameters.
  One code path, but every caller and the whole 19.7 KB test file change.
- **(c) Topology descriptor object** replacing the positional arguments.
  Cleanest long-term, widest blast radius.

**(a) is chosen.** `calculateMultiGPUVRAM` keeps its signature, its tests stay
green, and — decisively — its existing `numGPUs` 1–8 throw guard remains
*correct* under composition, because under (a) that bound is exactly the
per-node bound. Under (b) the guard would have to be relaxed and re-expressed.

```
calculateMultiNodeVRAM(singleGPU, model, gpuVramGB, gpusPerNode, numNodes,
                       intraNodeStrategy, gpu, fabric):

  1. numNodes === 1  → delegate straight to calculateMultiGPUVRAM (exact
     passthrough; today's results are unchanged by this feature)

  2. stageBreakdown = {
       modelWeights:      singleGPU.modelWeights.div(numNodes),
       kvCache:           singleGPU.kvCache.div(numNodes),      // Finding A
       activations:       singleGPU.activations.div(numNodes)
                            .mul(1 + PP_ACTIVATION_STASHING_OVERHEAD),
       frameworkOverhead: singleGPU.frameworkOverhead,          // per-process, not divided
     }

  3. inner = calculateMultiGPUVRAM(stageBreakdown, model, gpuVramGB,
                                   gpusPerNode, intraNodeStrategy, gpu)

  4. add inter-node communication overhead derived from FABRIC_SPECS,
     and populate the node-level efficiency fields
```

Callers to update: `src/hooks/useInferenceCalculation.ts:345` and
`src/workers/calculation.worker.ts:192`. `validateInterconnect`
(`multi-gpu.ts:348`) is called from both plus
`ShardingStrategySelector.tsx:26`, and must be passed `gpusPerNode` rather than
a total.

`MultiGPUVRAMBreakdown` (`types.ts:173`) gains:

```ts
numNodes: number
gpusPerNode: number
intraNodeEfficiency: number
interNodePrefillEfficiency: number
interNodeDecodeEfficiency: number
bubbleEfficiency: number
```

`numGPUs` is retained as the **total** (`gpusPerNode × numNodes`) so
`performance.ts`'s two multiply sites keep working unchanged in shape.
`scalingEfficiency` is retained as the **decode-path combined product**:

```
scalingEfficiency = intraNodeEfficiency × interNodeDecodeEfficiency × bubbleEfficiency
```

`performance.ts:155` (prefill) switches to the prefill product:

```
intraNodeEfficiency × interNodePrefillEfficiency × bubbleEfficiency
```

Both reduce to today's single-node values when `numNodes === 1`
(inter-node terms and bubble term are 1.0).

### §5 — State, URL, UI

**Topology model** (approved): two controls, GPUs-per-server and number of
servers. TP runs intra-node, PP inter-node, automatically. TP across a node
boundary is not offered — per-layer allreduce over a scale-out fabric an order
of magnitude slower than xGMI/NVLink is not a configuration worth modelling as
viable.

- `store/uiStore.ts` — `numGPUs` is reinterpreted as **GPUs per node**. New:
  `numNodes` (default 1), `interNodeFabric` (default `'ethernet-800g'`),
  `customFabric`. A persist-middleware migration is required; absent fields
  default such that existing persisted state is unchanged in meaning.
- `store/urlSerializer.ts` — `ng` keeps its key; its meaning shifts to
  per-node. **Back-compat holds**: an old link with `ng: 4` decodes to
  4 GPUs × 1 node = 4 total, producing an identical result. New keys `nn`,
  `fab`, `fabc` are all `.optional()`, so old URLs parse unchanged.
- `store/comparisonStore.ts` — the snapshot shape gains the node fields, or
  saved comparisons silently drop topology and compare unlike configurations.
- `components/inputs/GPUCountSelector.tsx` — relabel to "GPUs per server",
  add a server-count control, display the derived total.
- `components/inputs/ShardingStrategySelector.tsx` — becomes explicitly
  **intra-node**. Copy updated; the `recommendedMaxTPDegree` warning at line 41
  reads `gpusPerNode`, not the total.
- `components/inputs/InterNodeFabricSelector.tsx` — new, rendered only when
  `numNodes > 1`. Dropdown labels show per-node aggregate bandwidth.
- Output components showing the per-GPU breakdown gain a node-level summary
  (total GPUs, per-node aggregate fabric bandwidth, the two efficiencies).

### §6 — Testing and documentation

New `src/engines/multi-node.test.ts`:

- `numNodes === 1` produces results byte-identical to `calculateMultiGPUVRAM`
  today (the regression guard for the whole feature)
- composition math: weights, KV and activations divide by `numNodes` before
  the intra-node split
- KV-per-stage (Finding A) at both PP-only and composed topologies
- fabric efficiency applied at the correct roofline (Finding B): decode and
  prefill diverge for the same configuration
- bubble efficiency at batch 1 across 4 nodes
- custom fabric GB/s validation, including rejection of non-positive values
- an 8 × MI355X node produces no TP degradation warning

Updated: `multi-gpu.test.ts` PP KV assertions (lines 247, 297–301, 339);
`urlSerializer.test.ts` gains a back-compat round-trip asserting that a
pre-feature URL decodes to the same configuration.

Documentation, per the project's docs-sync rule: `CHANGELOG.md`, `README.md`
(GPU count 24 → 27, plus the multi-node feature), and
`docs/vram-calculation-pitfalls.md` gains an inter-node entry covering the
port-speed-vs-node-bandwidth trap and the TP-across-nodes anti-pattern.

### §7 — Sequencing

**Phase 1 — AMD hardware.** §1 and §2. Independently shippable and verifiable:
three GPU rows, the `InterconnectType` widening, the corrected Infinity Fabric
resolution. Derisks the enum change before topology work lands on top of it.

**Phase 2 — Multi-node.** §3 through §6, including both findings' fixes.

## Risks

- **Enum widening (§2)** touches every exhaustive `Record<InterconnectType, …>`.
  The compiler finds them; the risk is breadth, not subtlety. Phase 1 isolates
  it.
- **Efficiency figures are interpolations.** They will be directionally right
  and numerically approximate. The formulas and their anchors are documented so
  a future measurement can replace a constant without reverse-engineering
  intent.
- **`numGPUs` semantic shift** is the one genuinely subtle change. It is safe
  only because 1 node × N GPUs is arithmetically identical to today's N GPUs;
  the `numNodes === 1` passthrough test in §6 is the guard that keeps it so.
- **AMD spec figures** are from cross-checked secondary sources and must be
  confirmed against AMD's datasheet before Phase 1 merges. The dense-vs-sparse
  FP16 distinction is the specific trap.
