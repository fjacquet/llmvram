# GPU Scale-Up Domain Size — Design

**Status:** approved 2026-09-16
**Supersedes nothing.** Extends `2026-09-16-multi-node-scaling-design.md`.

## Problem

The app hardcodes a single number — 8 — as the maximum GPU count inside one
server. It appears in three places: the `GPUCountSelector` slider `max`, the
`recommendedMaxTPDegree` values in `INTERCONNECT_SPECS`, and the
`numGPUs < 1 || numGPUs > 8` guard in `src/engines/multi-gpu.ts:253`.

That number is a property of the HGX/UBB baseboard, not of GPUs in general.
A hardware cross-check against NVIDIA, AMD and Dell sources found it wrong
in one direction that matters, and carrying a data error alongside it.

### Confirmed findings

**NVL72 is unrepresentable.** NVIDIA GB300 NVL72 places 72 Blackwell Ultra
GPUs in a single NVLink domain, 130 TB/s of switch bandwidth, marketed as
acting like one GPU. Dell ships it as PowerEdge XE9712 and has delivered it
to CoreWeave; the Dell IR7000 rack scales to 256 Blackwell Ultra GPUs. The
app cannot express any of it. The closest it can do is nine 8-GPU servers
over Ethernet, understating inter-stage bandwidth by orders of magnitude.

**B200 capacity is wrong.** The database lists 192 GB. HGX B200 ships
1.44 TB across 8 GPUs — 180 GB each. 192 GB is the physical HBM3e stack
size before reserved capacity; 180 GB is the software-visible figure in
NVIDIA's OEM documentation and in every OEM listing (Dell XE9680L/XE9685L,
Lenovo ThinkSystem). For a tool that answers "does this fit", the
allocatable number is the correct one.

**The 8-pack is real.** Every SXM and OAM part sells only as an 8-GPU
baseboard, never as a single card: B200 (SXM6, no PCIe variant exists),
MI355X / MI350X / MI325X / MI300X (8 OAM on UBB 2.0, sold as a platform),
H100 SXM, A100 SXM. Dell's line confirms it — XE9680, XE9680L, XE9685L and
XE9785L are all exactly 8 GPUs. This is a procurement fact, not a modelling
one: per-GPU VRAM math is unaffected by how many you must buy. It is
recorded here because it motivated the investigation, and it is explicitly
NOT modelled (see Non-Goals).

### Findings investigated and rejected

Two claims made during the cross-check did not survive verification. They
are recorded so nobody re-derives them.

**PCIe GPUs are NOT blocked from multi-GPU.** `resolveInterconnect`
(`src/engines/multi-gpu.ts:341-344`) falls back on `tier`: an `interconnect`
of `'none'` on a `datacenter`-tier GPU resolves to `pcie-5`, not `'none'`.
L40S and RTX PRO 6000 are both `tier: 'datacenter'`, so `validateInterconnect`
returns `valid: true` with a soft warning above `recommendedMaxTPDegree`. The
`spec.type === 'none'` rejection branch at line 392 fires only for Apple
Silicon (`unified` maps to `none`), which is correct. Dell XE7745 ships 8
double-wide RTX PRO 6000; the app already permits that configuration.

**H200 is not over-bounded.** The 4-way limit belongs to H200 NVL PCIe, whose
NVLink bridge tops out at 4 GPUs with no switch. The database row
`nvidia-h200-141gb` is generic and carries `interconnect_options:
['nvlink-4', 'pcie-5']`, so it represents HGX H200, where 8-way is correct.
Not a bug — a row that means two products.

## Decisions

Recorded with their tradeoffs, as chosen.

1. **GPU-level field only.** No chassis entity, no platform presets. Fixes
   the domain-size bound; leaves PCIe chassis variation (RTX PRO 6000 is 8
   in an XE7745 but 2 in an R770) unmodelled. A user may still configure 8
   PCIe GPUs in a box that holds two.
2. **GB300 splits into two rows.** Same silicon, two platforms, two domain
   sizes; one field cannot say both.
3. **Clamping is silent.** No toast. A shared link that exceeds the new
   bound will display different numbers than the sender saw, with no
   explanation.
4. **B200 corrected to 180 GB.** Moves every existing B200 result by
   approximately 6%.
5. **H200 stays one row at 8.** The split to separate SXM and NVL PCIe rows
   was offered and declined. Risk: a user modelling H200 NVL PCIe can select
   8 GPUs and receive no hard stop, only the existing soft TP warning.
6. **Database ids do not change.** See Data.

## Schema

One new required field on `GPUSchema` in `src/utils/schemas.ts`:

```ts
max_gpus_per_node: z.number().int().positive()
```

Required, not optional: every row must state its bound, so a row added later
cannot silently inherit a default that happens to be wrong.

### Derivation rule

This comment travels with the field, so populating a new row is a lookup
rather than a judgment call:

> `max_gpus_per_node` = min(coherent interconnect limit, largest shipping
> chassis slot count). For parts with no coherent domain — anything on PCIe —
> the chassis bound alone.

Named `max_gpus_per_node` and not `max_coherent_domain` deliberately: PCIe
has no coherent domain, so a literal reading of that name yields 1 for every
PCIe card, re-creating the exact error this design removes.

### Relationship to `recommendedMaxTPDegree`

They are different things and both stay.

| | `max_gpus_per_node` | `recommendedMaxTPDegree` |
|---|---|---|
| Lives on | the GPU row | `INTERCONNECT_SPECS` |
| Means | cannot be built | can be built, scales badly |
| Enforcement | hard — bounds the slider | soft — emits a warning |

Eight RTX PRO 6000 in an XE7745 is buildable (`max_gpus_per_node: 8`) and a
poor tensor-parallel target (`recommendedMaxTPDegree: 4` for PCIe-5). Both
statements are true and the user should see both.

## Data

`scripts/fetch-gpus.ts` is the source; `src/data/gpus.json` is generated and
never hand-edited.

| `max_gpus_per_node` | GPUs |
|---|---|
| 72 | GB300 NVL72 (new row) |
| 8 | H100 PCIe, H100 SXM, H200, B200, GB300 (HGX B300), A100 PCIe, A100 SXM, L40S, RTX PRO 6000, RTX 6000 Ada, RTX 5090, RTX 4090, RTX 3090, MI355X, MI350X, MI325X, MI300X |
| 2 | GB10 (DGX Spark — two units linked over ConnectX-7) |
| 1 | GB300 Desktop Superchip, M1/M2/M3 Ultra, M1/M2/M3/M4/M5 Max |

Consumer RTX rows take 8 on the chassis rule: multi-GPU inference rigs with
eight PCIe cards exist, and the poor PCIe scaling is already expressed by
`recommendedMaxTPDegree`.

### Row changes

**`nvidia-gb300-288gb`** keeps its id, is relabelled to name the HGX B300
8-GPU baseboard, and takes `max_gpus_per_node: 8`.

**New row** for NVL72: a new id, `max_gpus_per_node: 72`, identical
`vram_gb`, `memory_bandwidth_gbps` and `fp16_tflops` to the row above — only
the domain differs.

**`nvidia-b200-192gb`** keeps its id, takes `vram_gb: 180`, and is renamed to
drop "192GB".

Ids are deliberately left stale rather than corrected. An id that no longer
exists fails to resolve; silent clamping cannot rescue it, because there is
no GPU to clamp against. A cosmetically wrong id costs nothing at runtime.

## Engine

`src/engines/multi-gpu.ts:253`:

```ts
if (numGPUs < 1 || numGPUs > 8) {
  throw new Error(`numGPUs must be between 1 and 8, got ${numGPUs}`)
}
```

becomes a check against a new exported constant `MAX_GPUS_PER_NODE = 72`.

The engine does not read `gpu.max_gpus_per_node`. The guard is a sanity
bound against absurd input; the per-GPU limit is a UI concern, enforced at
the store boundary. Keeping the engine GPU-agnostic here preserves its
purity and avoids a second source of truth for the same rule.

The fabric math needs no change. `perNodeFabricGBps = portGBps × gpusPerNode`
encodes one scale-out NIC per GPU, and that holds at 72: GB300 NVL72 carries
one 800 Gb/s ConnectX-8 port per GPU, 57.6 Tb/s per rack, which is exactly
`100 GB/s × 72 = 7200 GB/s`. At that bandwidth the prefill efficiency term
`L = max(0, log2(1600 / B))` clamps to zero and efficiency sits at its
`PP_BASE_EFFICIENCY` ceiling of 0.95. That is the cap behaving as designed,
not a fabricated number — inter-rack pipeline handoff is genuinely not the
bottleneck at 7.2 TB/s. `FABRIC_REFERENCE_GBPS = 1600` remains a reference
point and its comment stays accurate; it was never a maximum.

Tests asserting the old bound must move with it:
- `src/engines/multi-gpu.test.ts:431` — asserts a throw above 8
- `src/engines/multi-node.test.ts:169` — matches the literal message
  `/numGPUs must be between 1 and 8/`

## UI and store

**`GPUCountSelector`** reads `max` from the selected GPU's
`max_gpus_per_node` instead of the literal 8. Its tooltip currently claims 8
"is the size of a fully connected GPU domain in current hardware" — false
since GB200 NVL72 — and is rewritten to describe the selected GPU's actual
bound.

**`setNumGPUs`** in `src/store/uiStore.ts` clamps to the selected GPU's
bound. Selecting a different GPU re-clamps the current count.

**URL deserialization** clamps the same way on load. Silently, per decision 3.

Training mode is unaffected: `numGPUs` there is a total for ZeRO sharding,
and multi-node training remains a Non-Goal of the parent spec.

## Testing

- Schema: every row in `gpus.json` parses with the new required field.
- Data: the NVL72 row matches its HGX sibling on capacity, bandwidth and
  TFLOPS, differing only in `max_gpus_per_node` and id/name.
- Engine: `numGPUs: 72` computes; `numGPUs: 73` throws. The two existing
  boundary tests updated, not deleted.
- Store: clamping on `setNumGPUs`, on GPU switch, and on URL load.
- Regression: a 72-GPU NVL72 config produces a finite result with prefill
  efficiency at the `PP_BASE_EFFICIENCY` ceiling.
- Regression: B200 results move by the expected ratio 180/192.

## Non-Goals

- **Chassis modelling.** No slot counts, no power budgets, no GPU×chassis
  compatibility. The R770's 450 W cap on a 600 W card is not represented.
- **Purchase granularity.** The app will not say "you must buy 8 of these",
  even though that is true for every SXM and OAM part in the database.
- **Multi-node training.** Unchanged from the parent spec.
- **H200 platform split.** Declined; see decision 5.

## Risks

- A user can configure more PCIe GPUs than any real chassis holds, because
  only the interconnect-side bound is modelled (decision 1).
- Shared links exceeding a new bound change meaning silently (decision 3).
- Every saved B200 comparison shifts by ~6% with no migration notice
  (decision 4).
- H200 NVL PCIe users get no hard stop above 4 GPUs (decision 5).
- `max_gpus_per_node` is a point-in-time snapshot of shipping hardware. It
  will go stale the way `gpus.json` does, and has no automated source.
- The multi-GPU engine's scaling model is degree-independent, and this branch
  extrapolates it 9x past its old ceiling (8 to 72) without changing it. At
  TP=72 on DeepSeek R1: NCCL buffers reach 15.2 GB/GPU (~1 TB summed across
  the rack, which is not a physical allocation — it is the flat per-peer
  constant times 71 peers), replicated embeddings stay flat at 37.5 GB per
  GPU so TP-72 only ever delivers ~23x effective weight sharding despite
  claiming 72-way parallelism, and `scalingEfficiency` stays a constant 0.97,
  reporting ~70x throughput that the embedding-replication math above does
  not support. The branch accepts this rather than changing the engine,
  because the spec deliberately keeps the engine GPU-agnostic (Engine
  section) and names the `recommendedMaxTPDegree` soft warning — extended to
  pipeline parallelism in the same fix wave that recorded this risk — as the
  mitigation: past 8-way the UI now says "may have significant communication
  overhead" or "bubble overhead and stage imbalance" rather than silently
  endorsing the number. The NCCL term errs conservative in one direction only
  — it over-reports per-GPU VRAM, so it cannot produce a false "fits"; the
  risk is entirely in the optimistic throughput claim, not in a VRAM
  undercount.
