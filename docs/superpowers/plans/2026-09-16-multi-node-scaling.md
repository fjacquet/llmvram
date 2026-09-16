# Multi-Node Scaling + AMD Instinct MI350/MI325 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the AMD Instinct MI355X/MI350X/MI325X accelerators with a truthful Infinity Fabric model, then let the calculator scale inference across N identical servers (tensor parallel inside a node, pipeline parallel across nodes, with a selectable scale-out fabric).

**Architecture:** A new `calculateMultiNodeVRAM` *composes* over the existing `calculateMultiGPUVRAM` rather than extending it — it derives a per-stage VRAM breakdown (weights, KV and activations divided by node count) and delegates the intra-node split to the existing function, which keeps its signature, its tests, and its 1–8 GPU guard (now correctly a *per-node* bound). Scale-out bandwidth lives in a separate `FABRIC_SPECS` table, because node-to-node Ethernet/InfiniBand is different physics from intra-node NVLink/xGMI and the two coexist in one cluster.

**Tech Stack:** TypeScript strict (`noUncheckedIndexedAccess`), React 19, Zustand, Zod, decimal.js, Vitest + jsdom, Biome.

**Spec:** `docs/superpowers/specs/2026-09-16-multi-node-scaling-design.md` — read it before Task 1. It carries the rationale, the two defect findings, and the derivation of every efficiency constant used below.

## Global Constraints

- **Biome formatting is enforced:** 2-space indent, single quotes, **no semicolons** (ASI), 100-char line width. Unused imports and variables are **errors**, not warnings.
- **`src/data/gpus.json` is generated.** Never hand-edit it. Edit the `GPUS` array in `scripts/fetch-gpus.ts` and run `npm run refresh:gpus`.
- **`gpus.json` is grouped by vendor, not sorted alphabetically.** The `models.json` alphabetical-sort rule in CLAUDE.md does **not** apply to GPUs.
- **The `rtk` shell hook breaks `npm run lint` and `npm run typecheck`.** Use `rtk proxy npm run lint` and `rtk proxy npm run typecheck`. Plain `npx vitest run <file>` works normally.
- **`npm test` runs Vitest in watch mode and will hang.** Always use `npx vitest run <path>`.
- **Engines must stay pure** — no React, no DOM, no imports from `@components` or `@store`. They are designed for Web Worker offloading and `src/workers/calculation.worker.ts` imports them directly.
- **Zod schemas in `src/utils/schemas.ts` are the single source of truth.** Types re-export `z.infer<>`; all external data validates through Zod at the boundary.
- **`fp16_tflops` in the GPU database is DENSE throughput.** H100 = 989, B200 = 4500, MI300X = 1307. AMD's headline "4.6/5.0 PFLOPS FP16" marketing figures are with sparsity and must never be used.
- **Interconnect bandwidths in `INTERCONNECT_SPECS` are BIDIRECTIONAL.** NVLink-5's 1800 GB/s is NVIDIA's bidirectional per-GPU figure. Infinity Fabric's 1075 GB/s is AMD's bidirectional per-GPU figure for an 8-way fully-connected platform. A widely-quoted 538 GB/s is the *unidirectional* half of the same number — do not use it here.
- **`FABRIC_SPECS` port speeds are UNIDIRECTIONAL GB/s per port.** Different table, different convention, documented in place.
- **Every efficiency figure in this plan is derived from bandwidth by a documented formula, not measured.** Code comments must say so. Do not present them in the UI as vendor-sourced benchmarks.
- **Commit after every task.** End every commit message with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

# File Structure

**Phase 1 — AMD hardware**

| File | Responsibility | Action |
|---|---|---|
| `src/engines/types.ts` | `InterconnectType` gains `'infinity-fabric'` | Modify (line 155) |
| `src/engines/constants.ts` | `INTERCONNECT_SPECS` entry + label | Modify (lines ~199, ~404) |
| `src/engines/multi-gpu.ts` | `resolveInterconnect` stops aliasing AMD to PCIe-5; `validateInterconnect` label | Modify (lines 312, ~377) |
| `scripts/fetch-gpus.ts` | Three AMD Instinct rows | Modify (after line 205) |
| `src/data/gpus.json` | Generated output | Regenerate |

**Phase 2 — Multi-node**

| File | Responsibility | Action |
|---|---|---|
| `src/engines/fabric.ts` | Scale-out fabric table + the three efficiency formulas. One responsibility: turn bandwidth into efficiency. | **Create** |
| `src/engines/multi-node.ts` | `calculateMultiNodeVRAM` — node-level decomposition, delegates intra-node to `multi-gpu.ts` | **Create** |
| `src/engines/types.ts` | `FabricType`; `MultiGPUVRAMBreakdown` gains the node dimension | Modify |
| `src/engines/multi-gpu.ts` | PP KV-per-stage fix; populate new breakdown fields | Modify |
| `src/engines/performance.ts` | Prefill roofline uses the prefill-specific efficiency | Modify (line ~155) |
| `src/utils/schemas.ts` | `CustomFabricSchema` | Modify |
| `src/store/uiStore.ts` | `numGPUs` means per-node; adds `numNodes`, `interNodeFabric`, `customFabric` | Modify |
| `src/store/urlSerializer.ts` | New optional keys `nn`, `fab`, `fabc` | Modify |
| `src/hooks/useURLSync.ts` | Restore the new keys | Modify (line ~110) |
| `src/store/comparisonStore.ts` | Snapshot carries topology | Modify |
| `src/components/inputs/GPUCountSelector.tsx` | "GPUs per server" + server count | Modify |
| `src/components/inputs/NodeCountSelector.tsx` | Server-count slider | **Create** |
| `src/components/inputs/InterNodeFabricSelector.tsx` | Fabric dropdown + custom entry | **Create** |
| `src/components/inputs/ShardingStrategySelector.tsx` | Reads per-node, copy says "intra-node" | Modify (lines 15–41, ~130) |
| `src/hooks/useInferenceCalculation.ts` | Call the node-aware entry point | Modify (line 345) |
| `src/workers/calculation.worker.ts` | Same, worker side | Modify (line 192) |

`fabric.ts` and `multi-node.ts` are split because they change for different reasons: `fabric.ts` changes when network hardware changes, `multi-node.ts` when the parallelism model changes.

---

# Phase 1 — AMD Instinct hardware

Independently shippable. Ship and verify this before starting Phase 2.

### Task 1: Infinity Fabric becomes a real interconnect

Today `resolveInterconnect` aliases `'infinity-fabric'` to `'pcie-5'` (128 GB/s, 0.78 efficiency, max TP degree 4). Real 8-way xGMI is 1075 GB/s bidirectional. The alias inflates per-GPU VRAM for every AMD card and fires a spurious TP degradation warning.

**Files:**
- Modify: `src/engines/types.ts:155`
- Modify: `src/engines/constants.ts:199-230`, `src/engines/constants.ts:404-412`
- Modify: `src/engines/multi-gpu.ts:312`, `src/engines/multi-gpu.ts:376-378`
- Test: `src/engines/multi-gpu.test.ts`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `InterconnectType` gains the `'infinity-fabric'` member. `INTERCONNECT_SPECS['infinity-fabric']` exists with `bandwidthGBps: 1075`, `recommendedMaxTPDegree: 8`, `tpScalingEfficiency: 0.93`. Task 2 and every Phase 2 task depend on this member existing.

- [ ] **Step 1: Write the failing tests**

Append to `src/engines/multi-gpu.test.ts`. Match the existing import block at the top of that file — it already imports `resolveInterconnect`, `validateInterconnect` and `INTERCONNECT_SPECS`; add any that are missing rather than duplicating.

```ts
describe('Infinity Fabric interconnect', () => {
  const mi300x: GPU = {
    id: 'amd-mi300x',
    name: 'AMD MI300X',
    manufacturer: 'amd',
    vram_gb: 192,
    memory_bandwidth_gbps: 5300,
    memory_type: 'HBM3',
    bus_width: 8192,
    fp16_tflops: 1307,
    fp32_tflops: 163,
    tdp_watts: 750,
    interconnect: 'infinity-fabric',
    tier: 'datacenter',
  }

  it('resolves to its own type, not pcie-5', () => {
    expect(resolveInterconnect(mi300x)).toBe('infinity-fabric')
  })

  it('carries AMD bidirectional bandwidth and 8-way TP support', () => {
    const spec = INTERCONNECT_SPECS['infinity-fabric']
    expect(spec.bandwidthGBps).toBe(1075)
    expect(spec.recommendedMaxTPDegree).toBe(8)
    expect(spec.tpScalingEfficiency).toBe(0.93)
  })

  it('sits between NVLink-4 and NVLink-5 in scaling efficiency', () => {
    expect(INTERCONNECT_SPECS['infinity-fabric'].tpScalingEfficiency).toBeGreaterThan(
      INTERCONNECT_SPECS['nvlink-4'].tpScalingEfficiency,
    )
    expect(INTERCONNECT_SPECS['infinity-fabric'].tpScalingEfficiency).toBeLessThan(
      INTERCONNECT_SPECS['nvlink-5'].tpScalingEfficiency,
    )
  })

  it('does not warn at 8-way tensor parallel', () => {
    const result = validateInterconnect(mi300x, 8, 'tensor-parallel')
    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
  })

  it('names the interconnect in a warning rather than printing the enum value', () => {
    const pcie4GPU: GPU = { ...mi300x, interconnect: 'pcie-4', name: 'Test PCIe4' }
    const result = validateInterconnect(pcie4GPU, 8, 'tensor-parallel')
    expect(result.warning).toContain('PCIe 4.0')
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/multi-gpu.test.ts`
Expected: FAIL. The first test reports received `'pcie-5'`; the `INTERCONNECT_SPECS['infinity-fabric']` tests fail to compile because the key does not exist on the `Record<InterconnectType, …>`.

- [ ] **Step 3: Widen `InterconnectType`**

In `src/engines/types.ts`, replace lines 146–155 (the doc comment and the type):

```ts
/**
 * GPU interconnect types with different bandwidth characteristics
 *
 * All bandwidths are BIDIRECTIONAL per-GPU figures, matching how NVIDIA and AMD
 * publish them. Do not mix in unidirectional numbers.
 *
 * - nvlink-4: 4th gen NVLink (900 GB/s)
 * - nvlink-5: 5th gen NVLink (1800 GB/s)
 * - infinity-fabric: AMD xGMI, 8-way fully connected (1075 GB/s)
 * - pcie-4: PCIe 4.0 x16 (64 GB/s)
 * - pcie-5: PCIe 5.0 x16 (128 GB/s)
 * - none: No multi-GPU support (single GPU only)
 */
export type InterconnectType =
  | 'nvlink-4'
  | 'nvlink-5'
  | 'infinity-fabric'
  | 'pcie-4'
  | 'pcie-5'
  | 'none'
```

- [ ] **Step 4: Run the typechecker to find every exhaustive record**

Run: `rtk proxy npm run typecheck`
Expected: FAIL. TypeScript names each `Record<InterconnectType, …>` that is now missing the `'infinity-fabric'` key. `INTERCONNECT_SPECS` in `src/engines/constants.ts` is the one this plan knows about — if the compiler names others, they get the same treatment and are listed in the commit message.

- [ ] **Step 5: Add the `INTERCONNECT_SPECS` entry**

In `src/engines/constants.ts`, insert after the `'nvlink-5'` entry (ends line 211), before `'pcie-4'`:

```ts
  'infinity-fabric': {
    type: 'infinity-fabric',
    bandwidthGBps: 1075,
    // AMD's aggregate per-GPU GPU-to-GPU figure for an 8-way fully connected
    // MI350-series platform, bidirectional — the same convention as the NVLink
    // rows above. (A widely quoted 538 GB/s is the unidirectional half.)
    recommendedMaxTPDegree: 8,
    tpScalingEfficiency: 0.93,
    // Interpolated log-linearly between the adjacent anchors in this table:
    // NVLink-4 (900 GB/s, 0.92) and NVLink-5 (1800 GB/s, 0.97), i.e. +0.05 per
    // doubling. 0.92 + 0.05 * log2(1075/900) = 0.933. Derived, not measured —
    // as with every other row here.
  },
```

- [ ] **Step 6: Update the human-readable label**

In `src/engines/constants.ts`, in `INTERCONNECT_LABELS` (line ~404), replace the bare `'infinity-fabric': 'Infinity Fabric',` line:

```ts
  'infinity-fabric': 'Infinity Fabric — 1075 GB/s',
```

- [ ] **Step 7: Stop aliasing AMD to PCIe-5**

In `src/engines/multi-gpu.ts`, replace lines 311–312:

```ts
  // AMD Infinity Fabric (xGMI) — its own tier, not a PCIe stand-in
  if (interconnect === 'infinity-fabric') return 'infinity-fabric'
```

Leave the `'pcie-5'` direct mapping (line 303) and the datacenter tier fallback (line 316) exactly as they are. Both are still correct and are relied on by H200's `interconnect_options` and by L40S / RTX PRO 6000 respectively.

- [ ] **Step 8: Fix the warning label so it never prints a raw enum value**

In `src/engines/multi-gpu.ts`, replace the `interconnectName` ternary (lines ~376–378) inside `validateInterconnect`:

```ts
    const interconnectName = INTERCONNECT_LABELS[spec.type] ?? spec.type
```

Add `INTERCONNECT_LABELS` to the existing import from `./constants` at the top of the file.

- [ ] **Step 9: Run the full engine suite**

Run: `npx vitest run src/engines/`
Expected: PASS. Note that MI300X multi-GPU results now change — they were previously pessimistic. If an existing assertion pinned an MI300X number computed under the PCIe-5 alias, update it and say so in the commit message; do not revert the fix to keep an assertion green.

- [ ] **Step 10: Lint and typecheck**

Run: `rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: both clean.

- [ ] **Step 11: Commit**

```bash
rtk git add src/engines/types.ts src/engines/constants.ts src/engines/multi-gpu.ts src/engines/multi-gpu.test.ts
rtk git commit -m "$(cat <<'EOF'
fix(engines): model Infinity Fabric as its own interconnect tier

resolveInterconnect aliased AMD Infinity Fabric to pcie-5 (128 GB/s, 0.78
efficiency, max TP degree 4). Real 8-way xGMI is 1075 GB/s bidirectional.
The alias inflated per-GPU VRAM for every AMD card and fired a spurious
"TP with 8 GPUs may degrade" warning.

Adds 'infinity-fabric' to InterconnectType with tpScalingEfficiency 0.93,
interpolated log-linearly between the NVLink-4 and NVLink-5 anchors in the
same table. PCIe-5 keeps its direct mapping and its datacenter tier
fallback unchanged.

Also routes validateInterconnect's warning through INTERCONNECT_LABELS so
it can never print a raw enum value.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add MI355X, MI350X and MI325X

**Files:**
- Modify: `scripts/fetch-gpus.ts:192-205` (the AMD Datacenter block) and the source comment at line 11
- Regenerate: `src/data/gpus.json`
- Test: `src/utils/gpus.test.ts`
- Modify: `README.md`, `CHANGELOG.md`

**Interfaces:**
- Consumes: `'infinity-fabric'` as a resolvable `InterconnectType` (Task 1).
- Produces: GPU IDs `amd-mi355x`, `amd-mi350x`, `amd-mi325x` in the database. Nothing downstream imports these by ID; they are data.

- [ ] **Step 1: Verify every number against AMD's datasheet**

Before writing anything, fetch AMD's official MI350-series datasheet and the MI325X datasheet and confirm each field below. The figures in this plan are from cross-checked secondary sources.

The specific trap: **AMD publishes FP16 with sparsity** ("4.6 PFLOPS", "5.0 PFLOPS"). This database stores **dense**. Dense FP16 is half the sparse figure and equals the dense FP8 figure divided by two. MI355X dense FP8 is 5033 TFLOPS, so dense FP16 is 2516.

If a confirmed figure differs from the table below, use the confirmed figure and note the correction in the commit message.

- [ ] **Step 2: Write the failing test**

Append to `src/utils/gpus.test.ts`:

```ts
describe('AMD Instinct MI350 / MI325 series', () => {
  it('includes MI355X with 288 GB HBM3E at 8 TB/s', () => {
    const gpu = gpus.find((g) => g.id === 'amd-mi355x')
    expect(gpu?.vram_gb).toBe(288)
    expect(gpu?.memory_bandwidth_gbps).toBe(8000)
    expect(gpu?.memory_type).toBe('HBM3E')
    expect(gpu?.tdp_watts).toBe(1400)
  })

  it('includes MI350X as the air-cooled sibling: same memory, lower TBP', () => {
    const mi350x = gpus.find((g) => g.id === 'amd-mi350x')
    const mi355x = gpus.find((g) => g.id === 'amd-mi355x')
    expect(mi350x?.vram_gb).toBe(mi355x?.vram_gb)
    expect(mi350x?.memory_bandwidth_gbps).toBe(mi355x?.memory_bandwidth_gbps)
    expect(mi350x?.tdp_watts).toBe(1000)
    expect(mi350x?.fp16_tflops ?? 0).toBeLessThan(mi355x?.fp16_tflops ?? 0)
  })

  it('includes MI325X with 256 GB and MI300X compute', () => {
    const mi325x = gpus.find((g) => g.id === 'amd-mi325x')
    const mi300x = gpus.find((g) => g.id === 'amd-mi300x')
    expect(mi325x?.vram_gb).toBe(256)
    expect(mi325x?.memory_bandwidth_gbps).toBe(6000)
    expect(mi325x?.fp16_tflops).toBe(mi300x?.fp16_tflops)
  })

  it('stores dense FP16, not AMD sparse marketing figures', () => {
    // Sparse would be ~5033 for MI355X. Anything above 3000 means a sparse
    // number leaked into the database.
    for (const gpu of gpus.filter((g) => g.manufacturer === 'amd')) {
      expect(gpu.fp16_tflops ?? 0).toBeLessThan(3000)
    }
  })

  it('puts every AMD accelerator on Infinity Fabric', () => {
    for (const gpu of gpus.filter((g) => g.manufacturer === 'amd')) {
      expect(gpu.interconnect).toBe('infinity-fabric')
    }
  })
})
```

If `src/utils/gpus.test.ts` does not already import the GPU array, add the import the rest of that file uses — do not introduce a second import path for the same data.

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run src/utils/gpus.test.ts`
Expected: FAIL, `expected undefined to be 288` — the rows do not exist yet.

- [ ] **Step 4: Add the three rows to the generator**

In `scripts/fetch-gpus.ts`, inside the `// AMD Datacenter` block, insert these **before** the existing `amd-mi300x` entry (newest first, matching how the NVIDIA block is ordered):

```ts
  {
    id: 'amd-mi355x',
    name: 'AMD Instinct MI355X',
    manufacturer: 'amd',
    vram_gb: 288,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3E',
    bus_width: 8192,
    // Dense FP16. AMD's headline 5.0 PFLOPS figure is with sparsity.
    fp16_tflops: 2516,
    fp32_tflops: 157,
    tdp_watts: 1400,
    interconnect: 'infinity-fabric',
    tier: 'datacenter',
  },
  {
    id: 'amd-mi350x',
    name: 'AMD Instinct MI350X',
    manufacturer: 'amd',
    vram_gb: 288,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3E',
    bus_width: 8192,
    // Same die and memory as MI355X; the ~9% throughput gap is clocks,
    // enabled by 1400 W liquid vs 1000 W air cooling.
    fp16_tflops: 2300,
    fp32_tflops: 144,
    tdp_watts: 1000,
    interconnect: 'infinity-fabric',
    tier: 'datacenter',
  },
  {
    id: 'amd-mi325x',
    name: 'AMD Instinct MI325X',
    manufacturer: 'amd',
    vram_gb: 256,
    memory_bandwidth_gbps: 6000,
    memory_type: 'HBM3E',
    bus_width: 8192,
    // Same CDNA 3 compute core as MI300X, more memory.
    fp16_tflops: 1307,
    fp32_tflops: 163,
    tdp_watts: 1000,
    interconnect: 'infinity-fabric',
    tier: 'datacenter',
  },
```

Also extend the source comment at line 11:

```ts
 * - AMD MI300X: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
 * - AMD MI325X: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf
 * - AMD MI350X/MI355X: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf
```

- [ ] **Step 5: Regenerate the JSON**

Run: `npm run refresh:gpus`
Expected: `✓ All GPUs valid` then `✓ Wrote 27 GPUs to src/data/gpus.json`. The count is the check — 24 before, 27 after.

- [ ] **Step 6: Run the test to verify it passes**

Run: `npx vitest run src/utils/gpus.test.ts`
Expected: PASS.

- [ ] **Step 7: Update the docs**

In `README.md`, change the GPU count from 24 to 27 and add the three accelerators wherever the GPU database is described. In `CHANGELOG.md`, add under an Unreleased heading (match the file's existing format):

```markdown
### Added
- AMD Instinct MI355X (288 GB HBM3E, 8 TB/s, 1400 W), MI350X (288 GB, 1000 W air-cooled) and MI325X (256 GB HBM3E, 6 TB/s) to the GPU database.

### Fixed
- AMD Infinity Fabric is now modelled as its own interconnect tier (1075 GB/s, 8-way TP) instead of being aliased to PCIe 5.0 (128 GB/s). Multi-GPU VRAM estimates for AMD accelerators were previously inflated, and an 8-GPU AMD node raised a spurious tensor-parallel degradation warning.
```

- [ ] **Step 8: Full suite, lint, typecheck, build**

Run: `npx vitest run && rtk proxy npm run lint && rtk proxy npm run typecheck && npm run build`
Expected: all clean.

- [ ] **Step 9: Commit**

```bash
rtk git add scripts/fetch-gpus.ts src/data/gpus.json src/utils/gpus.test.ts README.md CHANGELOG.md
rtk git commit -m "$(cat <<'EOF'
feat(data): add AMD Instinct MI355X, MI350X and MI325X

MI355X: 288 GB HBM3E, 8 TB/s, 1400 W liquid.
MI350X: same die and memory, 1000 W air, ~9% lower clocks.
MI325X: 256 GB HBM3E, 6 TB/s, MI300X CDNA 3 compute.

fp16_tflops is dense throughout, matching the database convention
(H100 989, B200 4500). AMD's headline FP16 PFLOPS figures are with
sparsity and are not used.

GPU count 24 -> 27.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

**Phase 1 ends here. Ship and verify before continuing.**

---

# Phase 2 — Multi-node inference

### Task 3: Pipeline parallelism stops replicating the whole KV cache

`calculateKVCacheVRAM` (`kv-cache.ts:73`) multiplies by `num_hidden_layers`, so `singleGPU.kvCache` is the total across **all** layers. Pipeline parallelism splits layers across stages, so each stage holds KV for only its own layers. `multi-gpu.ts:147` assigns the full figure to every stage — 8× too high at 8-way PP.

This is fixed before the node dimension lands, because Task 5 composes PP over TP and the error would compound: weights shrink by the node count while KV would not shrink at all, making long-context multi-node configurations report as not fitting when they fit comfortably.

**Files:**
- Modify: `src/engines/multi-gpu.ts:140-152`
- Modify: `src/engines/multi-gpu.test.ts:247`, `:297-301`, `:339`

**Interfaces:**
- Consumes: nothing new.
- Produces: `calculatePipelineParallelVRAM` now returns `perGPU.kvCache === singleGPU.kvCache / numGPUs`. Task 5 relies on this so that its own division by `numNodes` does not double-count.

- [ ] **Step 1: Write the failing test**

Append to `src/engines/multi-gpu.test.ts`:

```ts
describe('pipeline parallel KV cache sharding', () => {
  it('divides the KV cache across stages, because layers are split', () => {
    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      80,
      4,
      'pipeline-parallel',
      h100SXM,
    )
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())
  })

  it('matches tensor parallel on the KV term — both shard it, by different axes', () => {
    const tp = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100SXM)
    const pp = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'pipeline-parallel', h100SXM)
    expect(pp.perGPU.kvCache.toString()).toBe(tp.perGPU.kvCache.toString())
  })
})
```

Reuse the `singleGPU`, `llama70b` and `h100SXM` fixtures already defined at the top of `multi-gpu.test.ts`. If the existing fixture names differ, use the existing names — do not create parallel fixtures.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/engines/multi-gpu.test.ts`
Expected: FAIL. The PP KV value is 4× the expected figure.

- [ ] **Step 3: Apply the fix**

In `src/engines/multi-gpu.ts`, replace lines 145–147:

```ts
  // KV cache is sharded by layer: PP assigns a contiguous slice of layers to
  // each stage, and the KV cache is per-layer, so a stage holds only its own
  // layers' cache. (singleGPU.kvCache is the all-layer total — kv-cache.ts
  // multiplies by num_hidden_layers.)
  const kvCachePerGPU = singleGPU.kvCache.div(numGPUs)
```

- [ ] **Step 4: Update the three stale assertions**

In `src/engines/multi-gpu.test.ts`:

- Line 247 — `expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.toString())` becomes `.toBe(singleGPU.kvCache.div(4).toString())`. Read the surrounding `it()` to confirm the GPU count in that case is 4; use whatever count it actually passes.
- Line 298 — `expect(ppResult.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.toString())` becomes `.toBe(singleGPU.kvCache.div(4).toString())`.
- Line 301 — the `kvRatio` assertion compared PP against TP expecting a 4× ratio. They are now equal; change it to assert `kvRatio.toNumber()` is `1`, and update the surrounding test name if it says PP uses more KV than TP.
- Line 339 — same substitution as line 247, using that test's own GPU count.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run src/engines/multi-gpu.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add src/engines/multi-gpu.ts src/engines/multi-gpu.test.ts
rtk git commit -m "$(cat <<'EOF'
fix(engines): shard the KV cache across pipeline stages

singleGPU.kvCache is the all-layer total (kv-cache.ts multiplies by
num_hidden_layers). Pipeline parallelism gives each stage a contiguous
slice of layers, so a stage holds only its own layers' cache. The
pipeline path assigned the full figure to every stage, overstating the
KV term by the stage count.

The old comment conflated "full cache for the sequence" (true) with
"full cache for all layers" (false).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Scale-out fabric table and efficiency formulas

**Files:**
- Create: `src/engines/fabric.ts`
- Create: `src/engines/fabric.test.ts`
- Modify: `src/engines/types.ts` (add `FabricType`, `FabricSpec`)
- Modify: `src/utils/schemas.ts` (add `CustomFabricSchema`)

**Interfaces:**
- Consumes: nothing new.
- Produces — Task 5 and Task 8 depend on exactly these signatures:
  - `type FabricType = 'ethernet-1600g' | 'ethernet-800g' | 'infiniband-xdr' | 'infiniband-ndr' | 'ethernet-400g' | 'ethernet-100g' | 'custom'`
  - `interface FabricSpec { type: FabricType; label: string; portGBps: number; classFactor: number }`
  - `FABRIC_SPECS: Record<Exclude<FabricType, 'custom'>, FabricSpec>`
  - `perNodeFabricGBps(portGBps: number, gpusPerNode: number): number`
  - `fabricPrefillEfficiency(perNodeGBps: number, classFactor: number): number`
  - `fabricDecodeEfficiency(perNodeGBps: number): number`
  - `pipelineBubbleEfficiency(batchSize: number, numNodes: number): number`
  - `resolveFabricSpec(type: FabricType, custom: CustomFabricInput | null): FabricSpec`
  - `CustomFabricSchema` / `type CustomFabricInput` from `@utils/schemas`

- [ ] **Step 1: Add the types**

In `src/engines/types.ts`, append after the `InterconnectSpec` interface (line ~166):

```ts
/**
 * Scale-out (node-to-node) fabric types
 *
 * Distinct from InterconnectType, which is scale-up (GPU-to-GPU inside one
 * chassis). Both exist simultaneously in a real cluster: NVLink or Infinity
 * Fabric inside a server, Ethernet or InfiniBand between servers.
 *
 * Unlike InterconnectType's bidirectional figures, portGBps is UNIDIRECTIONAL
 * per port, because that is how network hardware is specified.
 */
export type FabricType =
  | 'ethernet-1600g'
  | 'ethernet-800g'
  | 'infiniband-xdr'
  | 'infiniband-ndr'
  | 'ethernet-400g'
  | 'ethernet-100g'
  | 'custom'

export interface FabricSpec {
  type: FabricType
  label: string
  /** Unidirectional bandwidth per port, GB/s */
  portGBps: number
  /**
   * Efficiency multiplier for the fabric class. InfiniBand's credit-based flow
   * control avoids the drop-and-recover tail that RoCEv2's PFC/ECN incurs under
   * incast, so it edges out Ethernet at the same line rate.
   */
  classFactor: number
}
```

- [ ] **Step 2: Add the custom-fabric Zod schema**

In `src/utils/schemas.ts`, append after the GPU schema block:

```ts
/**
 * User-specified scale-out fabric
 *
 * port_gbps is unidirectional GB/s per port. NIC count is not asked for: it is
 * derived as one NIC per GPU, the standard AI-node build.
 */
export const CustomFabricSchema = z.object({
  name: z.string().min(1),
  port_gbps: z.number().positive().max(10_000),
})

export type CustomFabricInput = z.infer<typeof CustomFabricSchema>
```

- [ ] **Step 3: Write the failing tests**

Create `src/engines/fabric.test.ts`:

```ts
import {
  FABRIC_SPECS,
  fabricDecodeEfficiency,
  fabricPrefillEfficiency,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
  resolveFabricSpec,
} from '@engines/fabric'
import { describe, expect, it } from 'vitest'

describe('perNodeFabricGBps', () => {
  it('multiplies port speed by one NIC per GPU', () => {
    expect(perNodeFabricGBps(100, 8)).toBe(800)
    expect(perNodeFabricGBps(200, 8)).toBe(1600)
    expect(perNodeFabricGBps(50, 4)).toBe(200)
  })
})

describe('fabricPrefillEfficiency', () => {
  it('is capped by the pipeline stage-boundary base at the reference bandwidth', () => {
    expect(fabricPrefillEfficiency(1600, 1.0)).toBeCloseTo(0.95, 2)
  })

  it('degrades superlinearly as bandwidth halves', () => {
    const at1600 = fabricPrefillEfficiency(1600, 1.0)
    const at800 = fabricPrefillEfficiency(800, 1.0)
    const at400 = fabricPrefillEfficiency(400, 1.0)
    expect(at1600 - at800).toBeLessThan(at800 - at400)
  })

  it('matches the documented table values', () => {
    expect(fabricPrefillEfficiency(800, 1.0)).toBeCloseTo(0.874, 3)
    expect(fabricPrefillEfficiency(400, 1.0)).toBeCloseTo(0.76, 3)
    expect(fabricPrefillEfficiency(100, 1.0)).toBeCloseTo(0.418, 3)
  })

  it('gives InfiniBand an edge over Ethernet at the same line rate', () => {
    expect(fabricPrefillEfficiency(800, 1.02)).toBeGreaterThan(fabricPrefillEfficiency(800, 1.0))
  })

  it('never exceeds 1 or falls below the floor', () => {
    expect(fabricPrefillEfficiency(100_000, 1.02)).toBeLessThanOrEqual(1)
    expect(fabricPrefillEfficiency(0.5, 1.0)).toBeGreaterThanOrEqual(0.05)
  })
})

describe('fabricDecodeEfficiency', () => {
  it('stays near-flat: a decode hop ships kilobytes and is latency-bound', () => {
    expect(fabricDecodeEfficiency(1600)).toBeCloseTo(0.99, 2)
    expect(fabricDecodeEfficiency(100)).toBeCloseTo(0.95, 2)
  })

  it('is always far above the prefill efficiency at the same bandwidth', () => {
    expect(fabricDecodeEfficiency(100)).toBeGreaterThan(fabricPrefillEfficiency(100, 1.0))
  })
})

describe('pipelineBubbleEfficiency', () => {
  it('is 1.0 for a single node — no pipeline, no bubble', () => {
    expect(pipelineBubbleEfficiency(1, 1)).toBe(1)
  })

  it('costs real throughput at batch 1 across 4 nodes', () => {
    // M = max(1, 4) = 4, so 4 / (4 + 3) = 0.571
    expect(pipelineBubbleEfficiency(1, 4)).toBeCloseTo(0.571, 3)
  })

  it('improves as batch size feeds more microbatches into the pipeline', () => {
    expect(pipelineBubbleEfficiency(64, 4)).toBeGreaterThan(pipelineBubbleEfficiency(4, 4))
  })
})

describe('resolveFabricSpec', () => {
  it('returns the preset for a known type', () => {
    expect(resolveFabricSpec('ethernet-800g', null)).toEqual(FABRIC_SPECS['ethernet-800g'])
  })

  it('builds a spec from custom input', () => {
    const spec = resolveFabricSpec('custom', { name: 'Lab fabric', port_gbps: 25 })
    expect(spec.portGBps).toBe(25)
    expect(spec.label).toContain('Lab fabric')
    expect(spec.classFactor).toBe(1.0)
  })

  it('falls back to the 800G Ethernet preset when custom is selected with no input', () => {
    expect(resolveFabricSpec('custom', null)).toEqual(FABRIC_SPECS['ethernet-800g'])
  })
})

describe('FABRIC_SPECS', () => {
  it('orders 1.6TbE fastest and 100GbE slowest', () => {
    expect(FABRIC_SPECS['ethernet-1600g'].portGBps).toBe(200)
    expect(FABRIC_SPECS['ethernet-100g'].portGBps).toBe(12.5)
  })

  it('marks only the InfiniBand entries with the class bonus', () => {
    expect(FABRIC_SPECS['infiniband-xdr'].classFactor).toBe(1.02)
    expect(FABRIC_SPECS['infiniband-ndr'].classFactor).toBe(1.02)
    expect(FABRIC_SPECS['ethernet-800g'].classFactor).toBe(1.0)
  })
})
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `npx vitest run src/engines/fabric.test.ts`
Expected: FAIL — `Cannot find module '@engines/fabric'`.

- [ ] **Step 5: Implement the module**

Create `src/engines/fabric.ts`:

```ts
import type { CustomFabricInput } from '@utils/schemas'
import type { FabricSpec, FabricType } from './types'

/**
 * Reference per-node bandwidth for the efficiency formulas, GB/s
 *
 * 1600 GB/s is an 8-GPU node on 1.6TbE — currently the fastest mainstream
 * scale-out build. Efficiency is expressed as a penalty in halvings below it.
 */
export const FABRIC_REFERENCE_GBPS = 1600

/**
 * Efficiency ceiling for a pipeline stage boundary
 *
 * A stage handoff is never free, even at unlimited bandwidth: there is a
 * serialization point and a synchronization. Caps the prefill formula.
 */
export const PP_BASE_EFFICIENCY = 0.95

/** Floor applied to both efficiency formulas, to keep pathological inputs sane */
const EFFICIENCY_FLOOR = 0.05

/**
 * Scale-out fabric presets
 *
 * portGBps is UNIDIRECTIONAL GB/s per port: 800 Gb/s = 100 GB/s.
 *
 * Grounded in current hardware — Broadcom Tomahawk 6 (102.4 Tb/s, 128x800G or
 * 64x1.6T) has shipped since October 2025 with hardened SONiC available, and
 * IEEE 802.3dj finalizes 1.6T optics mid-2026.
 */
export const FABRIC_SPECS: Record<Exclude<FabricType, 'custom'>, FabricSpec> = {
  'ethernet-1600g': {
    type: 'ethernet-1600g',
    label: '1.6TbE (SONiC / RoCEv2)',
    portGBps: 200,
    classFactor: 1.0,
  },
  'infiniband-xdr': {
    type: 'infiniband-xdr',
    label: 'InfiniBand XDR 800G',
    portGBps: 100,
    classFactor: 1.02,
  },
  'ethernet-800g': {
    type: 'ethernet-800g',
    label: '800GbE (SONiC / RoCEv2)',
    portGBps: 100,
    classFactor: 1.0,
  },
  'infiniband-ndr': {
    type: 'infiniband-ndr',
    label: 'InfiniBand NDR 400G',
    portGBps: 50,
    classFactor: 1.02,
  },
  'ethernet-400g': {
    type: 'ethernet-400g',
    label: '400GbE (RoCEv2)',
    portGBps: 50,
    classFactor: 1.0,
  },
  'ethernet-100g': {
    type: 'ethernet-100g',
    label: '100GbE',
    portGBps: 12.5,
    classFactor: 1.0,
  },
}

/**
 * Per-node aggregate scale-out bandwidth
 *
 * The standard AI-node build is one NIC per GPU, and collective libraries
 * (NCCL/RCCL) stripe a pipeline stage handoff across all of them. So node
 * bandwidth is port speed times GPU count, not a single port's speed — a
 * distinction worth three-to-eight times the answer.
 */
export function perNodeFabricGBps(portGBps: number, gpusPerNode: number): number {
  return portGBps * gpusPerNode
}

/** Halvings below the reference bandwidth; 0 at or above it */
function halvingsBelowReference(perNodeGBps: number): number {
  if (perNodeGBps <= 0) return Number.POSITIVE_INFINITY
  return Math.max(0, Math.log2(FABRIC_REFERENCE_GBPS / perNodeGBps))
}

function clamp(value: number): number {
  if (!Number.isFinite(value)) return EFFICIENCY_FLOOR
  return Math.min(1, Math.max(EFFICIENCY_FLOOR, value))
}

/**
 * Inter-node efficiency for the PREFILL roofline
 *
 * Prefill ships full-sequence activations across a stage boundary
 * (batch x seqlen x hidden x 2 bytes), so it is genuinely bandwidth-bound.
 *
 *   eff = PP_BASE * (1 - 0.06*L - 0.020*L^2) * classFactor,  L = halvings
 *
 * The quadratic term encodes that degradation is superlinear: the first halving
 * of bandwidth costs little, the fourth is severe.
 *
 * DERIVED FROM BANDWIDTH, NOT MEASURED — as with INTERCONNECT_SPECS.
 */
export function fabricPrefillEfficiency(perNodeGBps: number, classFactor: number): number {
  const l = halvingsBelowReference(perNodeGBps)
  return clamp(PP_BASE_EFFICIENCY * (1 - 0.06 * l - 0.02 * l * l) * classFactor)
}

/**
 * Inter-node efficiency for the DECODE roofline
 *
 * Decode ships one token's activations per stage handoff — batch x hidden x 2
 * bytes, kilobytes. The hop is latency-bound at roughly 10 microseconds against
 * a 10-20 ms decode step, so it is near-free and this formula is near-flat.
 * Applying the prefill number here would badly understate decode throughput.
 *
 * DERIVED FROM BANDWIDTH, NOT MEASURED.
 */
export function fabricDecodeEfficiency(perNodeGBps: number): number {
  return clamp(0.99 - 0.01 * halvingsBelowReference(perNodeGBps))
}

/**
 * Pipeline fill/drain ("bubble") efficiency
 *
 *   eff = M / (M + numNodes - 1),  M = max(batchSize, numNodes)
 *
 * Independent of bandwidth: a pipeline of S stages idles S-1 slots at the start
 * and end of every batch. Omitting this flatters deep pipelines badly — 4 nodes
 * at batch 1 loses about 43% to bubbles alone.
 *
 * The M assumption — microbatch count tracks batch size, floored at the stage
 * count — is a heuristic. Real serving stacks tune it independently.
 */
export function pipelineBubbleEfficiency(batchSize: number, numNodes: number): number {
  if (numNodes <= 1) return 1
  const microbatches = Math.max(batchSize, numNodes)
  return microbatches / (microbatches + numNodes - 1)
}

/**
 * Resolve a fabric selection to a concrete spec
 *
 * Custom input with no value falls back to the 800GbE preset rather than
 * throwing: the UI can hold 'custom' selected while the field is still empty.
 */
export function resolveFabricSpec(
  type: FabricType,
  custom: CustomFabricInput | null,
): FabricSpec {
  if (type === 'custom') {
    if (!custom) return FABRIC_SPECS['ethernet-800g']
    return {
      type: 'custom',
      label: `${custom.name} (${custom.port_gbps} GB/s/port)`,
      portGBps: custom.port_gbps,
      classFactor: 1.0,
    }
  }
  return FABRIC_SPECS[type]
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `npx vitest run src/engines/fabric.test.ts`
Expected: PASS, all cases.

- [ ] **Step 7: Export from the barrel**

In `src/engines/index.ts`, insert after the `// DeepSpeed ZeRO engine` block to keep the alphabetical section order:

```ts
// Scale-out fabric
export {
  FABRIC_SPECS,
  fabricDecodeEfficiency,
  fabricPrefillEfficiency,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
  resolveFabricSpec,
} from './fabric'
```

and add `FabricSpec` and `FabricType` to the existing `export type { … } from './types'` block, keeping that list alphabetical.

- [ ] **Step 8: Lint and typecheck**

Run: `rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
rtk git add src/engines/fabric.ts src/engines/fabric.test.ts src/engines/types.ts src/engines/index.ts src/utils/schemas.ts
rtk git commit -m "$(cat <<'EOF'
feat(engines): add scale-out fabric model

Scale-out (node-to-node) is a separate tier from INTERCONNECT_SPECS:
different physics, different efficiency semantics, and the two coexist
in one cluster.

Presets for 1.6TbE/800GbE SONiC RoCEv2, InfiniBand XDR/NDR, 400GbE and
100GbE, plus a custom port-speed entry validated through Zod.

Per-node bandwidth is port speed times GPU count, not a single port:
the standard AI node has one NIC per GPU and collectives stripe across
all of them. Getting this wrong understates node bandwidth 8x.

Prefill and decode get separate efficiency formulas. Prefill ships
full-sequence activations and is bandwidth-bound; decode ships one
token's worth and is latency-bound and near-free. Pipeline bubble
efficiency is modelled separately from bandwidth.

All figures derived from bandwidth by documented formulas, not measured.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: The node dimension in `MultiGPUVRAMBreakdown`

Adds the fields without changing any behavior yet, so Task 6 and Task 7 have a stable shape to build against and the existing suite proves nothing moved.

**Files:**
- Modify: `src/engines/types.ts:173-203`
- Modify: `src/engines/multi-gpu.ts` (three return sites: ~105-123, ~174-192, ~240-257)
- Test: `src/engines/multi-gpu.test.ts`

**Interfaces:**
- Consumes: `FabricType` (Task 4).
- Produces — Tasks 6, 7 and 9 read these:
  - `MultiGPUVRAMBreakdown.numNodes: number`
  - `.gpusPerNode: number`
  - `.intraNodeEfficiency: number`
  - `.interNodeDecodeEfficiency: number`
  - `.interNodePrefillEfficiency: number`
  - `.bubbleEfficiency: number`
  - `.prefillScalingEfficiency: number`
  - `.scalingEfficiency` keeps its name and becomes the **decode-path** combined product.

- [ ] **Step 1: Write the failing test**

Append to `src/engines/multi-gpu.test.ts`:

```ts
describe('node dimension defaults', () => {
  it('reports a single node with all GPUs in it', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100SXM)
    expect(result.numNodes).toBe(1)
    expect(result.gpusPerNode).toBe(4)
  })

  it('leaves every inter-node term neutral', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100SXM)
    expect(result.interNodeDecodeEfficiency).toBe(1)
    expect(result.interNodePrefillEfficiency).toBe(1)
    expect(result.bubbleEfficiency).toBe(1)
  })

  it('makes prefill and decode efficiency identical within one node', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100SXM)
    expect(result.prefillScalingEfficiency).toBe(result.scalingEfficiency)
    expect(result.intraNodeEfficiency).toBe(result.scalingEfficiency)
  })

  it('holds for the single-GPU passthrough too', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 1, 'tensor-parallel', h100SXM)
    expect(result.numNodes).toBe(1)
    expect(result.gpusPerNode).toBe(1)
    expect(result.scalingEfficiency).toBe(1)
    expect(result.prefillScalingEfficiency).toBe(1)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/engines/multi-gpu.test.ts`
Expected: FAIL to compile — the properties do not exist on `MultiGPUVRAMBreakdown`.

- [ ] **Step 3: Add the fields to the type**

In `src/engines/types.ts`, inside `MultiGPUVRAMBreakdown` (line 173), replace the `scalingEfficiency` line (~201-202) with:

```ts
  /** Number of nodes (servers) in the configuration */
  numNodes: number
  /** GPUs in each node */
  gpusPerNode: number
  /** Intra-node scaling efficiency, from INTERCONNECT_SPECS */
  intraNodeEfficiency: number
  /** Inter-node efficiency on the decode path; 1.0 when numNodes === 1 */
  interNodeDecodeEfficiency: number
  /** Inter-node efficiency on the prefill path; 1.0 when numNodes === 1 */
  interNodePrefillEfficiency: number
  /** Pipeline fill/drain efficiency; 1.0 when numNodes === 1 */
  bubbleEfficiency: number
  /**
   * Combined efficiency for the DECODE roofline
   *
   * intraNodeEfficiency * interNodeDecodeEfficiency * bubbleEfficiency.
   * Keeps its original name so performance.ts's decode site is unchanged.
   */
  scalingEfficiency: number
  /**
   * Combined efficiency for the PREFILL roofline
   *
   * intraNodeEfficiency * interNodePrefillEfficiency * bubbleEfficiency.
   * Differs from scalingEfficiency only when numNodes > 1.
   */
  prefillScalingEfficiency: number
```

- [ ] **Step 4: Populate the fields at all three return sites**

In `src/engines/multi-gpu.ts`, each of the three `return { … }` objects gets the new fields. The pattern is identical in all three — only the efficiency value differs.

In `calculateTensorParallelVRAM` (return at line ~105), replace `scalingEfficiency,` with:

```ts
    numNodes: 1,
    gpusPerNode: numGPUs,
    intraNodeEfficiency: scalingEfficiency,
    interNodeDecodeEfficiency: 1,
    interNodePrefillEfficiency: 1,
    bubbleEfficiency: 1,
    scalingEfficiency,
    prefillScalingEfficiency: scalingEfficiency,
```

In `calculatePipelineParallelVRAM` (return at line ~174), replace the `scalingEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),` line with:

```ts
    numNodes: 1,
    gpusPerNode: numGPUs,
    // PP has lower communication overhead than TP; use flat 95% efficiency
    intraNodeEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),
    interNodeDecodeEfficiency: 1,
    interNodePrefillEfficiency: 1,
    bubbleEfficiency: 1,
    scalingEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),
    prefillScalingEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),
```

In the `numGPUs === 1` passthrough (return at line ~240), replace `scalingEfficiency: 1.0,` with:

```ts
      numNodes: 1,
      gpusPerNode: 1,
      intraNodeEfficiency: 1.0,
      interNodeDecodeEfficiency: 1,
      interNodePrefillEfficiency: 1,
      bubbleEfficiency: 1,
      scalingEfficiency: 1.0,
      prefillScalingEfficiency: 1.0,
```

- [ ] **Step 5: Run the full engine suite**

Run: `npx vitest run src/engines/`
Expected: PASS. Nothing behavioral changed — this step is purely additive, and the existing assertions on `scalingEfficiency` prove it.

- [ ] **Step 6: Lint and typecheck**

Run: `rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: clean. The worker serializes this object across `postMessage`; all new fields are plain numbers, so structured cloning is unaffected.

- [ ] **Step 7: Commit**

```bash
rtk git add src/engines/types.ts src/engines/multi-gpu.ts src/engines/multi-gpu.test.ts
rtk git commit -m "$(cat <<'EOF'
feat(engines): add the node dimension to MultiGPUVRAMBreakdown

Purely additive. Every single-node path reports numNodes 1, neutral
inter-node terms, and prefillScalingEfficiency equal to scalingEfficiency,
so behavior is unchanged and the existing assertions prove it.

scalingEfficiency keeps its name and becomes the decode-path combined
product, so performance.ts's decode site needs no signature change.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `calculateMultiNodeVRAM`

**Files:**
- Create: `src/engines/multi-node.ts`
- Create: `src/engines/multi-node.test.ts`
- Modify: `src/engines/index.ts`

**Interfaces:**
- Consumes: `calculateMultiGPUVRAM` (Task 5's shape), `fabric.ts` (Task 4), `MultiGPUVRAMBreakdown` node fields (Task 5).
- Produces — Tasks 8 and 10 call exactly this:

```ts
export function calculateMultiNodeVRAM(params: {
  singleGPU: InferenceVRAMBreakdown
  model: Model
  gpuVramGB: number
  gpusPerNode: number
  numNodes: number
  intraNodeStrategy: ShardingStrategy
  gpu: GPU
  fabric: FabricSpec
  batchSize: number
}): MultiGPUVRAMBreakdown
```

- [ ] **Step 1: Write the failing tests**

Create `src/engines/multi-node.test.ts`:

```ts
import { FABRIC_SPECS } from '@engines/fabric'
import { calculateInferenceVRAM } from '@engines/inference'
import { calculateMultiGPUVRAM } from '@engines/multi-gpu'
import { calculateMultiNodeVRAM } from '@engines/multi-node'
import type { GPU, Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'

const llama70b: Model = {
  id: 'meta-llama-3-70b',
  name: 'Llama 3 70B',
  num_parameters_billion: 70,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,
  intermediate_size: 28672,
  vocab_size: 128256,
  max_position_embeddings: 8192,
  architecture: 'dense',
}

const mi355x: GPU = {
  id: 'amd-mi355x',
  name: 'AMD Instinct MI355X',
  manufacturer: 'amd',
  vram_gb: 288,
  memory_bandwidth_gbps: 8000,
  memory_type: 'HBM3E',
  bus_width: 8192,
  fp16_tflops: 2516,
  fp32_tflops: 157,
  tdp_watts: 1400,
  interconnect: 'infinity-fabric',
  tier: 'datacenter',
}

const singleGPU = calculateInferenceVRAM({
  model: llama70b,
  quantization: 'fp16',
  sequenceLength: 4096,
  batchSize: 1,
})

const base = {
  singleGPU,
  model: llama70b,
  gpuVramGB: 288,
  intraNodeStrategy: 'tensor-parallel' as const,
  gpu: mi355x,
  fabric: FABRIC_SPECS['ethernet-800g'],
  batchSize: 1,
}

describe('calculateMultiNodeVRAM', () => {
  it('is an exact passthrough at one node — the regression guard', () => {
    const multiNode = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 1 })
    const multiGPU = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      288,
      8,
      'tensor-parallel',
      mi355x,
    )
    expect(multiNode.totalPerGPU.toString()).toBe(multiGPU.totalPerGPU.toString())
    expect(multiNode.scalingEfficiency).toBe(multiGPU.scalingEfficiency)
    expect(multiNode.prefillScalingEfficiency).toBe(multiGPU.prefillScalingEfficiency)
  })

  it('reports the topology', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.numNodes).toBe(4)
    expect(result.gpusPerNode).toBe(8)
    expect(result.numGPUs).toBe(32)
  })

  it('divides weights by the total GPU count across both levels', () => {
    const oneNode = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 1 })
    const fourNodes = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(fourNodes.perGPU.modelWeights.lessThan(oneNode.perGPU.modelWeights)).toBe(true)
  })

  it('shards the KV cache across nodes as well as within them', () => {
    const oneNode = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 1 })
    const fourNodes = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    // Four nodes split the layers four ways before the intra-node TP split.
    expect(fourNodes.perGPU.kvCache.toNumber()).toBeCloseTo(
      oneNode.perGPU.kvCache.toNumber() / 4,
      6,
    )
  })

  it('separates decode from prefill efficiency once nodes > 1', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.interNodeDecodeEfficiency).toBeGreaterThan(result.interNodePrefillEfficiency)
    expect(result.scalingEfficiency).toBeGreaterThan(result.prefillScalingEfficiency)
  })

  it('charges a pipeline bubble at batch 1 across nodes', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.bubbleEfficiency).toBeCloseTo(0.571, 3)
  })

  it('rewards a faster fabric', () => {
    const slow = calculateMultiNodeVRAM({
      ...base,
      gpusPerNode: 8,
      numNodes: 4,
      fabric: FABRIC_SPECS['ethernet-100g'],
    })
    const fast = calculateMultiNodeVRAM({
      ...base,
      gpusPerNode: 8,
      numNodes: 4,
      fabric: FABRIC_SPECS['ethernet-1600g'],
    })
    expect(fast.prefillScalingEfficiency).toBeGreaterThan(slow.prefillScalingEfficiency)
  })

  it('rejects a node count below 1', () => {
    expect(() => calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 0 })).toThrow(
      /numNodes/,
    )
  })

  it('still rejects more than 8 GPUs in one node', () => {
    expect(() => calculateMultiNodeVRAM({ ...base, gpusPerNode: 9, numNodes: 2 })).toThrow(
      /numGPUs must be between 1 and 8/,
    )
  })
})
```

If the `Model` or `GPU` fixture shape above does not match the current Zod schema, copy the fixture shape used in `src/engines/multi-gpu.test.ts` instead — that file is already kept in sync with the schema.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/multi-node.test.ts`
Expected: FAIL — `Cannot find module '@engines/multi-node'`.

- [ ] **Step 3: Implement the module**

Create `src/engines/multi-node.ts`:

```ts
import type { GPU, Model } from '@utils/schemas'
import { PP_ACTIVATION_STASHING_OVERHEAD } from './constants'
import {
  fabricDecodeEfficiency,
  fabricPrefillEfficiency,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
} from './fabric'
import { calculateMultiGPUVRAM } from './multi-gpu'
import type {
  FabricSpec,
  InferenceVRAMBreakdown,
  MultiGPUVRAMBreakdown,
  ShardingStrategy,
} from './types'

/**
 * Calculate VRAM for inference spread across multiple identical servers
 *
 * Two-level topology: tensor parallelism inside a node, pipeline parallelism
 * across nodes. Tensor parallelism across a node boundary is deliberately not
 * offered — its per-layer allreduce over a fabric an order of magnitude slower
 * than NVLink or Infinity Fabric is not a configuration worth presenting as
 * viable.
 *
 * Composes over calculateMultiGPUVRAM rather than replacing it: this function
 * derives a per-stage breakdown, then hands the intra-node split to the
 * existing code. That keeps calculateMultiGPUVRAM's 1-8 GPU guard correct —
 * under composition it is exactly the per-node bound.
 *
 * @throws Error if numNodes < 1, or if gpusPerNode is outside 1-8 (raised by
 *         calculateMultiGPUVRAM)
 *
 * @example
 * ```ts
 * calculateMultiNodeVRAM({
 *   singleGPU, model: llama405b, gpuVramGB: 288,
 *   gpusPerNode: 8, numNodes: 4,
 *   intraNodeStrategy: 'tensor-parallel', gpu: mi355x,
 *   fabric: FABRIC_SPECS['ethernet-800g'], batchSize: 1,
 * })
 * ```
 */
export function calculateMultiNodeVRAM(params: {
  singleGPU: InferenceVRAMBreakdown
  model: Model
  gpuVramGB: number
  gpusPerNode: number
  numNodes: number
  intraNodeStrategy: ShardingStrategy
  gpu: GPU
  fabric: FabricSpec
  batchSize: number
}): MultiGPUVRAMBreakdown {
  const {
    singleGPU,
    model,
    gpuVramGB,
    gpusPerNode,
    numNodes,
    intraNodeStrategy,
    gpu,
    fabric,
    batchSize,
  } = params

  if (!Number.isInteger(numNodes) || numNodes < 1) {
    throw new Error(`numNodes must be an integer >= 1, got ${numNodes}`)
  }

  // Single node: delegate unchanged. This is the regression guard for the whole
  // feature — one node must produce byte-identical results to before it existed.
  if (numNodes === 1) {
    return calculateMultiGPUVRAM(
      singleGPU,
      model,
      gpuVramGB,
      gpusPerNode,
      intraNodeStrategy,
      gpu,
    )
  }

  // Pipeline parallelism across nodes: each node owns a contiguous slice of
  // layers, so weights, KV cache and activations all divide by the node count.
  // Framework overhead does not — it is per-process (PyTorch + CUDA/ROCm
  // context) and every rank pays it in full.
  const stageBreakdown: InferenceVRAMBreakdown = {
    ...singleGPU,
    modelWeights: singleGPU.modelWeights.div(numNodes),
    kvCache: singleGPU.kvCache.div(numNodes),
    activations: singleGPU.activations
      .div(numNodes)
      .mul(new Decimal(1).add(PP_ACTIVATION_STASHING_OVERHEAD)),
  }

  const inner = calculateMultiGPUVRAM(
    stageBreakdown,
    model,
    gpuVramGB,
    gpusPerNode,
    intraNodeStrategy,
    gpu,
  )

  const perNodeGBps = perNodeFabricGBps(fabric.portGBps, gpusPerNode)
  const interNodePrefillEfficiency = fabricPrefillEfficiency(perNodeGBps, fabric.classFactor)
  const interNodeDecodeEfficiency = fabricDecodeEfficiency(perNodeGBps)
  const bubbleEfficiency = pipelineBubbleEfficiency(batchSize, numNodes)
  const intraNodeEfficiency = inner.intraNodeEfficiency

  return {
    ...inner,
    numGPUs: gpusPerNode * numNodes,
    numNodes,
    gpusPerNode,
    intraNodeEfficiency,
    interNodeDecodeEfficiency,
    interNodePrefillEfficiency,
    bubbleEfficiency,
    scalingEfficiency: intraNodeEfficiency * interNodeDecodeEfficiency * bubbleEfficiency,
    prefillScalingEfficiency:
      intraNodeEfficiency * interNodePrefillEfficiency * bubbleEfficiency,
    singleGPUBaseline: singleGPU.total,
  }
}
```

Add `import Decimal from 'decimal.js'` at the top — the activation term constructs one.

Note `totalPerGPU` and `utilizationPercent` come through from `inner` unchanged: they were computed from the already-divided stage breakdown, so they are correct per-GPU figures for the composed topology.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/engines/multi-node.test.ts`
Expected: PASS, all ten cases.

- [ ] **Step 5: Export from the barrel**

In `src/engines/index.ts`, add after the `// KV cache engine` block:

```ts
// Multi-node engine
export { calculateMultiNodeVRAM } from './multi-node'
```

- [ ] **Step 6: Run the full suite, lint, typecheck**

Run: `npx vitest run && rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
rtk git add src/engines/multi-node.ts src/engines/multi-node.test.ts src/engines/index.ts
rtk git commit -m "$(cat <<'EOF'
feat(engines): add calculateMultiNodeVRAM

Tensor parallel inside a node, pipeline parallel across nodes. TP across
a node boundary is not offered: its per-layer allreduce over a fabric an
order of magnitude slower than NVLink or Infinity Fabric is not a viable
configuration to present.

Composes over calculateMultiGPUVRAM instead of replacing it. That keeps
the existing 1-8 GPU guard correct as a per-node bound, and leaves the
existing test suite untouched. One node delegates straight through, which
is the regression guard for the whole feature.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Prefill uses the prefill efficiency

**Files:**
- Modify: `src/engines/performance.ts:152-160`
- Modify: `src/engines/performance.test.ts`

**Interfaces:**
- Consumes: `MultiGPUVRAMBreakdown.prefillScalingEfficiency` (Task 5).
- Produces: no new exports. `estimatePerformance` keeps its signature.

- [ ] **Step 1: Write the failing test**

Append to `src/engines/performance.test.ts`:

```ts
describe('multi-node roofline separation', () => {
  it('scales prefill by the prefill efficiency, not the decode one', () => {
    const multiGPUResult = {
      ...baseMultiGPUResult,
      numGPUs: 32,
      numNodes: 4,
      gpusPerNode: 8,
      scalingEfficiency: 0.9,
      prefillScalingEfficiency: 0.5,
    }
    const fast = estimatePerformance({
      model: llama70b,
      gpu: h100SXM,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      multiGPUResult: { ...multiGPUResult, prefillScalingEfficiency: 0.9 },
    })
    const slow = estimatePerformance({
      model: llama70b,
      gpu: h100SXM,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      multiGPUResult,
    })
    // Decode is identical: both carry scalingEfficiency 0.9.
    expect(slow.tokensPerSecond.toString()).toBe(fast.tokensPerSecond.toString())
    // Prefill is not: the slow fabric doubles time-to-first-token's compute term.
    expect(slow.timeToFirstToken.greaterThan(fast.timeToFirstToken)).toBe(true)
  })
})
```

`baseMultiGPUResult` is a fixture you add near the top of the describe block, built by calling `calculateMultiGPUVRAM` with the file's existing fixtures — do not hand-write a `MultiGPUVRAMBreakdown` literal, or it will drift from the type.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/engines/performance.test.ts`
Expected: FAIL — `timeToFirstToken` is identical in both cases, because prefill currently reads `scalingEfficiency`.

- [ ] **Step 3: Apply the fix**

In `src/engines/performance.ts`, in the prefill block (line ~155), replace:

```ts
    if (multiGPUResult && multiGPUResult.numGPUs > 1) {
      effectiveFLOPS = effectiveFLOPS
        .mul(multiGPUResult.numGPUs)
        .mul(multiGPUResult.prefillScalingEfficiency)
    }
```

Leave the decode site (line ~104) reading `scalingEfficiency` — that is now explicitly the decode-path product, and the comment above it should say so:

```ts
  // 4b. Apply multi-GPU scaling: effective TPS = single-GPU TPS × numGPUs ×
  //     scalingEfficiency. scalingEfficiency is the DECODE-path product; the
  //     prefill roofline below uses prefillScalingEfficiency instead, because a
  //     decode hop across nodes is latency-bound and near-free while prefill is
  //     bandwidth-bound.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/engines/performance.test.ts`
Expected: PASS. Existing single-node tests are unaffected — `prefillScalingEfficiency` equals `scalingEfficiency` whenever `numNodes === 1`.

- [ ] **Step 5: Commit**

```bash
rtk git add src/engines/performance.ts src/engines/performance.test.ts
rtk git commit -m "$(cat <<'EOF'
fix(engines): scale prefill by the prefill efficiency

A single scalingEfficiency was applied to both the decode and prefill
rooflines. Across a node boundary these diverge sharply: decode ships one
token's activations (kilobytes, latency-bound, ~10us against a 10-20ms
step) while prefill ships full-sequence activations and is genuinely
bandwidth-bound. One number penalized decode and flattered prefill.

Single-node results are unchanged: prefillScalingEfficiency equals
scalingEfficiency whenever numNodes is 1.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Store, URL and comparison state

**Files:**
- Modify: `src/store/uiStore.ts:47-50`, `:77`, `:128-131`, `:163`
- Modify: `src/store/urlSerializer.ts:44-50` (schema), `:98-145` (serialize params + body)
- Modify: `src/hooks/useURLSync.ts:110`
- Modify: `src/store/comparisonStore.ts:19`
- Modify: `src/store/urlSerializer.test.ts`

**Interfaces:**
- Consumes: `FabricType` (Task 4), `CustomFabricInput` (Task 4).
- Produces — Tasks 9 and 10 read these from the store:
  - `numGPUs: number` — **now means GPUs per node**
  - `numNodes: number` (default 1)
  - `interNodeFabric: FabricType` (default `'ethernet-800g'`)
  - `customFabric: CustomFabricInput | null` (default `null`)
  - actions `setNumNodes`, `setInterNodeFabric`, `setCustomFabric`
  - URL keys `nn`, `fab`, `fabc`, all optional

- [ ] **Step 1: Write the failing test**

Append to `src/store/urlSerializer.test.ts`:

```ts
describe('multi-node URL state', () => {
  it('round-trips the node topology', () => {
    const hash = serializeToURL({ ...baseState, numGPUs: 8, numNodes: 4, interNodeFabric: 'ethernet-1600g' })
    const decoded = deserializeFromURL(hash)
    expect(decoded?.ng).toBe(8)
    expect(decoded?.nn).toBe(4)
    expect(decoded?.fab).toBe('ethernet-1600g')
  })

  it('omits the node keys at a single node, keeping shared links short', () => {
    const hash = serializeToURL({ ...baseState, numGPUs: 4, numNodes: 1 })
    const decoded = deserializeFromURL(hash)
    expect(decoded?.nn).toBeUndefined()
    expect(decoded?.fab).toBeUndefined()
  })

  it('accepts a pre-feature URL, where ng meant total GPUs', () => {
    // 1 node x 4 GPUs is arithmetically the same configuration as the old
    // "4 GPUs", so old links keep working and keep meaning the same thing.
    const legacy = serializeToURL({ ...baseState, numGPUs: 4, numNodes: 1 })
    const decoded = deserializeFromURL(legacy)
    expect(decoded?.ng).toBe(4)
    expect(decoded?.nn ?? 1).toBe(1)
  })

  it('round-trips a custom fabric', () => {
    const hash = serializeToURL({
      ...baseState,
      numGPUs: 8,
      numNodes: 2,
      interNodeFabric: 'custom',
      customFabric: { name: 'Lab', port_gbps: 25 },
    })
    const decoded = deserializeFromURL(hash)
    expect(decoded?.fab).toBe('custom')
    expect(decoded?.fabc).toEqual({ name: 'Lab', port_gbps: 25 })
  })
})
```

`baseState` is the existing fixture in that file (it appears around line 34 with `numGPUs: 2`); extend it with `numNodes: 1`, `interNodeFabric: 'ethernet-800g'` and `customFabric: null` so every existing case still typechecks.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/store/urlSerializer.test.ts`
Expected: FAIL to compile — `numNodes` is not in `serializeToURL`'s parameter type.

- [ ] **Step 3: Extend the URL schema**

In `src/store/urlSerializer.ts`, in `URLStateSchema` after the `ss: z.string(),` line (~50):

```ts
  // Multi-node (absent = single node, for backward compatibility with links
  // created before this feature, where ng meant the total GPU count)
  nn: z.number().int().min(1).max(64).optional(), // numNodes
  fab: z
    .enum([
      'ethernet-1600g',
      'ethernet-800g',
      'infiniband-xdr',
      'infiniband-ndr',
      'ethernet-400g',
      'ethernet-100g',
      'custom',
    ])
    .optional(), // interNodeFabric
  fabc: z
    .object({
      name: z.string(),
      port_gbps: z.number(),
    })
    .optional(), // customFabric
```

Also update the `ng` comment on line 49 to `// numGPUs — PER NODE since v1.10`.

- [ ] **Step 4: Extend `serializeToURL`**

In the parameter type (after `numGPUs: number`):

```ts
  numNodes: number
  interNodeFabric: FabricType
  customFabric: CustomFabricInput | null
```

and in the returned `urlState` object, after `ss: state.shardingStrategy,`:

```ts
    // Multi-node (only when actually multi-node, to keep single-node links short)
    ...(state.numNodes > 1
      ? {
          nn: state.numNodes,
          fab: state.interNodeFabric,
          ...(state.interNodeFabric === 'custom' && state.customFabric
            ? { fabc: state.customFabric }
            : {}),
        }
      : {}),
```

Add `FabricType` to the existing `import type { … } from '@engines/types'` block and `CustomFabricInput` to the `from '@utils/schemas'` import.

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run src/store/urlSerializer.test.ts`
Expected: PASS.

- [ ] **Step 6: Extend the store**

In `src/store/uiStore.ts`, in the state interface replace the multi-GPU block (lines ~47-50):

```ts
  // Multi-GPU parameters (persisted)
  /** GPUs per node. Was the total GPU count before multi-node; 1 node x N GPUs
   *  is the same configuration, so persisted state needs no migration. */
  numGPUs: number
  shardingStrategy: ShardingStrategy

  // Multi-node parameters (persisted)
  numNodes: number
  interNodeFabric: FabricType
  customFabric: CustomFabricInput | null
```

In the actions interface, after `setNumGPUs`:

```ts
  setNumNodes: (numNodes: number) => void
  setInterNodeFabric: (fabric: FabricType) => void
  setCustomFabric: (fabric: CustomFabricInput | null) => void
```

In the defaults, after `numGPUs: 1,`:

```ts
      numNodes: 1,
      interNodeFabric: 'ethernet-800g' as FabricType,
      customFabric: null,
```

In the actions, after `setNumGPUs`:

```ts
      setNumNodes: (numNodes) => set({ numNodes }),
      setInterNodeFabric: (interNodeFabric) => set({ interNodeFabric }),
      setCustomFabric: (customFabric) => set({ customFabric }),
```

No persist `migrate` function is needed. Zustand's persist merges stored state over the defaults, so a pre-feature payload with no `numNodes` picks up the default of 1 — which is the same configuration it always described. Confirm the store's `persist` options do not set a `version` that would discard unknown-version payloads; if they do, bump it and add a `migrate` that returns the persisted state unchanged.

- [ ] **Step 7: Restore the new keys from a shared link**

In `src/hooks/useURLSync.ts`, after line 111 (`store.setShardingStrategy(...)`):

```ts
    // Multi-node (absent = single node, for links created before the feature)
    store.setNumNodes(urlState.nn ?? 1)
    if (urlState.fab) store.setInterNodeFabric(urlState.fab)
    if (urlState.fabc) store.setCustomFabric(urlState.fabc)
```

- [ ] **Step 8: Carry topology into saved comparisons**

In `src/store/comparisonStore.ts`, in the snapshot type at line 19, after `numGPUs: number`:

```ts
    numNodes: number
    interNodeFabric: FabricType
```

Add the `FabricType` import, and update every site that builds a snapshot to populate both fields. Run `rtk proxy npm run typecheck` to find them — TypeScript names each one. Without this, two saved comparisons with different topologies would compare as identical.

- [ ] **Step 9: Run everything**

Run: `npx vitest run && rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: clean. If `comparisonStore.test.ts` fails on the new required fields, add them to its fixture.

- [ ] **Step 10: Commit**

```bash
rtk git add src/store/uiStore.ts src/store/urlSerializer.ts src/store/urlSerializer.test.ts src/store/comparisonStore.ts src/hooks/useURLSync.ts
rtk git commit -m "$(cat <<'EOF'
feat(store): add node topology to state, URL and comparisons

numGPUs is reinterpreted as GPUs per node. This is safe because 1 node x
N GPUs is arithmetically the configuration N GPUs always described, so
persisted state and pre-feature shared links keep working and keep
meaning the same thing. The nn/fab/fabc URL keys are all optional and are
omitted entirely at a single node, so single-node links stay short.

Comparison snapshots carry the topology, or two comparisons with
different node counts would compare as identical.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: UI controls

**Files:**
- Modify: `src/components/inputs/GPUCountSelector.tsx`
- Create: `src/components/inputs/NodeCountSelector.tsx`
- Create: `src/components/inputs/InterNodeFabricSelector.tsx`
- Modify: `src/components/inputs/ShardingStrategySelector.tsx:15-45`, `:125-135`

**Interfaces:**
- Consumes: store fields and actions from Task 8; `FABRIC_SPECS`, `perNodeFabricGBps` from Task 4.
- Produces: `<NodeCountSelector />` and `<InterNodeFabricSelector />`, both self-mounting from the store and rendering nothing when `numNodes <= 1` (fabric selector only). They must be added to whichever panel already renders `<GPUCountSelector />` — find it with `grep -rn "GPUCountSelector" src/components`.

- [ ] **Step 1: Relabel the GPU count slider**

In `src/components/inputs/GPUCountSelector.tsx`, update the doc comment, the label text, the tooltip and the footnote:

```tsx
/**
 * GPU count selector with range slider (1-8 GPUs per server)
 *
 * This is the PER-NODE count. Total GPUs is this times the server count from
 * NodeCountSelector. The 1-8 bound is real: both NVLink and Infinity Fabric
 * top out at an 8-GPU fully connected domain.
 */
```

Label: `GPUs per server`. Tooltip text:

```tsx
<InfoTip text="GPUs inside one server. Tensor parallelism runs at this level, over NVLink or Infinity Fabric. Capped at 8 — that is the size of a fully connected GPU domain in current hardware." />
```

Footnote — replace the `numGPUs > 1 &&` block:

```tsx
      {numGPUs > 1 && (
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {numGPUs} GPUs per server, tensor parallel
        </p>
      )}
```

- [ ] **Step 2: Create the server-count selector**

Create `src/components/inputs/NodeCountSelector.tsx`:

```tsx
import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

/**
 * Server count selector with range slider (1-8 servers)
 *
 * Pipeline parallelism runs across servers. Tensor parallelism across a server
 * boundary is not offered: its per-layer allreduce over a network fabric an
 * order of magnitude slower than NVLink is not a viable configuration.
 */
export function NodeCountSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const numNodes = useUIStore((s) => s.numNodes)
  const setNumNodes = useUIStore((s) => s.setNumNodes)

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="node-count"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Number of servers
        </label>
        <InfoTip text="Identical servers in the cluster. Model layers are split across them with pipeline parallelism, so more servers fit larger models but add a network hop between layer groups." />
      </div>
      <div className="flex items-center gap-4">
        <input
          id="node-count"
          type="range"
          min={1}
          max={8}
          step={1}
          value={numNodes}
          onChange={(e) => setNumNodes(Number(e.target.value))}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <span className="text-lg font-semibold text-gray-900 dark:text-white w-8 text-center tabular-nums">
          {numNodes}
        </span>
      </div>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        {numNodes * numGPUs} GPUs total
        {numNodes > 1 ? ' · pipeline parallel across servers' : ''}
      </p>
    </div>
  )
}
```

- [ ] **Step 3: Create the fabric selector**

Create `src/components/inputs/InterNodeFabricSelector.tsx`:

```tsx
import { InfoTip } from '@components/common/InfoTip'
import { FABRIC_SPECS, perNodeFabricGBps } from '@engines/fabric'
import type { FabricType } from '@engines/types'
import { useUIStore } from '@store/uiStore'

/**
 * Scale-out fabric selector, shown only when the cluster spans servers
 *
 * Labels show per-node aggregate bandwidth, not port speed. The standard AI
 * node has one NIC per GPU and collectives stripe across all of them, so an
 * 8-GPU node on 800GbE has 800 GB/s of scale-out, not 100. Showing only the
 * port speed would understate node bandwidth eightfold.
 */
export function InterNodeFabricSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const numNodes = useUIStore((s) => s.numNodes)
  const interNodeFabric = useUIStore((s) => s.interNodeFabric)
  const setInterNodeFabric = useUIStore((s) => s.setInterNodeFabric)
  const customFabric = useUIStore((s) => s.customFabric)
  const setCustomFabric = useUIStore((s) => s.setCustomFabric)

  if (numNodes <= 1) {
    return null
  }

  const options = Object.values(FABRIC_SPECS)

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="inter-node-fabric"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Network between servers
        </label>
        <InfoTip text="The scale-out fabric carrying activations between servers. Assumes one NIC per GPU, the standard build, so per-server bandwidth is the port speed times the GPU count." />
      </div>
      <select
        id="inter-node-fabric"
        value={interNodeFabric}
        onChange={(e) => setInterNodeFabric(e.target.value as FabricType)}
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white"
      >
        {options.map((spec) => (
          <option key={spec.type} value={spec.type}>
            {spec.label} — {perNodeFabricGBps(spec.portGBps, numGPUs)} GB/s per server
          </option>
        ))}
        <option value="custom">Custom…</option>
      </select>

      {interNodeFabric === 'custom' && (
        <div className="mt-2">
          <label
            htmlFor="custom-fabric-gbps"
            className="block text-xs text-gray-600 dark:text-gray-400 mb-1"
          >
            Port bandwidth, GB/s (unidirectional, per NIC)
          </label>
          <input
            id="custom-fabric-gbps"
            type="number"
            min={0.1}
            max={10000}
            step={0.1}
            value={customFabric?.port_gbps ?? ''}
            onChange={(e) => {
              const value = Number(e.target.value)
              setCustomFabric(
                Number.isFinite(value) && value > 0
                  ? { name: 'Custom fabric', port_gbps: value }
                  : null,
              )
            }}
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white"
          />
          {customFabric && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {perNodeFabricGBps(customFabric.port_gbps, numGPUs)} GB/s per server across{' '}
              {numGPUs} NICs
            </p>
          )}
        </div>
      )}

      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        Efficiency figures are derived from bandwidth, not measured benchmarks.
      </p>
    </div>
  )
}
```

- [ ] **Step 4: Make the sharding selector read per-node**

In `src/components/inputs/ShardingStrategySelector.tsx`:

- Update the doc comment at line 9 from `Only visible when numGPUs > 1` to `Only visible when there is more than one GPU per server. This selects the INTRA-node strategy; across servers the strategy is always pipeline parallel.`
- The `numGPUs` the component already reads (line 15) is now per-node, so lines 21 and 41 are correct as written — no change needed to the logic.
- Update the heading/label copy to say **Intra-server sharding strategy**.
- Update the warning at line 130 to name the scope:

```tsx
            ⚠ Tensor Parallel with {numGPUs} GPUs per server may experience performance
            degradation on{' '}
```

- [ ] **Step 5: Mount the new controls**

Run: `grep -rn "GPUCountSelector" src/components src/App.tsx`

In the panel that renders `<GPUCountSelector />`, add `<NodeCountSelector />` immediately after it and `<InterNodeFabricSelector />` after that, keeping the existing spacing wrapper pattern of the surrounding controls.

- [ ] **Step 6: Verify in the running app**

Run: `npm run dev`

Check, in the browser:
1. Default state shows "GPUs per server 1", "Number of servers 1", "1 GPUs total", and **no** fabric selector.
2. Raising servers to 4 reveals the fabric dropdown, whose options read e.g. "800GbE (SONiC / RoCEv2) — 800 GB/s per server" when GPUs per server is 8.
3. Selecting "Custom…" reveals the number field; typing 25 shows "200 GB/s per server across 8 NICs".
4. The VRAM-per-GPU figure drops when server count rises.
5. An 8-GPU MI355X server on tensor parallel shows no interconnect warning.

- [ ] **Step 7: Lint, typecheck, build**

Run: `rtk proxy npm run lint && rtk proxy npm run typecheck && npm run build`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
rtk git add src/components/inputs/
rtk git commit -m "$(cat <<'EOF'
feat(ui): add server count and scale-out fabric controls

GPU count is relabelled "GPUs per server" and the sharding selector is
scoped to intra-server, since across servers the strategy is always
pipeline parallel.

Fabric dropdown labels show per-server aggregate bandwidth rather than
port speed: one NIC per GPU is the standard build and collectives stripe
across all of them, so showing 100 GB/s for an 8-GPU node on 800GbE would
understate it eightfold.

The panel states that efficiency figures are derived, not benchmarked.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Wire the calculation paths

**Files:**
- Modify: `src/hooks/useInferenceCalculation.ts:225-360`
- Modify: `src/workers/calculation.worker.ts:15`, `:186-205`
- Test: `src/engines/inference.integration.test.ts`

**Interfaces:**
- Consumes: `calculateMultiNodeVRAM` (Task 6), `resolveFabricSpec` (Task 4), store fields (Task 8).
- Produces: end-to-end multi-node results reaching the UI.

- [ ] **Step 1: Write the failing integration test**

Append to `src/engines/inference.integration.test.ts`:

```ts
describe('multi-node end to end', () => {
  it('fits a 405B model on 2 servers of 8 MI355X that will not fit on one', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama405b,
      quantization: 'fp16',
      sequenceLength: 8192,
      batchSize: 1,
    })
    const oneNode = calculateMultiNodeVRAM({
      singleGPU,
      model: llama405b,
      gpuVramGB: 288,
      gpusPerNode: 8,
      numNodes: 1,
      intraNodeStrategy: 'tensor-parallel',
      gpu: mi355x,
      fabric: FABRIC_SPECS['ethernet-800g'],
      batchSize: 1,
    })
    const twoNodes = calculateMultiNodeVRAM({
      singleGPU,
      model: llama405b,
      gpuVramGB: 288,
      gpusPerNode: 8,
      numNodes: 2,
      intraNodeStrategy: 'tensor-parallel',
      gpu: mi355x,
      fabric: FABRIC_SPECS['ethernet-800g'],
      batchSize: 1,
    })
    expect(twoNodes.totalPerGPU.lessThan(oneNode.totalPerGPU)).toBe(true)
    expect(twoNodes.utilizationPercent.lessThan(oneNode.utilizationPercent)).toBe(true)
  })
})
```

Use whatever `llama405b` and `mi355x` fixtures that file already defines; if it has none, build them from `src/data/models.json` and `src/data/gpus.json` via the same lookup helper the file already uses.

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run src/engines/inference.integration.test.ts`
Expected: FAIL — import error or missing fixture.

- [ ] **Step 3: Wire the hook**

In `src/hooks/useInferenceCalculation.ts`, add the store reads near line 230 alongside the existing `interconnectOverride` read:

```ts
  const numNodes = useUIStore((s) => s.numNodes)
  const interNodeFabric = useUIStore((s) => s.interNodeFabric)
  const customFabric = useUIStore((s) => s.customFabric)
```

Replace the multi-GPU block (lines ~339-359):

```ts
        let multiGPU = null
        let interconnectWarning = null
        const gpusPerNode = numGPUs ?? 1
        const effectiveNumNodes = numNodes ?? 1
        const effectiveStrategy = shardingStrategy ?? 'tensor-parallel'
        if (gpusPerNode > 1 || effectiveNumNodes > 1) {
          const baseBreakdown = offloading ? offloading.onDevice : vram
          multiGPU = multiNodeModule.calculateMultiNodeVRAM({
            singleGPU: baseBreakdown,
            model,
            gpuVramGB: effectiveGPU.vram_gb,
            gpusPerNode,
            numNodes: effectiveNumNodes,
            intraNodeStrategy: effectiveStrategy,
            gpu: effectiveGPU,
            fabric: fabricModule.resolveFabricSpec(interNodeFabric, customFabric),
            batchSize,
          })
          // Validation is per-node: the interconnect bounds apply inside a
          // server, not across the cluster.
          const validation = multiGPUModule.validateInterconnect(
            effectiveGPU,
            gpusPerNode,
            effectiveStrategy,
          )
          interconnectWarning = validation.warning
        }
```

Add `multiNodeModule` and `fabricModule` to the existing dynamic-import block that already produces `multiGPUModule` — match its exact `import()` style rather than introducing a static import, since that block exists to keep the engines out of the initial bundle.

Add `numNodes`, `interNodeFabric` and `customFabric` to the effect's dependency array (near line 380, where `interconnectOverride` already appears).

- [ ] **Step 4: Wire the worker**

In `src/workers/calculation.worker.ts`, extend the import at line 15 and add the fabric import:

```ts
import { validateInterconnect } from '../engines/multi-gpu'
import { calculateMultiNodeVRAM } from '../engines/multi-node'
import { resolveFabricSpec } from '../engines/fabric'
```

Replace the multi-GPU block (lines ~186-204):

```ts
      let multiGPUResult = null
      let interconnectWarning = null

      const gpusPerNode = numGPUs
      const nodes = numNodes ?? 1

      if (gpusPerNode > 1 || nodes > 1) {
        const baseBreakdown = offloadingResult ? offloadingResult.onDevice : vramBreakdown

        multiGPUResult = calculateMultiNodeVRAM({
          singleGPU: baseBreakdown,
          model,
          gpuVramGB: gpu.vram_gb,
          gpusPerNode,
          numNodes: nodes,
          intraNodeStrategy: shardingStrategy,
          gpu,
          fabric: resolveFabricSpec(interNodeFabric ?? 'ethernet-800g', customFabric ?? null),
          batchSize,
        })

        const validation = validateInterconnect(gpu, gpusPerNode, shardingStrategy)
        interconnectWarning = validation.warning
      }
```

Add `numNodes`, `interNodeFabric` and `customFabric` to the worker's inbound message type and to whatever posts to it. Run `rtk proxy npm run typecheck` to find the sender. If `calculateMultiGPUVRAM` is now unused in the worker, drop it from the import — unused imports are a Biome error.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run`
Expected: PASS, whole suite.

- [ ] **Step 6: Verify in the running app**

Run: `npm run dev`

Select Llama 3.1 405B, FP16, 8192 context, MI355X, 8 GPUs per server. Confirm it does not fit on one server. Raise servers to 2 and confirm per-GPU VRAM drops and utilization falls below 100%. Switch the fabric from 1.6TbE to 100GbE and confirm time-to-first-token worsens noticeably while tokens/sec barely moves — that is Finding B working.

- [ ] **Step 7: Lint, typecheck, build**

Run: `rtk proxy npm run lint && rtk proxy npm run typecheck && npm run build`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
rtk git add src/hooks/useInferenceCalculation.ts src/workers/calculation.worker.ts src/engines/inference.integration.test.ts
rtk git commit -m "$(cat <<'EOF'
feat: route inference through the multi-node calculation path

Both the hook and the worker now call calculateMultiNodeVRAM. Interconnect
validation is passed the per-node GPU count, since NVLink and Infinity
Fabric bounds apply inside a server, not across the cluster.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Surface the topology in results, and document it

**Files:**
- Modify: whichever output component renders the multi-GPU breakdown (find with `grep -rln "multiGPU" src/components/outputs`)
- Modify: `README.md`, `CHANGELOG.md`, `docs/vram-calculation-pitfalls.md`

**Interfaces:**
- Consumes: the node fields on `MultiGPUVRAMBreakdown` (Task 5).
- Produces: no new exports.

- [ ] **Step 1: Find the output component**

Run: `grep -rln "multiGPU" src/components/outputs`

- [ ] **Step 2: Add a topology summary**

In the component that renders the multi-GPU breakdown, add — inside the block that already guards on a multi-GPU result being present:

```tsx
{multiGPU.numNodes > 1 && (
  <div className="mt-3 text-xs text-gray-600 dark:text-gray-400 space-y-1">
    <p>
      {multiGPU.numNodes} servers × {multiGPU.gpusPerNode} GPUs = {multiGPU.numGPUs} GPUs ·
      tensor parallel within each server, pipeline parallel across them
    </p>
    <p>
      Efficiency: {(multiGPU.intraNodeEfficiency * 100).toFixed(0)}% intra-server ·{' '}
      {(multiGPU.interNodePrefillEfficiency * 100).toFixed(0)}% inter-server on prefill ·{' '}
      {(multiGPU.interNodeDecodeEfficiency * 100).toFixed(0)}% on decode ·{' '}
      {(multiGPU.bubbleEfficiency * 100).toFixed(0)}% pipeline fill
    </p>
  </div>
)}
```

- [ ] **Step 3: Add the pitfalls entries**

In `docs/vram-calculation-pitfalls.md`, append a section matching the file's existing heading style:

```markdown
## Multi-node inference

### Port speed is not node bandwidth

The standard AI node has one NIC per GPU, and NCCL/RCCL stripe a pipeline stage
handoff across all of them. An 8-GPU node on 800GbE has 8 × 100 = 800 GB/s of
scale-out bandwidth, not 100. Sizing a cluster from a single port's speed
understates inter-node capacity eightfold and makes multi-node look far worse
than it is.

### Tensor parallelism does not cross a node boundary

TP does a per-layer allreduce. Over NVLink-5 (1800 GB/s) or Infinity Fabric
(1075 GB/s) that is cheap; over even the fastest scale-out fabric it is an order
of magnitude slower and dominates the step time. Real deployments run TP inside
a node and pipeline or data parallelism between nodes. A calculator that offers
TP across nodes as a normal option is describing a configuration nobody runs.

### Prefill and decode do not pay the same network cost

A decode step ships one token's activations across a stage boundary — batch ×
hidden × 2 bytes, kilobytes — so the hop is latency-bound at roughly 10 µs
against a 10–20 ms decode step, near-free. Prefill ships full-sequence
activations and is genuinely bandwidth-bound. Applying one network efficiency to
both penalizes decode throughput and flatters time-to-first-token.

### Pipeline bubbles are not a bandwidth problem

A pipeline of S stages idles S−1 slots at the start and end of every batch,
regardless of how fast the network is. Four nodes at batch 1 lose about 43% of
throughput to fill and drain alone. Modelling only bandwidth makes deep
pipelines look far better than they are.
```

- [ ] **Step 4: Update README and CHANGELOG**

In `README.md`, document multi-node in the features list: GPUs per server, server count, selectable scale-out fabric, and that TP runs intra-server while PP runs inter-server.

In `CHANGELOG.md`, under the same Unreleased heading Task 2 created:

```markdown
### Added
- Multi-node inference: configure GPUs per server and server count, with a selectable scale-out fabric (1.6TbE / 800GbE SONiC RoCEv2, InfiniBand XDR/NDR, 400GbE, 100GbE, or a custom port speed). Tensor parallelism runs inside a server, pipeline parallelism across servers.

### Fixed
- Pipeline parallelism no longer replicates the full all-layer KV cache on every stage. Each stage owns a slice of layers, so it holds only that slice's cache. The previous behaviour overstated the KV term by the stage count.
- The prefill roofline now uses a prefill-specific scaling efficiency. A single efficiency applied to both rooflines penalized decode throughput and flattered time-to-first-token in multi-node configurations.
```

- [ ] **Step 5: Full verification**

Run: `npx vitest run && rtk proxy npm run lint && rtk proxy npm run typecheck && npm run build && npm run test:coverage`

Expected: all pass, and coverage stays at or above 75% lines/functions/branches/statements on `src/engines/` and `src/utils/`. `fabric.ts` and `multi-node.ts` are both well covered by their own suites; if coverage dipped, the gap is an uncovered branch in one of them — add the case rather than lowering the threshold.

- [ ] **Step 6: Commit**

```bash
rtk git add src/components/outputs/ README.md CHANGELOG.md docs/vram-calculation-pitfalls.md
rtk git commit -m "$(cat <<'EOF'
feat(ui): show node topology in results, document multi-node pitfalls

Results panel reports the server x GPU split and breaks efficiency into
its four terms, so a surprising number can be traced to its cause.

Adds four pitfalls entries: port speed is not node bandwidth, TP does not
cross a node boundary, prefill and decode pay different network costs,
and pipeline bubbles are not a bandwidth problem.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
)"
```

---

# Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Finding A — PP KV cache | Task 3 |
| Finding B — split rooflines | Tasks 5, 7 |
| §1 GPU data | Task 2 |
| §2 Infinity Fabric | Task 1 |
| §3 Fabric tier, formulas, custom entry, bubble | Task 4 |
| §4 Compose-not-extend, breakdown fields | Tasks 5, 6 |
| §5 Store, URL, comparison, UI | Tasks 8, 9, 10 |
| §6 Testing and docs | Tasks 3–11; docs in 2 and 11 |
| §7 Sequencing | Phase 1 = Tasks 1–2; Phase 2 = Tasks 3–11 |

No gaps.

**Type consistency:** `prefillScalingEfficiency` is defined in Task 5 and consumed under that exact name in Tasks 6 and 7. `resolveFabricSpec`, `perNodeFabricGBps`, `fabricPrefillEfficiency`, `fabricDecodeEfficiency` and `pipelineBubbleEfficiency` are defined in Task 4 and used under those names in Tasks 6, 9 and 10. `calculateMultiNodeVRAM`'s single-object parameter shape is declared in Task 6's Interfaces block and matched at both call sites in Task 10. `FabricType` / `FabricSpec` / `CustomFabricInput` are defined in Task 4 and imported in Tasks 6, 8, 9, 10.

**Known lookups left to the implementer** (each has an exact command, not a guess): the output component in Task 11 Step 1, the panel mounting `GPUCountSelector` in Task 9 Step 5, the worker's message sender in Task 10 Step 4, and any additional exhaustive `Record<InterconnectType, …>` in Task 1 Step 4. These are `grep`/`typecheck` lookups against the live tree rather than unknowns in the design.
