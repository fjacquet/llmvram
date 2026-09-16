# GPU Scale-Up Domain Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded 8-GPU-per-node bound with a per-GPU `max_gpus_per_node` field, so GB300 NVL72 (72 GPUs in one NVLink domain) is representable and single-GPU parts cannot be multiplied.

**Architecture:** One new required field on the Zod `GPUSchema`, populated for all 27 existing rows from `scripts/fetch-gpus.ts` (the generator — `src/data/gpus.json` is generated output and must never be hand-edited). The calculation engine stays GPU-agnostic: its guard widens to a flat sanity bound of 72, while the per-GPU limit is enforced at the store boundary by a pure `clampGPUCount` helper. The UI slider reads its maximum from the selected GPU.

**Tech Stack:** TypeScript strict (`noUncheckedIndexedAccess`), Zod schemas as the single source of truth, Zustand store, Vitest + jsdom, Biome (2-space, single quotes, no semicolons, 100-char lines).

**Spec:** `docs/superpowers/specs/2026-09-16-gpu-domain-size-design.md`

## Global Constraints

- `src/data/gpus.json` is GENERATED from the `GPUS` array in `scripts/fetch-gpus.ts`. Never hand-edit the JSON — edits are silently lost on the next `npm run refresh:gpus`. Every data change in this plan edits the script, then regenerates.
- GPUs in `gpus.json` are grouped by manufacturer, NOT sorted alphabetically. Do not re-sort. (The alphabetical rule applies to `models.json` only.)
- `max_gpus_per_node` = min(coherent interconnect limit, largest shipping chassis slot count). For parts with no coherent domain (anything on PCIe), the chassis bound alone.
- `max_gpus_per_node` is a HARD bound (cannot be built). `recommendedMaxTPDegree` in `INTERCONNECT_SPECS` stays a SOFT warning (buildable, scales badly). Both remain; neither replaces the other.
- Database ids never change in this plan. `nvidia-b200-192gb` keeps its id while its `vram_gb` becomes 180; `nvidia-gb300-288gb` keeps its id and becomes the HGX B300 row. A stale id costs nothing at runtime; a changed id breaks every shared link naming it.
- Clamping is SILENT. No toast, no warning banner, on any clamp path.
- `numGPUs` in the store means GPUs PER NODE in inference mode; total GPUs is `numGPUs × numNodes`. In training mode it is the total (multi-node training is a Non-Goal).
- Biome treats unused imports and variables as errors. Run `npm run lint` before every commit.
- Test files are excluded from `tsconfig.app.json` and are typechecked by nothing. Build test fixtures by calling real factories where one exists, rather than hand-writing object literals that go stale silently.
- `validateGPUs(gpusData)` runs at MODULE LOAD in `src/store/uiStore.ts:22` and `src/components/inputs/GPUSelector.tsx:18`. A required schema field with unpopulated data rows crashes the app and every test that imports the store. Task 1 must land schema and data in one commit.

---

### Task 1: Add `max_gpus_per_node` to the schema and populate every GPU row

**Files:**
- Modify: `src/utils/schemas.ts:22-23` (insert field into `GPUSchema`)
- Modify: `scripts/fetch-gpus.ts` (27 GPU rows)
- Modify: `src/types/gpu.ts:15-29` (`createCustomGPU`)
- Modify: `src/hooks/useURLSync.ts:76-101` (two custom-GPU object literals)
- Regenerate: `src/data/gpus.json` (via `npm run refresh:gpus`)
- Test: `src/utils/schemas.test.ts`, `src/utils/gpus.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `GPU.max_gpus_per_node: number` — a required, positive integer on every `GPU`. Tasks 2, 3, 5 and 6 all read it.

- [ ] **Step 1: Write the failing schema tests**

Append to the existing `describe('validateGPUs / validateModels array helpers')` block's sibling scope in `src/utils/schemas.test.ts` (top level of the file, after the existing GPU describes):

```ts
describe('GPUSchema max_gpus_per_node', () => {
  it('rejects a GPU with no max_gpus_per_node', () => {
    const { max_gpus_per_node, ...withoutField } = {
      ...validDatacenterGPU,
      max_gpus_per_node: 8,
    }
    expect(() => GPUSchema.parse(withoutField)).toThrow(ZodError)
  })

  it('rejects a non-integer max_gpus_per_node', () => {
    expect(() =>
      GPUSchema.parse({ ...validDatacenterGPU, max_gpus_per_node: 4.5 }),
    ).toThrow(ZodError)
  })

  it('rejects a zero or negative max_gpus_per_node', () => {
    expect(() => GPUSchema.parse({ ...validDatacenterGPU, max_gpus_per_node: 0 })).toThrow(
      ZodError,
    )
  })

  it('accepts a positive integer max_gpus_per_node', () => {
    const parsed = GPUSchema.parse({ ...validDatacenterGPU, max_gpus_per_node: 72 })
    expect(parsed.max_gpus_per_node).toBe(72)
  })
})
```

Note: `validDatacenterGPU` is an existing fixture in that file. It has no `max_gpus_per_node` yet, so add `max_gpus_per_node: 8` to the `validDatacenterGPU` and `validAppleSiliconGPU` fixtures at the top of the file — otherwise every pre-existing test in the file starts failing once the field is required. Give the Apple fixture `max_gpus_per_node: 1`.

Add to `src/utils/gpus.test.ts` inside the existing top-level `describe`:

```ts
it('every GPU row declares max_gpus_per_node as a positive integer', () => {
  const result = validateGPUs(gpusData)
  for (const gpu of result) {
    expect(Number.isInteger(gpu.max_gpus_per_node)).toBe(true)
    expect(gpu.max_gpus_per_node).toBeGreaterThan(0)
  }
})

it('bounds every Apple Silicon row at a single GPU', () => {
  const result = validateGPUs(gpusData)
  const apple = result.filter((g) => g.tier === 'apple-silicon')
  expect(apple.length).toBeGreaterThan(0)
  for (const gpu of apple) {
    expect(gpu.max_gpus_per_node).toBe(1)
  }
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/utils/schemas.test.ts src/utils/gpus.test.ts`
Expected: FAIL. The schema tests fail because `GPUSchema` has no `max_gpus_per_node` (an unknown key is stripped, not rejected, so `parse` succeeds and the `toThrow` assertions fail). The `gpus.test.ts` tests fail because `gpu.max_gpus_per_node` is `undefined`.

- [ ] **Step 3: Add the field to `GPUSchema`**

In `src/utils/schemas.ts`, insert immediately after the `bus_width` line (currently line 22) and before the `// Performance` comment:

```ts
  bus_width: z.number().int().nonnegative(), // 0 for unified memory (Apple Silicon)

  /**
   * Largest GPU count that can exist inside one node for this part.
   *
   * Derivation: min(coherent interconnect limit, largest shipping chassis slot
   * count). For parts with no coherent domain — anything on PCIe — the chassis
   * bound alone.
   *
   * This is a HARD bound: configurations above it cannot be built. It is not
   * the same as INTERCONNECT_SPECS.recommendedMaxTPDegree, which is a SOFT
   * warning about configurations that are buildable but scale badly. Eight
   * RTX PRO 6000 in a Dell XE7745 is both buildable (8) and a poor tensor-
   * parallel target (4).
   *
   * Required, not optional: a row added later must state its own bound rather
   * than inherit a default that happens to be wrong.
   */
  max_gpus_per_node: z.number().int().positive(),
```

- [ ] **Step 4: Populate all 27 rows in the generator**

In `scripts/fetch-gpus.ts`, add a `max_gpus_per_node` line to each GPU object, placed immediately before that object's `tier:` line. Use exactly these values:

| id | `max_gpus_per_node` |
|---|---|
| `nvidia-h100-80gb-pcie` | 8 |
| `nvidia-h100-80gb-sxm` | 8 |
| `nvidia-h200-141gb` | 8 |
| `nvidia-b200-192gb` | 8 |
| `nvidia-gb300-288gb` | 8 |
| `nvidia-a100-80gb-pcie` | 8 |
| `nvidia-a100-80gb-sxm` | 8 |
| `nvidia-l40s` | 8 |
| `nvidia-rtx-pro-6000-server` | 8 |
| `nvidia-rtx-6000-ada` | 8 |
| `nvidia-rtx-5090` | 8 |
| `nvidia-rtx-4090` | 8 |
| `nvidia-rtx-3090` | 8 |
| `nvidia-gb300-desktop-252gb` | 1 |
| `nvidia-gb10` | 2 |
| `amd-mi355x` | 8 |
| `amd-mi350x` | 8 |
| `amd-mi325x` | 8 |
| `amd-mi300x` | 8 |
| `apple-m1-ultra` | 1 |
| `apple-m2-ultra` | 1 |
| `apple-m3-ultra` | 1 |
| `apple-m5-max` | 1 |
| `apple-m4-max` | 1 |
| `apple-m3-max` | 1 |
| `apple-m2-max` | 1 |
| `apple-m1-max` | 1 |

Example of the edit shape, for the first row:

```ts
  {
    id: 'nvidia-h100-80gb-pcie',
    name: 'NVIDIA H100 80GB PCIe',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 2000,
    memory_type: 'HBM3',
    bus_width: 5120,
    fp16_tflops: 989,
    fp32_tflops: 51,
    tdp_watts: 350,
    interconnect: 'nvlink-4',
    max_gpus_per_node: 8,
    tier: 'datacenter',
    spec_url: 'https://www.nvidia.com/en-us/data-center/h100/',
  },
```

Rationale for the non-8 values, so a reviewer can check them: `nvidia-gb300-desktop-252gb` is the DGX Station / Dell Pro Max single Grace-Blackwell superchip — one GPU, no baseboard. `nvidia-gb10` is DGX Spark, where exactly two units link over ConnectX-7. Apple Silicon has unified memory and no multi-GPU interconnect at all (`resolveInterconnect` maps `unified` to `none`). The consumer RTX rows take 8 on the chassis rule — eight-card PCIe inference rigs exist, and their poor scaling is already expressed by `recommendedMaxTPDegree: 4` for PCIe.

- [ ] **Step 5: Add the field to the three custom-GPU construction sites**

The field is required, so every place that builds a `GPU` object literal must supply it. Custom GPUs get 8 — a generic server.

In `src/types/gpu.ts`, inside `createCustomGPU`, add before `tier`:

```ts
    interconnect: 'none',
    max_gpus_per_node: 8,
  }
}
```

Full corrected function body:

```ts
export function createCustomGPU(input: CustomGPUInput): GPU {
  return {
    id: `custom-${Date.now()}`,
    name: input.name,
    manufacturer: 'nvidia', // Default for custom
    vram_gb: input.vram_gb,
    memory_bandwidth_gbps: input.memory_bandwidth_gbps || 0,
    memory_type: 'Custom',
    bus_width: 0,
    fp16_tflops: input.fp16_tflops,
    fp32_tflops: input.fp32_tflops,
    tier: 'consumer',
    interconnect: 'none',
    max_gpus_per_node: 8,
  }
}
```

In `src/hooks/useURLSync.ts` there are TWO object literals restoring a custom GPU (the `urlState.gpuId` fallback branch and the `urlState.customGPU`-without-id branch). Both currently read:

```ts
        store.setSelectedGPU({
          id: 'custom-restored',
          manufacturer: 'nvidia',
          memory_type: 'Custom',
          bus_width: 0,
          tier: 'consumer',
          interconnect: 'none',
          ...urlState.customGPU,
        })
```

Add `max_gpus_per_node: 8,` after `interconnect: 'none',` in BOTH. Note the spread comes last, so the literal's value is the default and any future serialized field would win — that ordering is existing behaviour, leave it.

- [ ] **Step 6: Regenerate the JSON**

Run: `npm run refresh:gpus`
Expected: writes `src/data/gpus.json`. The script calls `validateGPUs(GPUS)` at line 433 before writing, so a missed row fails here with a `ZodError` naming the index.

- [ ] **Step 7: Run the full suite**

Run: `npx vitest run`
Expected: PASS, including the new tests. Then `npm run typecheck` and `npm run lint` — both clean.

- [ ] **Step 8: Commit**

```bash
git add src/utils/schemas.ts src/utils/schemas.test.ts src/utils/gpus.test.ts \
  scripts/fetch-gpus.ts src/data/gpus.json src/types/gpu.ts src/hooks/useURLSync.ts
git commit -m "feat: add required max_gpus_per_node to the GPU schema"
```

---

### Task 2: Correct B200 capacity to 180GB

**Files:**
- Modify: `scripts/fetch-gpus.ts:68-82` (the `nvidia-b200-192gb` block)
- Regenerate: `src/data/gpus.json`
- Test: `src/utils/gpus.test.ts`

**Interfaces:**
- Consumes: `GPU.max_gpus_per_node` from Task 1 (the row already carries 8; this task does not touch it).
- Produces: nothing new. The row id `nvidia-b200-192gb` is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `src/utils/gpus.test.ts`:

```ts
it('lists B200 at its allocatable 180GB, not the 192GB stack size', () => {
  const result = validateGPUs(gpusData)
  const b200 = result.find((g) => g.id === 'nvidia-b200-192gb')
  expect(b200).toBeDefined()
  expect(b200?.vram_gb).toBe(180)
  expect(b200?.name).not.toContain('192')
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/utils/gpus.test.ts`
Expected: FAIL — `expected 192 to be 180`.

- [ ] **Step 3: Change the generator row**

In `scripts/fetch-gpus.ts`, change two lines of the `nvidia-b200-192gb` object. The id stays exactly as it is:

```ts
  {
    id: 'nvidia-b200-192gb',
    name: 'NVIDIA B200 180GB',
    manufacturer: 'nvidia',
    vram_gb: 180,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3e',
    bus_width: 8192,
    fp16_tflops: 4500,
    fp32_tflops: 90,
    tdp_watts: 1000,
    interconnect: 'nvlink-5',
    max_gpus_per_node: 8,
    tier: 'datacenter',
    spec_url: 'https://www.nvidia.com/en-us/data-center/b200/',
  },
```

Add this comment directly above the object, so the next person does not "fix" it back:

```ts
  // HGX B200 ships 1.44TB across 8 GPUs = 180GB each. 192GB is the physical
  // HBM3e stack size before reserved capacity; 180GB is the software-visible
  // figure in NVIDIA's OEM documentation and in Dell XE9680L / XE9685L and
  // Lenovo ThinkSystem listings. A VRAM calculator wants the allocatable one.
  // The id keeps its stale "192gb" suffix on purpose: changing it would break
  // every shared link naming this GPU.
```

- [ ] **Step 4: Regenerate and verify the test passes**

Run: `npm run refresh:gpus && npx vitest run src/utils/gpus.test.ts`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `npx vitest run`
Expected: PASS. Any other test asserting a B200 total will shift by the ratio 180/192 = 0.9375; if one fails, update its expected value — do not revert the data.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch-gpus.ts src/data/gpus.json src/utils/gpus.test.ts
git commit -m "fix: list B200 at its allocatable 180GB rather than 192GB stack size"
```

---

### Task 3: Split GB300 into HGX B300 and NVL72 rows

**Files:**
- Modify: `scripts/fetch-gpus.ts:83-97` (the `nvidia-gb300-288gb` block, plus a new sibling object after it)
- Regenerate: `src/data/gpus.json`
- Test: `src/utils/gpus.test.ts`

**Interfaces:**
- Consumes: `GPU.max_gpus_per_node` from Task 1.
- Produces: a new GPU id `nvidia-gb300-nvl72` with `max_gpus_per_node: 72` — the only row in the database above 8. Task 4's engine test uses it conceptually; Task 6's UI reads its bound.

- [ ] **Step 1: Write the failing test**

Add to `src/utils/gpus.test.ts`:

```ts
it('offers GB300 as both an 8-GPU HGX baseboard and a 72-GPU NVL72 rack', () => {
  const result = validateGPUs(gpusData)
  const hgx = result.find((g) => g.id === 'nvidia-gb300-288gb')
  const nvl72 = result.find((g) => g.id === 'nvidia-gb300-nvl72')

  expect(hgx).toBeDefined()
  expect(nvl72).toBeDefined()
  expect(hgx?.max_gpus_per_node).toBe(8)
  expect(nvl72?.max_gpus_per_node).toBe(72)
})

it('gives the two GB300 rows identical silicon specs', () => {
  const result = validateGPUs(gpusData)
  const hgx = result.find((g) => g.id === 'nvidia-gb300-288gb')
  const nvl72 = result.find((g) => g.id === 'nvidia-gb300-nvl72')

  expect(nvl72?.vram_gb).toBe(hgx?.vram_gb)
  expect(nvl72?.memory_bandwidth_gbps).toBe(hgx?.memory_bandwidth_gbps)
  expect(nvl72?.fp16_tflops).toBe(hgx?.fp16_tflops)
  expect(nvl72?.interconnect).toBe(hgx?.interconnect)
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/utils/gpus.test.ts`
Expected: FAIL — `expected undefined to be defined` for the NVL72 row.

- [ ] **Step 3: Relabel the existing row and add the NVL72 sibling**

In `scripts/fetch-gpus.ts`, replace the `nvidia-gb300-288gb` object with these two objects. The first keeps its id:

```ts
  // GB300 (Blackwell Ultra) ships in two platforms with different scale-up
  // domains, so it needs two rows. Same silicon, same capacity, same
  // bandwidth — only max_gpus_per_node differs.
  //
  // HGX B300: an 8-GPU baseboard in a conventional x86 server (Dell XE9680L,
  // XE9685L, XE9785L). NVL72: 72 Blackwell Ultra GPUs in one NVLink domain,
  // 130TB/s of switch bandwidth, shipping as Dell PowerEdge XE9712.
  //
  // The HGX row keeps the original id so existing shared links resolve.
  {
    id: 'nvidia-gb300-288gb',
    name: 'NVIDIA GB300 (HGX B300, 8-GPU) 288GB',
    manufacturer: 'nvidia',
    vram_gb: 288,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3e',
    bus_width: 8192,
    fp16_tflops: 5000,
    fp32_tflops: 83,
    tdp_watts: 1400,
    interconnect: 'nvlink-5',
    max_gpus_per_node: 8,
    tier: 'datacenter',
    spec_url: 'https://www.nvidia.com/en-us/data-center/gb300-nvl72/',
  },
  {
    id: 'nvidia-gb300-nvl72',
    name: 'NVIDIA GB300 NVL72 (72-GPU rack) 288GB',
    manufacturer: 'nvidia',
    vram_gb: 288,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3e',
    bus_width: 8192,
    fp16_tflops: 5000,
    fp32_tflops: 83,
    tdp_watts: 1400,
    interconnect: 'nvlink-5',
    max_gpus_per_node: 72,
    tier: 'datacenter',
    spec_url: 'https://www.nvidia.com/en-us/data-center/gb300-nvl72/',
  },
```

Keep both rows inside the NVIDIA group — the file is grouped by manufacturer and must not be re-sorted.

- [ ] **Step 4: Regenerate and verify the tests pass**

Run: `npm run refresh:gpus && npx vitest run src/utils/gpus.test.ts`
Expected: PASS. The GPU count in the database goes from 27 to 28; if a test asserts a count, update it.

- [ ] **Step 5: Run the full suite**

Run: `npx vitest run`, then `npm run lint`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch-gpus.ts src/data/gpus.json src/utils/gpus.test.ts
git commit -m "feat: add GB300 NVL72 as a 72-GPU row alongside HGX B300"
```

---

### Task 4: Widen the engine guard to `MAX_GPUS_PER_NODE`

**Files:**
- Modify: `src/engines/constants.ts` (add the exported constant)
- Modify: `src/engines/multi-gpu.ts:228` (doc comment) and `:253-255` (the guard)
- Test: `src/engines/multi-gpu.test.ts:418-440`, `src/engines/multi-node.test.ts:169`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `export const MAX_GPUS_PER_NODE = 72` from `src/engines/constants.ts`. Task 5 imports it as the fallback bound for a GPU-less store state.

- [ ] **Step 1: Update the two existing boundary tests and add a new one**

In `src/engines/multi-gpu.test.ts`, the test at line 431 currently asserts a throw above 8. Replace that test and add a passing-case test beside it. Keep the `numGPUs < 1` test at line 418 exactly as it is — the lower bound does not move.

```ts
  it('throws error for numGPUs > 72', () => {
    expect(() =>
      calculateMultiGPUVRAM({
        singleGPU: baseBreakdown,
        model: testModel,
        gpuVramGB: 80,
        numGPUs: 73,
        strategy: 'tensor-parallel',
        gpu: testGPU,
      }),
    ).toThrow(/numGPUs must be between 1 and 72/)
  })

  it('accepts a 72-GPU node, for NVL72-class racks', () => {
    const result = calculateMultiGPUVRAM({
      singleGPU: baseBreakdown,
      model: testModel,
      gpuVramGB: 288,
      numGPUs: 72,
      strategy: 'tensor-parallel',
      gpu: testGPU,
    })
    expect(result.numGPUs).toBe(72)
    expect(result.totalPerGPU.toNumber()).toBeGreaterThan(0)
    expect(result.totalPerGPU.isFinite()).toBe(true)
  })
```

Match the exact argument shape used by the neighbouring tests in that file — read the test at line 418 and mirror its call, including whatever fixture names it uses for the breakdown, model and GPU. Do not invent fixture names.

In `src/engines/multi-node.test.ts:169`, the assertion matches the literal message. Change:

```ts
      /numGPUs must be between 1 and 8/,
```

to:

```ts
      /numGPUs must be between 1 and 72/,
```

Check the surrounding test: if it passes a value like 9 expecting a throw, that value is now legal and the test no longer tests anything. Raise it to 73.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/multi-gpu.test.ts src/engines/multi-node.test.ts`
Expected: FAIL — the 72-GPU case throws `numGPUs must be between 1 and 8, got 72`, and the regex assertions do not match the old message.

- [ ] **Step 3: Add the constant**

In `src/engines/constants.ts`, add near the `INTERCONNECT_SPECS` block:

```ts
/**
 * Sanity bound on GPUs inside one node
 *
 * 72 is the largest scale-up domain in shipping hardware: NVIDIA GB300 NVL72
 * places 72 GPUs in a single NVLink domain. This is a flat guard against
 * absurd input, NOT the per-GPU limit — that is GPU.max_gpus_per_node, and it
 * is enforced at the store boundary so the engine stays GPU-agnostic.
 */
export const MAX_GPUS_PER_NODE = 72
```

- [ ] **Step 4: Widen the guard**

In `src/engines/multi-gpu.ts`, import the constant alongside the existing imports from `./constants`, then replace lines 253-255:

```ts
  if (numGPUs < 1 || numGPUs > MAX_GPUS_PER_NODE) {
    throw new Error(`numGPUs must be between 1 and ${MAX_GPUS_PER_NODE}, got ${numGPUs}`)
  }
```

Update the doc comment at line 228 from `@throws Error if numGPUs < 1 or > 8` to:

```ts
 * @throws Error if numGPUs < 1 or > MAX_GPUS_PER_NODE (72)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run src/engines/multi-gpu.test.ts src/engines/multi-node.test.ts`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `npx vitest run`, then `npm run typecheck` and `npm run lint`
Expected: PASS, clean. The fabric math needs no change — `perNodeFabricGBps = portGBps × gpusPerNode` encodes one NIC per GPU, which holds at 72 (NVL72 carries one 800Gb/s ConnectX-8 port per GPU, 57.6Tb/s per rack = 100 GB/s × 72).

- [ ] **Step 7: Commit**

```bash
git add src/engines/constants.ts src/engines/multi-gpu.ts \
  src/engines/multi-gpu.test.ts src/engines/multi-node.test.ts
git commit -m "feat: widen the per-node GPU guard from 8 to 72"
```

---

### Task 5: Clamp the GPU count at the store boundary

**Files:**
- Create: `src/utils/gpuLimits.ts`
- Create: `src/utils/gpuLimits.test.ts`
- Modify: `src/store/uiStore.ts:170` (`setSelectedGPU`) and `:177` (`setNumGPUs`)

**Interfaces:**
- Consumes: `GPU.max_gpus_per_node` (Task 1), `MAX_GPUS_PER_NODE` from `@engines/constants` (Task 4).
- Produces: `clampGPUCount(numGPUs: number, gpu: GPU | null): number` from `@utils/gpuLimits`. Task 6's selector imports nothing from it — the UI reads the bound directly — but relies on the store never holding an out-of-range value.

The URL path needs no separate change: `src/hooks/useURLSync.ts` calls `store.setSelectedGPU(...)` at line 75 BEFORE `store.setNumGPUs(urlState.ng)` at line 111, so a clamping `setNumGPUs` already sees the restored GPU. Verify that ordering still holds before relying on it.

- [ ] **Step 1: Write the failing tests for the helper**

Create `src/utils/gpuLimits.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import type { GPU } from '@utils/schemas'
import { clampGPUCount } from './gpuLimits'

function gpuWithBound(max: number): GPU {
  return {
    id: 'test-gpu',
    name: 'Test GPU',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 2000,
    memory_type: 'HBM3',
    bus_width: 5120,
    max_gpus_per_node: max,
    tier: 'datacenter',
  }
}

describe('clampGPUCount', () => {
  it('leaves a count inside the bound untouched', () => {
    expect(clampGPUCount(4, gpuWithBound(8))).toBe(4)
  })

  it('clamps a count above the bound down to it', () => {
    expect(clampGPUCount(8, gpuWithBound(4))).toBe(4)
  })

  it('clamps to 1 for a single-GPU part', () => {
    expect(clampGPUCount(8, gpuWithBound(1))).toBe(1)
  })

  it('allows 72 for an NVL72-class part', () => {
    expect(clampGPUCount(72, gpuWithBound(72))).toBe(72)
  })

  it('floors at 1 for zero and negative input', () => {
    expect(clampGPUCount(0, gpuWithBound(8))).toBe(1)
    expect(clampGPUCount(-3, gpuWithBound(8))).toBe(1)
  })

  it('truncates a fractional count', () => {
    expect(clampGPUCount(3.7, gpuWithBound(8))).toBe(3)
  })

  it('falls back to the engine sanity bound when no GPU is selected', () => {
    expect(clampGPUCount(72, null)).toBe(72)
    expect(clampGPUCount(999, null)).toBe(72)
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/utils/gpuLimits.test.ts`
Expected: FAIL — cannot resolve `./gpuLimits`.

- [ ] **Step 3: Write the helper**

Create `src/utils/gpuLimits.ts`:

```ts
import { MAX_GPUS_PER_NODE } from '@engines/constants'
import type { GPU } from '@utils/schemas'

/**
 * Clamp a per-node GPU count to what the selected GPU can actually form
 *
 * The bound is GPU.max_gpus_per_node — a hard limit, min(coherent interconnect
 * limit, largest shipping chassis slot count). With no GPU selected there is
 * nothing to bound against, so the engine's flat sanity bound applies.
 *
 * Clamping is silent by design: no toast, no warning. A shared link carrying a
 * count above the bound will render a different number than its sender saw.
 */
export function clampGPUCount(numGPUs: number, gpu: GPU | null): number {
  const max = gpu?.max_gpus_per_node ?? MAX_GPUS_PER_NODE
  return Math.min(Math.max(1, Math.trunc(numGPUs)), max)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/utils/gpuLimits.test.ts`
Expected: PASS.

If the import of `@engines/constants` from `src/utils/` creates a circular import (it should not — `src/engines/constants.ts` imports only `decimal.js` and `./types`), Vitest will report `Cannot access 'X' before initialization`. Should that happen, move `MAX_GPUS_PER_NODE` into `src/utils/gpuLimits.ts` and re-export it from `src/engines/constants.ts` instead.

- [ ] **Step 5: Wire the clamp into the store**

In `src/store/uiStore.ts`, add the import:

```ts
import { clampGPUCount } from '@utils/gpuLimits'
```

Replace the `setSelectedGPU` action (line 170):

```ts
      setSelectedGPU: (gpu) =>
        set((state) => ({
          selectedGPU: gpu,
          interconnectOverride: null,
          numGPUs: clampGPUCount(state.numGPUs, gpu),
        })),
```

Replace the `setNumGPUs` action (line 177):

```ts
      setNumGPUs: (numGPUs) =>
        set((state) => ({ numGPUs: clampGPUCount(numGPUs, state.selectedGPU) })),
```

- [ ] **Step 6: Write a store test**

Add to the existing uiStore test file (find it with `ls src/store/*.test.ts`; if there is no `uiStore.test.ts`, create one following the pattern in `src/store/comparisonStore.test.ts`). Note the project gotcha: Zustand stores with `persist` middleware fail under jsdom, so mock the store with `vi.hoisted()` + `vi.mock()` to build a plain store without persist, exactly as the existing store tests do.

```ts
it('clamps numGPUs down when switching to a single-GPU part', () => {
  const store = useUIStore.getState()
  store.setSelectedGPU(findGPUById('nvidia-h100-80gb-sxm'))
  store.setNumGPUs(8)
  expect(useUIStore.getState().numGPUs).toBe(8)

  store.setSelectedGPU(findGPUById('apple-m3-ultra'))
  expect(useUIStore.getState().numGPUs).toBe(1)
})

it('rejects a count above the selected GPU bound', () => {
  const store = useUIStore.getState()
  store.setSelectedGPU(findGPUById('nvidia-h100-80gb-sxm'))
  store.setNumGPUs(72)
  expect(useUIStore.getState().numGPUs).toBe(8)
})

it('allows 72 on an NVL72 part', () => {
  const store = useUIStore.getState()
  store.setSelectedGPU(findGPUById('nvidia-gb300-nvl72'))
  store.setNumGPUs(72)
  expect(useUIStore.getState().numGPUs).toBe(72)
})
```

- [ ] **Step 7: Run the full suite**

Run: `npx vitest run`, then `npm run typecheck` and `npm run lint`
Expected: PASS, clean.

- [ ] **Step 8: Commit**

```bash
git add src/utils/gpuLimits.ts src/utils/gpuLimits.test.ts src/store/uiStore.ts \
  src/store/uiStore.test.ts
git commit -m "feat: clamp the per-node GPU count to the selected GPU bound"
```

---

### Task 6: Drive the slider from the selected GPU and fix its tooltip

**Files:**
- Modify: `src/components/inputs/GPUCountSelector.tsx` (whole component)
- Test: `src/components/inputs/GPUCountSelector.test.tsx` (create)

**Interfaces:**
- Consumes: `GPU.max_gpus_per_node` (Task 1), the clamping store actions (Task 5).
- Produces: nothing consumed elsewhere.

The current tooltip claims 8 "is the size of a fully connected GPU domain in current hardware". That has been false since GB200 NVL72 and is the specific copy this task removes.

- [ ] **Step 1: Write the failing component test**

Create `src/components/inputs/GPUCountSelector.test.tsx`, following the setup in the existing `src/components/inputs/NodeCountSelector.test.tsx` (read it first — it shows how this project mocks the store for component tests):

```tsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { GPUCountSelector } from './GPUCountSelector'

describe('GPUCountSelector', () => {
  it('caps the slider at the selected GPU max_gpus_per_node', () => {
    // Arrange the mocked store with an 8-GPU part selected.
    render(<GPUCountSelector />)
    const slider = screen.getByLabelText(/GPUs per server/i)
    expect(slider).toHaveAttribute('max', '8')
  })

  it('raises the cap to 72 for an NVL72 part', () => {
    // Arrange the mocked store with nvidia-gb300-nvl72 selected.
    render(<GPUCountSelector />)
    expect(screen.getByLabelText(/GPUs per server/i)).toHaveAttribute('max', '72')
  })

  it('renders no slider for a single-GPU part', () => {
    // Arrange the mocked store with apple-m3-ultra selected.
    render(<GPUCountSelector />)
    expect(screen.queryByRole('slider')).toBeNull()
    expect(screen.getByText(/single GPU/i)).toBeInTheDocument()
  })
})
```

Replace each `// Arrange ...` comment with the actual store arrangement that `NodeCountSelector.test.tsx` uses — do not leave them as comments.

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/components/inputs/GPUCountSelector.test.tsx`
Expected: FAIL — the slider's `max` is the hardcoded `8` in all cases, and a slider renders for Apple Silicon.

- [ ] **Step 3: Rewrite the component**

Replace `src/components/inputs/GPUCountSelector.tsx` with:

```tsx
import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

/**
 * GPU count selector, bounded by the selected GPU's scale-up domain
 *
 * In inference mode this is the PER-NODE count: total GPUs is this times the
 * server count from NodeCountSelector. The upper bound is the selected GPU's
 * max_gpus_per_node — 8 for an HGX or OAM baseboard, 72 for a GB300 NVL72
 * rack, 2 for a pair of DGX Sparks, 1 for Apple Silicon and the GB300 Desktop
 * Superchip.
 *
 * In training mode there is no server concept: useTrainingCalculation reads
 * this value directly as the total GPU count for ZeRO data-parallel sharding
 * (multi-node training is an explicit spec Non-Goal, and NodeCountSelector is
 * hidden in this mode). The same per-GPU bound still applies, because the
 * GPUs still have to share one node.
 */
export function GPUCountSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const setNumGPUs = useUIStore((s) => s.setNumGPUs)
  const selectedGPU = useUIStore((s) => s.selectedGPU)
  const mode = useUIStore((s) => s.mode)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)

  const isTraining = mode === 'training'
  const maxGPUs = selectedGPU?.max_gpus_per_node ?? 8
  const label = isTraining ? 'Number of GPUs' : 'GPUs per server'

  const tooltip = isTraining
    ? 'GPUs used for data-parallel training, e.g. DeepSpeed ZeRO sharding. Multi-node training is not modelled, so this is the total GPU count.'
    : `GPUs inside one server. Tensor or pipeline parallelism runs at this level, over NVLink, Infinity Fabric or PCIe. Capped at ${maxGPUs} — the largest GPU count this hardware forms in one node.`

  if (maxGPUs === 1) {
    return (
      <div>
        <div className="flex items-center gap-1 mb-1">
          <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </span>
          <InfoTip text={tooltip} />
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Single GPU — {selectedGPU?.name ?? 'this part'} has no multi-GPU interconnect.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="gpu-count"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          {label}
        </label>
        <InfoTip text={tooltip} />
      </div>
      <div className="flex items-center gap-4">
        <input
          id="gpu-count"
          type="range"
          min={1}
          max={maxGPUs}
          step={1}
          value={numGPUs}
          onChange={(e) => setNumGPUs(Number(e.target.value))}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <span className="text-lg font-semibold text-gray-900 dark:text-white w-10 text-center tabular-nums">
          {numGPUs}
        </span>
      </div>
      {numGPUs > 1 && (
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {isTraining
            ? `${numGPUs} GPUs`
            : `${numGPUs} GPUs per server, ${
                shardingStrategy === 'tensor-parallel' ? 'tensor parallel' : 'pipeline parallel'
              }`}
        </p>
      )}
    </div>
  )
}
```

The count readout widens from `w-8` to `w-10` because it now has to fit two digits.

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/components/inputs/GPUCountSelector.test.tsx`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `npx vitest run`, then `npm run typecheck` and `npm run lint`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add src/components/inputs/GPUCountSelector.tsx \
  src/components/inputs/GPUCountSelector.test.tsx
git commit -m "feat: bound the GPU count slider by the selected GPU domain size"
```

---

### Task 7: Update the project documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `CHANGELOG.md`
- Modify: `README.md` (only if it states a GPU count or a per-server maximum — grep first)

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Check what the README claims**

Run: `grep -n "27 GPU\|8 GPU\|GPUs per\|per server" README.md`
If any line states a GPU count or an 8-GPU maximum, update it: the database now holds 28 GPUs and the per-node maximum is per-GPU, up to 72.

- [ ] **Step 2: Add the CLAUDE.md entries**

Under **Key Patterns**, after the existing `numGPUs` bullet, add:

```markdown
- **`max_gpus_per_node` is a hard bound, `recommendedMaxTPDegree` is soft advice**: the first is a per-GPU field meaning "cannot be built" (8 for an HGX/OAM baseboard, 72 for GB300 NVL72, 1 for Apple Silicon); the second lives in `INTERCONNECT_SPECS` and means "buildable but scales badly". Both apply; neither replaces the other.
- **GPU count clamping is silent**: `clampGPUCount` in `src/utils/gpuLimits.ts` bounds `numGPUs` at the store boundary, with no toast. A shared link above the bound renders different numbers than its sender saw.
- **Database ids are deliberately stale**: `nvidia-b200-192gb` holds a 180GB GPU. Ids are never renamed, because a changed id breaks every shared link naming it.
```

Under **Domain Pitfalls**, add:

```markdown
8. **B200 is 180GB, not 192GB**: 192 is the physical HBM3e stack size before reserved capacity. HGX B200 ships 1.44TB across 8 GPUs. Use the allocatable figure.
```

- [ ] **Step 3: Add the CHANGELOG entry**

Under `## [Unreleased]`, add an `### Added` / `### Changed` / `### Fixed` grouping:

```markdown
### Added
- Per-GPU `max_gpus_per_node` field bounding how many GPUs can share one node, derived as min(coherent interconnect limit, largest shipping chassis slot count)
- NVIDIA GB300 NVL72 as a 72-GPU row, alongside the existing GB300 which now names the 8-GPU HGX B300 baseboard

### Changed
- The GPU count slider is bounded by the selected GPU rather than a fixed 8, and shows no slider for single-GPU parts
- The per-node engine guard widened from 8 to 72

### Fixed
- NVIDIA B200 listed at its allocatable 180GB rather than the 192GB physical stack size
- The GPU count tooltip no longer claims 8 is the largest fully connected GPU domain in current hardware
```

- [ ] **Step 4: Verify the whole branch is green**

Run: `npx vitest run && npm run typecheck && npm run lint && npm run build`
Expected: all clean.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md CHANGELOG.md README.md
git commit -m "docs: record the per-GPU domain size model and the B200 correction"
```
