# Rack Data and Per-Precision Compute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GB200 NVL72, Vera Rubin NVL72 and HGX Rubin NVL8 with usable memory and dense FP8/FP4 compute. Make decode and prefill use the tensor core vLLM actually runs each weight format on.

**Architecture:**

- **Data:** stays generated: `scripts/fetch-gpus.ts` (GPUS array), then `npm run refresh:gpus`, then `src/data/gpus.json`.
- **New pure module:** `src/engines/compute.ts`. It owns the format-to-precision table (vLLM's kernel choice, spec D5) and the per-precision prefill efficiency (D9).
- **`performance.ts`:** splits linear FLOPs (at the format's precision) from attention FLOPs (always FP16), in both the decode compute ceiling and prefill.

**Tech Stack:** TypeScript strict, Zod, decimal.js, Vitest, Biome.

**Spec:** `docs/superpowers/specs/2026-11-10-rack-data-precision-compute-design.md` (read it first; D1-D9 are referenced below)

## Global Constraints

- Branch `feat/rack-precision-compute` (already created, spec committed). Run `npm ci` once before Task 1. `node_modules` is absent.
- Never hand-edit `src/data/gpus.json`. Edit `scripts/fetch-gpus.ts`, then run `npm run refresh:gpus`.
- Ids are never renamed. `nvidia-gb300-nvl72` keeps its id.
- All TFLOPS fields are **dense**. Every non-obvious figure gets a source comment, saying whether it is dense.
- `vram_gb` is usable memory, never physical (D1).
- Rubin `name`s contain `preliminary` (D2).
- Field presence = vLLM runs that precision natively on that card (D4). Workstation/consumer cards, GB10 and Apple get no `fp8_tflops`/`fp4_tflops`.
- Formats that compute at FP16 today (fp16, bf16, fp32, int8, int4, nf4, gptq, awq, nvfp6, gguf-*) must give today's results (assert `toBeCloseTo(x, 9)`, as the existing pins do; splitting one division into two can move the last digit), except on `nvidia-gb300-nvl72` (memory change).
- Biome: 2-space indent, single quotes, no semicolons, 100 columns, no `!` non-null assertions.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Verification commands: `npx vitest run`, `npm run typecheck`, `npm run lint`, `npm run build` (never `npm test`, which watches).

## Review Focus

1. **A GPU whose `interconnect` is `nvlink-6`** must resolve to `nvlink-6` in `resolveInterconnect`. Today an unknown value falls through to `'none'` (efficiency 0, so TP is broken silently). Pinned in Task 1.
2. **Custom GPUs (FP16 figure only) running an fp8/nvfp4 format** must give exactly today's numbers (fallback to FP16). Pinned in Task 3.
3. **AMD running `nvfp4`** must use FP16, while **AMD running `mxfp4`** uses FP4 (MI350-series only). Pinned in Task 3.
4. **A GPU with `fp8_tflops`/`fp4_tflops` but no `fp16_tflops`** would break the "no FLOPS means degraded estimate" guard. The data test forbids it. Pinned in Task 2.
5. **A shared link with GB300 NVL72 that fit at 288 GB** now restores at 278 GB and may read "does not fit". The fit must just flip, never throw. Pinned in Task 5 (fit test near the boundary).

---

### Task 1: Schema fields and the NVLink 6 interconnect

**Files:**

- Modify: `src/utils/schemas.ts` (GPU_INTERCONNECTS ~line 46; GPUSchema after `fp32_tflops` ~line 88)
- Modify: `src/engines/types.ts:128` (InterconnectType)
- Modify: `src/engines/constants.ts` (INTERCONNECT_SPECS ~line 232, INTERCONNECT_LABELS ~line 467)
- Modify: `src/engines/multi-gpu.ts:430` (resolveInterconnect)
- Test: `src/engines/multi-gpu.test.ts`, `src/utils/gpus.test.ts`

**Interfaces:**

- Produces:
  - `GPU.fp8_tflops?: number` and `GPU.fp4_tflops?: number` (dense, positive);
  - `'nvlink-6'` in `GPU_INTERCONNECTS` and `InterconnectType`;
  - `INTERCONNECT_SPECS['nvlink-6']`;
  - `INTERCONNECT_LABELS['nvlink-6']`.

- [ ] **Step 1: Write the failing tests**

In `multi-gpu.test.ts`:

```ts
it('resolves an NVLink 6 GPU to nvlink-6, not none', () => {
  const gpu = { ...someDatacenterGpuFixture, interconnect: 'nvlink-6' as const }
  expect(resolveInterconnect(gpu, 8)).toBe('nvlink-6')
})
it('prices NVLink 6 at 3000 GB/s bidirectional with efficiency below 1', () => {
  const s = INTERCONNECT_SPECS['nvlink-6']
  expect(s.bandwidthGBps).toBe(3000)
  expect(s.tpScalingEfficiency).toBe(0.98)
  expect(s.tpScalingEfficiency).toBeLessThan(1)
  expect(s.recommendedMaxTPDegree).toBe(8)
  expect(s.allreduceLatencyUs).toBe(11)
})
```
Build the fixture from a real row (`validateGPUs(gpusData).find(...)`) and spread it. Don't hand-write a literal (CLAUDE.md: test files are not typechecked).

In `gpus.test.ts`: `GPUSchema.parse({...validRow, fp8_tflops: 5000, fp4_tflops: 15000})` succeeds, and `fp8_tflops: 0` throws.

- [ ] **Step 2: Run them and confirm they fail**

Run: `npx vitest run src/engines/multi-gpu.test.ts src/utils/gpus.test.ts`
Expected: FAIL (`'none'` received; `INTERCONNECT_SPECS['nvlink-6']` undefined).

- [ ] **Step 3: Implement**

- `fp8_tflops` and `fp4_tflops`: `z.number().positive().optional()`, each with a one-line doc comment: "dense; present only where vLLM runs this precision natively (spec D4)".
- `INTERCONNECT_SPECS['nvlink-6']` has `bandwidthGBps: 3000` and `tpScalingEfficiency: 0.98`. Its comment says:
  - 216 TB/s / 72 per NVIDIA's Vera Rubin NVL72 page, the same convention as nvlink-5 (130 / 72);
  - the slope would give 1.007, capped at 0.98;
  - derived, not measured.
- `allreduceLatencyUs: NVLINK_ALLREDUCE_LATENCY_US`.
- Label: `'NVLink 6 — 3000 GB/s'`.
- `resolveInterconnect`: add `if (interconnect === 'nvlink-6') return 'nvlink-6'`.
- Do not add nvlink-6 to the `nvlink_bridge` enums (no bridge product).

- [ ] **Step 4: Run them and confirm they pass**

Run: `npx vitest run src/engines/multi-gpu.test.ts src/utils/gpus.test.ts && npm run typecheck`
Expected: PASS. Typecheck fails only if a `Record<InterconnectType, …>` is missing its key. Fix any such record by adding the nvlink-6 entry.

- [ ] **Step 5: Commit**

`git commit -m "feat(gpu-schema): dense fp8/fp4 TFLOPS fields and NVLink 6 interconnect"`

---

### Task 2: GPU data, the three new rows, GB300 usable memory, Grace/Vera links

**Files:**

- Modify: `scripts/fetch-gpus.ts` (GPUS array; NVIDIA datacenter block next to the GB300 rows)
- Regenerate: `src/data/gpus.json` (`npm run refresh:gpus`)
- Modify: `src/engines/kv-tier.ts:112` (`graceLinkGBps`)
- Test: `src/utils/gpus.test.ts`, `src/engines/kv-tier.test.ts`

**Interfaces:**

- Consumes: Task 1 fields and `'nvlink-6'`.
- Produces: rows `nvidia-gb200-nvl72`, `nvidia-vera-rubin-nvl72`, `nvidia-hgx-rubin-nvl8`. `graceLinkGBps('nvidia-gb200-nvl72') === 225`, `graceLinkGBps('nvidia-vera-rubin-nvl72') === 450`, and `graceLinkGBps('nvidia-hgx-rubin-nvl8') === null`.

- [ ] **Step 1: Write the failing tests** (`gpus.test.ts`)

```ts
it.each([
  // id, vram, GB/s, fp16, fp8, fp4, interconnect, max per node
  ['nvidia-gb200-nvl72', 186, 8000, 2500, 5000, 10000, 'nvlink-5', 72],
  ['nvidia-vera-rubin-nvl72', 278, 19200, 4000, 17500, 35000, 'nvlink-6', 72],
  ['nvidia-hgx-rubin-nvl8', 250, 16250, 4000, 17500, 35000, 'nvlink-6', 8],
])('lists %s at usable memory and dense compute', (id, vram, bw, f16, f8, f4, ic, max) => { … })
it('lists GB300 NVL72 at usable memory (20 TB / 72 ≈ 278, InferenceX 278), not 288', () => {
  expect(nvl72?.vram_gb).toBe(278)
  expect(nvl72?.fp8_tflops).toBe(5000)
  expect(nvl72?.fp4_tflops).toBe(15000)
})
it('tags every Rubin row preliminary', …) // name includes 'preliminary'
it('lists B200 and HGX B300 dense FP8/FP4 from the HGX page', …)
// b200: 4500 / 9000; nvidia-gb300-288gb: 4500 / 13500
it('never stores fp8/fp4 without fp16, and orders fp16 <= fp8 <= fp4', …)
it('gives no fp8/fp4 to workstation, consumer, GB10 or Apple rows', …)
// explicit id list (not tier: GB300 Desktop is tier 'consumer' yet gets fp8/fp4):
// l40s, rtx-6000-ada, rtx-pro-6000-server, rtx-5090, rtx-4090, rtx-3090, gb10, every apple-*
```

Replace the existing `'stores dense FP16 for every GPU…'` test with this. Every GPU's `fp16_tflops` stays below 2600, except an allow-list `{ 'nvidia-vera-rubin-nvl72': 4000, 'nvidia-hgx-rubin-nvl8': 4000 }`, whose comment says NVIDIA's Rubin page footnotes FP16 as dense.

In `kv-tier.test.ts`: `graceLinkGBps` returns 225 for gb200-nvl72, 450 for vera-rubin-nvl72 and null for hgx-rubin-nvl8.

- [ ] **Step 2: Run them and confirm they fail**

Run: `npx vitest run src/utils/gpus.test.ts src/engines/kv-tier.test.ts`
Expected: FAIL (rows missing, GB300 at 288).

- [ ] **Step 3: Add the rows and fields in `scripts/fetch-gpus.ts`**

New rows use the values in spec D3. The other fields:

- `manufacturer: 'nvidia'`, `tier: 'datacenter'`, `memory_type` `'HBM3e'` (GB200) or `'HBM4'` (Rubin);
- `spec_url`: the NVIDIA rack page; the HGX page for NVL8;
- `fp32_tflops` and `tdp_watts`: from the same pages. Omit a field the page doesn't give; don't guess;
- `bus_width` copied from the GB300 row for GB200. For Rubin, use 0 unless a source gives it.

HGX Rubin NVL8: one rule for every figure, namely NVIDIA HGX page system totals / 8: 2 TB to 250 GB, 130 TB/s to 16250 GB/s, NVLink 24 TB/s to 3000 GB/s. The page's per-GPU 288 GB / 22 TB/s / 3.6 TB/s are higher and inconsistent with its own totals; the lower figure wins (D1 "safe version"). Comment says so.

GB300 NVL72: `vram_gb: 278`, with a comment citing NVIDIA's 20 TB / 72 and InferenceX 278. Look for one more source (an nvidia-smi or vLLM log). If sources disagree, take the lower value, never above 278.

`fp8_tflops` / `fp4_tflops` for the existing rows follow spec D4:

- B200 4500 / 9000;
- HGX B300 4500 / 13500;
- GB300 NVL72 5000 / 15000;
- GB300 Desktop: source it (DGX Station page, FP4 is footnoted sparse);
- H100 SXM, H100 PCIe, H200, H200 NVL: FP8 only;
- MI300X, MI325X: FP8 only;
- MI350X, MI355X: FP8 and FP4 (MXFP4).

Source each of these with one vendor page plus one second source, as dense figures. If no second source turns up, leave the field absent and note it in the commit message.

- [ ] **Step 4: Regenerate and add the Grace/Vera links**

Run `npm run refresh:gpus`, then add the two `graceLinkGBps` branches to `kv-tier.ts`. The comments say:

- GB200: 2 GPUs per Grace, 900 GB/s bidirectional C2C, the same as GB300;
- Vera Rubin: 1.8 TB/s C2C per superchip of 2 GPUs (NVIDIA Vera Rubin NVL72 page), halved for one direction, then shared by 2, giving 450.

Config rule R3 keys on `graceLinkGBps`, so it needs no edit.

**Verify the host side before trusting the link figure.** Each value is min(link share, host memory bandwidth share), as the Desktop row already does (396). GB300/GB200 NVL72: NVIDIA's rack page gives 14 TB/s LPDDR5X / 36 Grace = 389 GB/s per Grace, which is 194 per GPU, below 225. If that holds after one second source, both NVL72 rows become 194. That changes the existing GB300 value too, so update `kv-tier.test.ts`/`offloading.test.ts` pins and put it in the CHANGELOG (Task 5). Vera: use 450 unless a Vera LPDDR5X bandwidth figure gives a lower per-GPU share.

- [ ] **Step 5: Run them and confirm they pass**

Run: `npx vitest run src/utils/gpus.test.ts src/engines/kv-tier.test.ts`
Expected: PASS. Then run `npx vitest run` and **list** every other failing test without fixing it (Task 5 owns those).

- [ ] **Step 6: Commit**

`git commit -m "feat(gpus): GB200 NVL72, Vera Rubin NVL72, HGX Rubin NVL8; GB300 NVL72 usable 278 GB; dense FP8/FP4"`

---

### Task 3: `compute.ts`, format to tensor core, and per-precision prefill efficiency

**Files:**

- Create: `src/engines/compute.ts`
- Create: `src/engines/compute.test.ts`
- Modify: `src/engines/constants.ts` (next to `PREFILL_MFU` ~line 485)
- Modify: `src/engines/index.ts` (export the new module, matching the existing pattern)

**Interfaces:**

- Consumes: `GPU` (with Task 1 fields), `QuantizationFormat` from `./types`.
- Produces:

  ```ts
  export type ComputePrecision = 'fp4' | 'fp8' | 'fp16'
  export interface ComputeRate { precision: ComputePrecision; tflops: number }
  /** Rate the linear (GEMM) FLOPs run at for this format on this GPU (spec D5). tflops 0 = no FLOPS data */
  export function linearComputeRate(gpu: GPU, format: QuantizationFormat): ComputeRate
  /** Rate attention FLOPs run at: always the 16-bit figure (fp16 ?? fp32 ?? 0) */
  export function attentionComputeRate(gpu: GPU): ComputeRate
  ```

  plus `PREFILL_MFU_BY_PRECISION: Record<ComputePrecision, Decimal>` in `constants.ts`. Keep `PREFILL_MFU` as the `fp16` value, so existing imports still work.

- [ ] **Step 1: Research the prefill efficiency (D9) before any code**

Look for a measurement that **isolates prefill**: input (prompt) tokens/s, or time to first token at a known prompt length and batch, at FP8 and at FP4, on Hopper or Blackwell. MLPerf offline throughput does **not** qualify, because it is mostly decode. Candidates: SemiAnalysis InferenceX, NVIDIA/vLLM/SGLang prefill benchmarks, the TensorRT-LLM perf overview (context-phase numbers). Derive efficiency = (2 × active params × prompt tokens/s, plus attention FLOPs) / (peak TFLOPS at that precision × 1e12).

Record the result in the commit message and in ADR 0008 (Task 5):

- **Outcome A (sourced):** `fp8` and `fp4` get their sourced values, each with a source comment.
- **Outcome B (no prefill-isolating FP4 source):** where the format *would* get FP4 (card has `fp4_tflops`, and is NVIDIA for nvfp4), it gets the card's `fp8_tflops` instead (precision `'fp8'`). Everywhere else the normal fallback to FP16 applies: H100 + nvfp4 stays FP16, and MI355X + nvfp4 stays FP16. The comment cites spec D9. `fp8` without a source keeps 0.45.

Steps 2-5 below are written for Outcome A. Under Outcome B, the FP4 rows of the test table expect `{precision:'fp8', tflops: fp8_tflops}`.

- [ ] **Step 2: Write the failing tests** (`compute.test.ts`, fixtures from real rows)

```ts
it.each([
  // gpu id, format, expected precision, expected tflops
  ['nvidia-b200-192gb', 'fp8', 'fp8', 4500],
  ['nvidia-b200-192gb', 'nvfp4', 'fp4', 9000],
  ['nvidia-b200-192gb', 'mxfp4', 'fp4', 9000],
  ['nvidia-b200-192gb', 'bf16', 'fp16', 2250],
  ['nvidia-b200-192gb', 'awq', 'fp16', 2250],
  ['nvidia-b200-192gb', 'gguf-q4_k_m', 'fp16', 2250],
  ['nvidia-b200-192gb', 'nvfp6', 'fp16', 2250],
  ['nvidia-b200-192gb', 'int8', 'fp16', 2250],
  ['nvidia-b200-192gb', 'fp32', 'fp16', 2250],
  ['nvidia-h100-80gb-sxm', 'fp8', 'fp8', /* sourced H100 dense FP8 */],
  ['nvidia-h100-80gb-sxm', 'nvfp4', 'fp16', 989],   // Marlin W4A16, not FP8
  ['amd-mi355x', 'mxfp4', 'fp4', /* sourced */],
  ['amd-mi355x', 'nvfp4', 'fp16', 2516],            // AMD runs MXFP4, not NVFP4
  ['nvidia-rtx-pro-6000-server', 'nvfp4', 'fp16', 240],
])('%s + %s computes at %s', …)
it('a custom GPU with only fp16 falls back to fp16 for every format', …) // createCustomGPU({ name:'x', vram_gb: 80, fp16_tflops: 500 })
it('a GPU with only fp32 returns fp32 as the 16-bit rate (today fp16 ?? fp32)', …)
it('returns tflops 0 when the GPU has no FLOPS data', …)
it('attention always uses the 16-bit figure', …) // B200 + nvfp4 -> 2250
it('PREFILL_MFU stays the fp16 entry of PREFILL_MFU_BY_PRECISION', …)
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `npx vitest run src/engines/compute.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 4: Implement `compute.ts`**

- A `FORMAT_PRECISION: Record<QuantizationFormat, ComputePrecision>` table, so a new format fails to typecheck until someone classifies it. It maps `fp8` to `fp8`, `nvfp4` and `mxfp4` to `fp4`, and everything else to `fp16`.
- Fallback order: fp4 to fp16, fp8 to fp16, never fp4 to fp8 (Marlin computes in 16-bit). `nvfp4` uses `fp4` only when `gpu.manufacturer === 'nvidia'`.
- The 16-bit rate is `gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0`, exactly today's expression.
- Module doc comment: the vLLM sources from spec D5. Namely, `CompressedTensorsW4A4Mxfp4` uses W4A4 on SM100+, otherwise W4A16 Marlin; `MarlinNvFp4LinearKernel` and `TorchNvFp4LinearKernel` (sm_10x); FP8 W8A8 needs compute capability ≥ 8.9. Also state that this table is vLLM's (the user's decision, 2026-11-10).

- [ ] **Step 5: Run them and confirm they pass**

Run: `npx vitest run src/engines/compute.test.ts && npm run typecheck && npm run lint`
Expected: PASS.

- [ ] **Step 6: Commit**

`git commit -m "feat(engines): compute rate per weight format (vLLM kernels) and per-precision prefill MFU"`

---

### Task 4: Wire the compute rates into decode and prefill

**Files:**

- Modify: `src/engines/performance.ts:240-262` (decode compute) and `:338-386` (prefill)
- Test: `src/engines/performance.test.ts`

**Interfaces:**

- Consumes: `linearComputeRate`, `attentionComputeRate`, `PREFILL_MFU_BY_PRECISION` (Task 3).
- Produces: no new exports. `PerformanceEstimate` keeps its shape.

- [ ] **Step 1: Write the failing tests**

```ts
it('FP8 linear prefill on B200 is faster than BF16 by fp8 peak x MFU over fp16 peak x MFU', () => {
  // same model, sequenceLength chosen so linear dominates (e.g. 1024, Llama 3.1 70B)
  // expected ratio = (4500 * MFU.fp8) / (2250 * MFU.fp16) on the linear term;
  // assert the total prefillSeconds against the two-term formula computed in the test
})
it('nvfp4 on H100 gives exactly the bf16 prefill and decode (no native FP4)', …)
it('a custom FP16-only GPU gives identical results for fp8 and the pre-change formula', …)
// pin today's number: the existing 0.656594895267678 s case stays exact for bf16
it('attention FLOPs stay on FP16 at long context: 128k nvfp4 prefill on B200 is attention-bound and slower than linear-only scaling predicts', …)
it('decode compute ceiling uses the format rate: large-batch fp8 decode on B200 is compute-bound at fp8 peak', …)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `npx vitest run src/engines/performance.test.ts`
Expected: the new FP8/NVFP4 tests fail. The existing bf16 pins pass.

- [ ] **Step 3: Implement**

Replace `decodeGpuTFLOPS` with `linear = linearComputeRate(gpu, quantization)` and `attention = attentionComputeRate(gpu)`.

- **Decode:** split `flopsPerToken` into its linear part (`activeParams × 2e9`) and attention part (`4 × layers × context × hidden`). `computeSeconds = (linear FLOPs / linear.tflops + attention FLOPs / attention.tflops) × batch / GPUs`. No MFU in decode, as today.
- **Prefill:** `prefillSeconds = linearFLOPs / (linear.tflops × 1e12 × MFU[linear.precision] × multi-GPU factor) + attentionFLOPs / (attention.tflops × 1e12 × MFU.fp16 × the same factor) + hopSeconds`.
- **Guard:** `attention.tflops > 0` replaces `decodeGpuTFLOPS > 0`, so the degraded path triggers exactly as today. `linear.tflops` is always > 0 whenever attention's is, because it falls back to the same figure.
- Update the "same FLOPS figure and same guard" comment so it names `compute.ts` as the one source.

- [ ] **Step 4: Run them and confirm they pass**

Run: `npx vitest run src/engines/performance.test.ts`
Expected: PASS, with every pre-existing bf16/fp16/awq/gguf pin unchanged.

- [ ] **Step 5: Commit**

`git commit -m "fix(performance): prefill and decode compute at the tensor core each format runs on"`

---

### Task 5: Numbers that move on purpose, docs, ADR 0008

**Files:**

- Modify: every test Task 2 Step 5 or Task 4 listed as failing, mostly fp8/nvfp4/mxfp4 pins and GB300 NVL72 fit/sessions in `performance.test.ts`, `multi-gpu.test.ts`, `inference.integration.test.ts`, `config-rules.test.ts`, `kv-tier.test.ts` and `uiStore.test.ts`.
- Create: `docs/adr/0008-per-precision-compute-and-usable-rack-memory.md`, plus a row in `docs/adr/README.md`.
- Modify: `CLAUDE.md` (pitfalls #6 and #8), `CHANGELOG.md` (`## [Unreleased]`), `README.md` (NVIDIA Datacenter GPU row), `docs/gpu-specs-reference.md`, and `docs/vram-calculation-pitfalls.md` if it repeats pitfall #6.
- Modify: memory file `MEMORY.md` "Current counts" line (29 to 32 GPUs).

- [ ] **Step 1: Re-pin the moved numbers**

For each failing test, recompute the expected value from the formula in the test's own comment. Update both the comment and the value, and give the reason in one line, e.g. `// 2026-11-10: fp8 linear term now at FP8 peak (ADR 0008)`. A test that fails for any other reason is a bug: stop and investigate (superpowers:systematic-debugging).

- [ ] **Step 2: Pin the shared-link boundary case (Review Focus 5)**

In `inference.integration.test.ts`, choose a model, format and context that fit on GB300 NVL72 at 288 GB but not at 278 GB × `GPU_MEMORY_UTILIZATION`. Assert the result reports "does not fit" and doesn't throw.

- [ ] **Step 3: Write the docs**

- **ADR 0008:** context (the five problems from the spec's "What is wrong today"), decisions D1-D9 in one line each, the D9 outcome (A or B), and sources (the NVIDIA rack pages, the HGX page, InferenceX, the vLLM classes named in D5).
- **CLAUDE.md pitfall #6:**
  - FP16/FP8/FP4 are separate dense fields;
  - "dense FP16 = dense FP8 / 2" holds through Blackwell, not Rubin (4 / 17.5);
  - the guard test now allow-lists Rubin;
  - field presence means native vLLM support.
- **CLAUDE.md pitfall #8:** add "GB300 NVL72 is 278 GB usable (20 TB / 72), not 288".
- **CHANGELOG** under `### Breaking changes`:
  - GB300 NVL72 is now 278 GB;
  - fp8/nvfp4/mxfp4 prefill and compute-bound decode are faster on Hopper/Blackwell/AMD, and by how much for one example;
  - under `### Added`, the three racks and NVLink 6.
- **README:** add GB200 NVL72, Vera Rubin NVL72 (preliminary) and HGX Rubin NVL8 to the NVIDIA Datacenter row.

- [ ] **Step 4: Full verification**

Run: `npx vitest run && npm run typecheck && npm run lint && npm run build`
Expected: all green. Paste the pass counts in the commit body.

- [ ] **Step 5: Commit**

`git commit -m "docs: ADR 0008, pitfalls #6/#8, CHANGELOG, README for rack data and per-precision compute"`
