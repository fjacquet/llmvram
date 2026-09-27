# MoE Weight Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expert-parallel memory and MoE decode charge a MoE checkpoint's base and routed experts at their measured rates instead of one averaged bytes/param.

**Architecture:** Each quantized safetensors weight ref of a MoE model gains `high_precision: {params_b, gib}` (tensors stored as >=16-bit floats), read from the Hugging Face dtype summary. `moeWeightSplit` in `inference.ts` turns it into `{baseGiB, routedGiB, measured}` by filling the base from the wide tensors first; without it, it falls back to the parameter-fraction split. EP memory and decode consume it.

**Tech Stack:** TypeScript 7 strict, Zod 4, decimal.js, Vitest 5, tsx scripts.

**Spec:** `docs/superpowers/specs/2026-09-27-moe-weight-split-design.md`

## Global Constraints

- Wide dtypes are exactly `BF16`, `F16`, `F32`, `F64` with widths 2, 2, 4, 8. Every other dtype is quantized storage.
- `high_precision` only for MoE models (`architecture: 'moe'`), only for refs of formats other than `fp32`/`fp16`/`bf16` and not `gguf-*`, only when the summary has at least one wide and one non-wide dtype.
- `params_b` rounded to 3 decimals; `gib` rounded to 2 decimals (same as `totalGiB`).
- `baseGiB + routedGiB` equals `calculateModelWeightVRAM(num_parameters_billion, format, model)` exactly.
- Fallback without `high_precision`: `baseGiB = total x baseB / N`, `measured: false`.
- Totals, TP/PP memory, KV, FLOPs, prefill, dense models: unchanged.
- The script never writes `models.json`; data merges are reviewed by hand.
- Biome style (single quotes, no semicolons, 100 cols); tests are not typechecked: build fixtures from `validateModels(modelsData)`, never hand-written model literals, except in `tests/model-audit.test.ts` which already uses one.
- Commits end with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z`

## Review Focus

1. Offloading plus expert parallelism: `singleGPU.modelWeights` may be the on-device share after offloading; the EP split must scale that figure by the base share, not replace it with the full-model base (Task 5 test).
2. fp16 selected on a model whose ref is bf16: the twin rule must reach `high_precision` too, but bf16 refs never carry it (16-bit is uniform), so fp16 falls back (Task 4 test).
3. A ref whose wide params exceed the model total or whose wide GiB exceeds the ref GiB (bad paste): integrity test rejects it (Task 3).
4. `H >= baseB` and `H < baseB` both covered, and the degenerate `N - H <= 0` never divides by zero (Task 4 test).
5. Batch sizes 1 and large on EP decode: routed term divides by GPUs; base read in full (Task 6 test).

---

### Task 1: Pure `highPrecision` extraction in the auditor

**Files:**
- Modify: `scripts/model-audit.ts` (append exports)
- Test: `tests/model-audit.test.ts`

**Interfaces:**
- Produces: `export const WIDE_DTYPE_BYTES: Record<string, number>`;
  `export function highPrecision(parameters: Record<string, number>, format: QuantizationFormat): { params_b: number; gib: number } | null`;
  `export function highPrecisionDrift(format: QuantizationFormat, curated: { params_b: number; gib: number }, measured: { params_b: number; gib: number } | null): string | null`

- [ ] **Step 1: Write the failing tests** (append to `tests/model-audit.test.ts`, add `highPrecision, highPrecisionDrift` to the import list)

```ts
describe('highPrecision', () => {
  it('sums >=16-bit float tensors of a native MXFP4 checkpoint (moonshotai/Kimi-K3)', () => {
    const p = { F32: 11122432, BF16: 57179884544, U8: 2722740830208 }
    expect(highPrecision(p, 'mxfp4')).toEqual({ params_b: 57.191, gib: 106.55 })
  })
  it('counts FP8 as quantized storage (deepseek-ai/DeepSeek-R1)', () => {
    const p = { BF16: 3918786560, F8_E4M3: 680571043840, F32: 15104 }
    expect(highPrecision(p, 'fp8')).toEqual({ params_b: 3.919, gib: 7.3 })
  })
  it('treats I32-packed int4 as quantized storage (Qwen GPTQ)', () => {
    const p = { I32: 233800000000, F16: 1290000000 }
    expect(highPrecision(p, 'gptq')).toEqual({ params_b: 1.29, gib: 2.4 })
  })
  it('treats NVFP4 F8 block scales as quantized storage', () => {
    const p = { U8: 116900000000, F8_E4M3: 14610000000, BF16: 1290000000 }
    expect(highPrecision(p, 'nvfp4')).toEqual({ params_b: 1.29, gib: 2.4 })
  })
  it('returns null for 16-bit formats, GGUF, and single-kind summaries', () => {
    expect(highPrecision({ BF16: 235e9 }, 'bf16')).toBeNull()
    expect(highPrecision({ BF16: 1e9, U8: 9e9 }, 'fp16')).toBeNull()
    expect(highPrecision({ BF16: 1e9, U8: 9e9 }, 'gguf-q4_k_m')).toBeNull()
    expect(highPrecision({ F8_E4M3: 9e9 }, 'fp8')).toBeNull()
    expect(highPrecision({ BF16: 9e9 }, 'fp8')).toBeNull()
  })
})

describe('highPrecisionDrift', () => {
  const curated = { params_b: 57.191, gib: 106.55 }
  it('passes within 1%', () => {
    expect(highPrecisionDrift('mxfp4', curated, { params_b: 57.2, gib: 106.9 })).toBeNull()
  })
  it('reports drift beyond 1% and a missing summary', () => {
    expect(highPrecisionDrift('mxfp4', curated, { params_b: 57.2, gib: 110 })).toBe(
      'mxfp4.high_precision: curated 106.55 GiB, measured 110 GiB',
    )
    expect(highPrecisionDrift('mxfp4', curated, null)).toBe(
      'mxfp4.high_precision: no dtype summary',
    )
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/model-audit.test.ts`
Expected: FAIL, `highPrecision is not a function` (import undefined).

- [ ] **Step 3: Implement** (append to `scripts/model-audit.ts`)

```ts
/** Floating-point dtypes of 16 bits or more. Everything else (F8_*, U8, I8, I32) is
 * quantized storage: packed int4 is stored as I32 and NVFP4 block scales as F8_E4M3,
 * so width alone cannot identify the base (spec 2026-09-27-moe-weight-split). */
export const WIDE_DTYPE_BYTES: Record<string, number> = { BF16: 2, F16: 2, F32: 4, F64: 8 }

const UNIFORM_FORMATS = new Set<QuantizationFormat>(['fp32', 'fp16', 'bf16'])

/** Tensors a quantized checkpoint keeps as >=16-bit floats, from the HF dtype summary. */
export function highPrecision(
  parameters: Record<string, number>,
  format: QuantizationFormat,
): { params_b: number; gib: number } | null {
  if (UNIFORM_FORMATS.has(format) || format.startsWith('gguf-')) return null
  const entries = Object.entries(parameters)
  const wide = entries.filter(([dtype]) => dtype in WIDE_DTYPE_BYTES)
  if (wide.length === 0 || wide.length === entries.length) return null
  const count = wide.reduce((s, [, n]) => s + n, 0)
  const bytes = wide.reduce((s, [dtype, n]) => s + n * (WIDE_DTYPE_BYTES[dtype] ?? 0), 0)
  return {
    params_b: Math.round((count / 1e9) * 1000) / 1000,
    gib: Math.round((bytes / 1024 ** 3) * 100) / 100,
  }
}

export function highPrecisionDrift(
  format: QuantizationFormat,
  curated: { params_b: number; gib: number },
  measured: { params_b: number; gib: number } | null,
): string | null {
  if (!measured) return `${format}.high_precision: no dtype summary`
  if (Math.abs(measured.gib - curated.gib) <= curated.gib * 0.01) return null
  return `${format}.high_precision: curated ${curated.gib} GiB, measured ${measured.gib} GiB`
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/model-audit.test.ts`
Expected: PASS (all, including existing).

- [ ] **Step 5: Commit**

```bash
rtk proxy git add scripts/model-audit.ts tests/model-audit.test.ts
rtk proxy git commit -m "feat: extract high-precision tensors from the HF dtype summary"
```

---

### Task 2: Schema field, `weightRef` accessor, CLI wiring

**Files:**
- Modify: `src/utils/schemas.ts:163-168`
- Modify: `src/engines/quantization.ts` (export accessor)
- Modify: `scripts/fetch-models.ts` (measureRefs, audit, new `--split`)
- Test: `src/engines/quantization.test.ts`, `src/utils/schemas.test.ts`

**Interfaces:**
- Consumes: `highPrecision`, `highPrecisionDrift` (Task 1)
- Produces: `export function weightRef(format: QuantizationFormat, model?: Model): { repo: string; gib: number; high_precision?: { params_b: number; gib: number } } | undefined` (the existing private `resolveRef`, renamed and exported; `effectiveBytesPerParameter` and `weightSource` call it)

- [ ] **Step 1: Failing tests**

In `src/utils/schemas.test.ts` (inside an existing or new `describe('ModelSchema weight_refs')`):

```ts
it('accepts an optional high_precision on a weight ref and rejects non-positive values', () => {
  const base = validateModels(modelsData)[0]
  const ok = { ...base, weight_refs: { fp8: { repo: 'a/b', gib: 10, high_precision: { params_b: 1, gib: 2 } } } }
  expect(() => validateModels([ok])).not.toThrow()
  const bad = { ...base, weight_refs: { fp8: { repo: 'a/b', gib: 10, high_precision: { params_b: 0, gib: 2 } } } }
  expect(() => validateModels([bad])).toThrow()
})
```

(If `schemas.test.ts` lacks `modelsData`/`validateModels` imports, add `import modelsData from '@data/models.json'` and `validateModels` from `./schemas`.)

In `src/engines/quantization.test.ts`:

```ts
it('weightRef resolves the fp16/bf16 twin and exposes high_precision', () => {
  const m = validateModels(modelsData).find((x) => x.id === 'moonshotai-kimi-k3')
  if (!m) throw new Error('fixture')
  expect(weightRef('mxfp4', m)?.repo).toBe('moonshotai/Kimi-K3')
  expect(weightRef('int8', m)).toBeUndefined()
})
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/utils/schemas.test.ts src/engines/quantization.test.ts`
Expected: FAIL (`weightRef` not exported; `high_precision` stripped or bad value accepted).

- [ ] **Step 3: Implement**

`src/utils/schemas.ts` weight_refs value object:

```ts
      z.object({
        repo: z.string().min(1),
        gib: z.number().positive(),
        high_precision: z
          .object({ params_b: z.number().positive(), gib: z.number().positive() })
          .optional(),
      }),
```

`src/engines/quantization.ts`: rename `function resolveRef` to `export function weightRef`, keep its body and JSDoc, update its two call sites.

`scripts/fetch-models.ts`:
- import `highPrecision, highPrecisionDrift` from `./model-audit`.
- `measureRefs(repo)` takes a second arg `moe: boolean`; after each ref is set (native and picked), when `moe`, fetch `fetchSafetensors(ref.repo)` (reuse `st` for the native repo) and set `high_precision` when `highPrecision(st.parameters, format)` is non-null. Callers: `measure()` passes `m.architecture === 'moe'`; `draft()` passes `!!f.num_experts`.
- `audit()`: inside the weight_refs loop, when `ref.high_precision`, `const st2 = await fetchSafetensors(ref.repo)` and push `highPrecisionDrift(format, ref.high_precision, st2 ? highPrecision(st2.parameters, format as QuantizationFormat) : null)` when non-null (prefix `weight_refs.`).
- New mode `--split`: for every MoE model, for every existing non-GGUF ref, fetch the summary and collect `{ [id]: { [format]: high_precision } }` for non-null results; print the JSON. Wire: `else if (args.includes('--split')) await split()`. Update the header comment line list with `--split`.

- [ ] **Step 4: Run to verify pass, then smoke the CLI**

Run: `npx vitest run src/utils/schemas.test.ts src/engines/quantization.test.ts tests/model-audit.test.ts && npm run typecheck`
Expected: PASS, typecheck clean.
Run: `./node_modules/.bin/tsx scripts/fetch-models.ts --split > /tmp/split.json; head -c 600 /tmp/split.json` (use the scratchpad dir instead of /tmp if available)
Expected: JSON containing `"moonshotai-kimi-k3": { "mxfp4": { "params_b": 57.191, "gib": 106.55 }, ...`.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add src/utils/schemas.ts src/utils/schemas.test.ts src/engines/quantization.ts src/engines/quantization.test.ts scripts/fetch-models.ts
rtk proxy git commit -m "feat: weight refs carry high_precision; refresh:models measures and audits it"
```

---

### Task 3: Data for all MoE refs, integrity tests

**Files:**
- Modify: `src/data/models.json`
- Test: `src/utils/models.test.ts`

**Interfaces:**
- Consumes: `--split` output (Task 2)

- [ ] **Step 1: Failing tests** (append inside the weight_refs describe block of `src/utils/models.test.ts`)

```ts
it('high_precision is consistent with its ref and model', () => {
  for (const m of models) {
    for (const [format, ref] of Object.entries(m.weight_refs ?? {})) {
      const hp = ref?.high_precision
      if (!hp || !ref) continue
      expect(m.architecture, `${m.id} ${format}`).toBe('moe')
      expect(['fp32', 'fp16', 'bf16'].includes(format) || format.startsWith('gguf-')).toBe(false)
      expect(hp.gib, `${m.id} ${format}`).toBeLessThanOrEqual(ref.gib)
      expect(hp.params_b, `${m.id} ${format}`).toBeLessThanOrEqual(m.num_parameters_billion * 1.02)
    }
  }
})

it('pins measured high_precision anchors (HF dtype summaries, 2026-09-27)', () => {
  const hp = (id: string, f: string) =>
    models.find((m) => m.id === id)?.weight_refs?.[f as keyof NonNullable<Model['weight_refs']>]
      ?.high_precision
  expect(hp('moonshotai-kimi-k3', 'mxfp4')).toEqual({ params_b: 57.191, gib: 106.55 })
  expect(hp('deepseek-r1', 'fp8')).toEqual({ params_b: 3.919, gib: 7.3 })
  expect(hp('qwen-qwen3-235b-a22b', 'bf16')).toBeUndefined()
})
```

(`models` and `Model` are already in scope in that file; if `Model` is not imported, import the type from `@utils/schemas`.)

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/utils/models.test.ts`
Expected: FAIL on the anchors (`undefined` vs the objects).

- [ ] **Step 3: Merge the data**

Run `--split` (Task 2) to a file in the scratchpad, then merge with a one-off script kept in the scratchpad (not committed):

```js
// merge-split.mjs <models.json> <split.json>
import { readFileSync, writeFileSync } from 'node:fs'
const [mp, sp] = process.argv.slice(2)
const models = JSON.parse(readFileSync(mp, 'utf8'))
const split = JSON.parse(readFileSync(sp, 'utf8'))
for (const m of models) for (const [f, hp] of Object.entries(split[m.id] ?? {})) {
  if (m.weight_refs?.[f]) m.weight_refs[f].high_precision = hp
}
writeFileSync(mp, `${JSON.stringify(models, null, 2)}\n`)
```

Run: `node <scratchpad>/merge-split.mjs src/data/models.json <scratchpad>/split.json && ./node_modules/.bin/biome format --write src/data/models.json`
Review `git diff --stat src/data/models.json` and spot-check 3 entries against the probe table in the spec. Order of models is unchanged (no re-sort needed).

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/utils/models.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add src/data/models.json src/utils/models.test.ts
rtk proxy git commit -m "data: high_precision for every quantized MoE safetensors ref"
```

---

### Task 4: `moeWeightSplit` and `routedTouchedFraction`

**Files:**
- Modify: `src/engines/inference.ts` (next to `splitMoEParams`, rewrite `calculateMoEBatchedParams`)
- Test: `src/engines/inference.test.ts`

**Interfaces:**
- Consumes: `weightRef` (Task 2), `splitMoEParams`, `calculateModelWeightVRAM`
- Produces:
  `export function routedTouchedFraction(model: Model, batchSize: number): number | null` (null when not splittable; `k/E` at batch <= 1; `1 - (1 - k/E)^B` above)
  `export function moeWeightSplit(model: Model, format: QuantizationFormat): { baseGiB: Decimal; routedGiB: Decimal; measured: boolean } | null`

- [ ] **Step 1: Failing tests** (append to `src/engines/inference.test.ts`; import `moeWeightSplit, routedTouchedFraction` and `modelsData`/`validateModels` if missing)

```ts
describe('moeWeightSplit', () => {
  const models = validateModels(modelsData)
  const get = (id: string) => {
    const m = models.find((x) => x.id === id)
    if (!m) throw new Error(id)
    return m
  }

  it('Kimi K3 mxfp4: wide tensors cover the base, charged at the BF16 rate', () => {
    const s = moeWeightSplit(get('moonshotai-kimi-k3'), 'mxfp4')
    expect(s?.measured).toBe(true)
    expect(s?.baseGiB.toNumber()).toBeCloseTo(103.11, 1) // 106.55 x 55.347 / 57.191
    expect(s?.routedGiB.toNumber()).toBeCloseTo(1350.63, 1)
  })

  it('DeepSeek R1 fp8: wide tensors fall short, the rest of the base at the FP8 rate', () => {
    const s = moeWeightSplit(get('deepseek-r1'), 'fp8')
    // 7.30 + (16.548 - 3.919) x (641.3 - 7.30) / (671 - 3.919)
    expect(s?.baseGiB.toNumber()).toBeCloseTo(19.3, 1)
  })

  it('parts always sum to the weight total', () => {
    for (const m of models.filter((x) => x.architecture === 'moe')) {
      for (const f of ['mxfp4', 'nvfp4', 'fp8', 'int4', 'awq', 'fp16'] as const) {
        const s = moeWeightSplit(m, f)
        if (!s) continue
        const total = calculateModelWeightVRAM(m.num_parameters_billion, f, m)
        expect(s.baseGiB.add(s.routedGiB).toNumber()).toBeCloseTo(total.toNumber(), 6)
      }
    }
  })

  it('falls back to the parameter-fraction split without high_precision (fp16 twin of bf16)', () => {
    const m = get('qwen-qwen3-235b-a22b')
    const s = moeWeightSplit(m, 'fp16')
    const split = splitMoEParams(m)
    const total = calculateModelWeightVRAM(m.num_parameters_billion, 'fp16', m)
    expect(s?.measured).toBe(false)
    expect(s?.baseGiB.toNumber()).toBeCloseTo(
      total.mul(split?.baseB ?? 0).div(m.num_parameters_billion).toNumber(),
      6,
    )
  })

  it('returns null for dense models', () => {
    expect(moeWeightSplit(get('meta-llama-llama-3.1-70b'), 'fp16')).toBeNull()
  })

  it('never divides by zero when wide params reach the total', () => {
    const m = get('moonshotai-kimi-k3')
    const ref = m.weight_refs?.mxfp4
    if (!ref) throw new Error('fixture')
    const odd = {
      ...m,
      weight_refs: { mxfp4: { ...ref, high_precision: { params_b: m.num_parameters_billion, gib: ref.gib } } },
    }
    const s = moeWeightSplit(odd, 'mxfp4')
    expect(Number.isFinite(s?.baseGiB.toNumber())).toBe(true)
  })
})

describe('routedTouchedFraction', () => {
  it('is k/E at batch 1 and grows toward 1', () => {
    const m = validateModels(modelsData).find((x) => x.id === 'moonshotai-kimi-k3')
    if (!m) throw new Error('fixture')
    expect(routedTouchedFraction(m, 1)).toBeCloseTo(16 / 896, 10)
    expect(routedTouchedFraction(m, 64)).toBeCloseTo(1 - (1 - 16 / 896) ** 64, 10)
    // calculateMoEBatchedParams is base + routed x fraction
    const split = splitMoEParams(m)
    expect(calculateMoEBatchedParams(m, 64)).toBeCloseTo(
      (split?.baseB ?? 0) + (split?.routedB ?? 0) * (1 - (1 - 16 / 896) ** 64),
      6,
    )
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/engines/inference.test.ts`
Expected: FAIL (`moeWeightSplit is not a function`).

- [ ] **Step 3: Implement** (in `src/engines/inference.ts`; import `weightRef` from `./quantization`, `QuantizationFormat` type)

```ts
/** Share of routed experts a decode step reads: k/E at batch 1, 1 - (1 - k/E)^B above. */
export function routedTouchedFraction(model: Model, batchSize: number): number | null {
  const split = splitMoEParams(model)
  if (!split) return null
  const k = new Decimal(split.expertsPerToken).div(split.experts)
  if (batchSize <= 1) return k.toNumber()
  return new Decimal(1).sub(new Decimal(1).sub(k).pow(batchSize)).toNumber()
}
```

Rewrite `calculateMoEBatchedParams` body (keep its JSDoc):

```ts
  const activeParams = calculateMoEActiveParams(model)
  const split = splitMoEParams(model)
  const fraction = routedTouchedFraction(model, batchSize)
  if (!split || fraction === null || batchSize <= 1) return activeParams
  const batched = new Decimal(split.baseB).add(new Decimal(split.routedB).mul(fraction))
  const total = new Decimal(model.num_parameters_billion)
  // Never below the batch-1 figure, never above the full weight set.
  return Decimal.min(Decimal.max(batched, activeParams), total).toNumber()
```

Add after `splitMoEParams`:

```ts
/**
 * Weight GiB of a MoE model's replicated base and its routed experts.
 *
 * With a measured `high_precision` (tensors kept as >=16-bit floats), the base is filled
 * from those tensors first: if they cover it, the base is charged at their rate; if not,
 * the rest of the base is charged at the rate of the remaining (quantized) bytes. Without
 * it, the checkpoint's single average applies to both parts. Spec:
 * docs/superpowers/specs/2026-09-27-moe-weight-split-design.md
 */
export function moeWeightSplit(
  model: Model,
  format: QuantizationFormat,
): { baseGiB: Decimal; routedGiB: Decimal; measured: boolean } | null {
  const split = splitMoEParams(model)
  if (!split) return null
  const n = new Decimal(model.num_parameters_billion)
  const total = calculateModelWeightVRAM(model.num_parameters_billion, format, model)
  const base = new Decimal(split.baseB)
  const hp = weightRef(format, model)?.high_precision
  let baseGiB: Decimal
  if (!hp) {
    baseGiB = total.mul(base).div(n)
  } else if (base.lessThanOrEqualTo(hp.params_b)) {
    baseGiB = new Decimal(hp.gib).mul(base).div(hp.params_b)
  } else {
    const rest = n.sub(hp.params_b)
    const lowRate = rest.greaterThan(0) ? total.sub(hp.gib).div(rest) : new Decimal(0)
    baseGiB = new Decimal(hp.gib).add(base.sub(hp.params_b).mul(lowRate))
  }
  baseGiB = Decimal.min(baseGiB, total)
  return { baseGiB, routedGiB: total.sub(baseGiB), measured: !!hp }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/engines && npm run typecheck`
Expected: PASS (existing `calculateMoEBatchedParams` tests unchanged), typecheck clean. If `inference.ts` importing `weightRef` from `./quantization` creates a cycle error, confirm `quantization.ts` does not import `./inference` (it does not today).

- [ ] **Step 5: Commit**

```bash
rtk proxy git add src/engines/inference.ts src/engines/inference.test.ts
rtk proxy git commit -m "feat: moeWeightSplit prices a MoE base and experts at measured rates"
```

---

### Task 5: Expert-parallel memory uses the split

**Files:**
- Modify: `src/engines/multi-gpu.ts:~237-247` (`calculateExpertParallelVRAM`)
- Test: `src/engines/multi-gpu.test.ts`

**Interfaces:**
- Consumes: `moeWeightSplit` (Task 4). The function needs the quantization format: add a `quantization: QuantizationFormat` parameter to `calculateExpertParallelVRAM` and thread it from `calculateMultiGPUVRAM` (read the existing params object; if `calculateMultiGPUVRAM` has no format today, take it from a new optional `quantization` field on its params, defaulting to `'fp16'`, and pass it from `multi-node.ts`, the worker and the sync hook — grep `calculateMultiGPUVRAM(` and `calculateMultiNodeVRAM(`).

- [ ] **Step 1: Failing tests**

```ts
describe('expert-parallel memory with a measured split', () => {
  const models = validateModels(modelsData)
  const kimi = models.find((m) => m.id === 'moonshotai-kimi-k3')
  const gb300 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-gb300-nvl72')
  if (!kimi || !gb300) throw new Error('fixture')
  const single = calculateInferenceVRAM({ model: kimi, quantization: 'mxfp4', sequenceLength: 4096, batchSize: 1 })

  it('Kimi K3 mxfp4 EP8 replicates the BF16 base: ~272 GiB of weights per GPU (was 207)', () => {
    const r = calculateMultiGPUVRAM({ /* existing EP call shape */ singleGPU: single, model: kimi, gpuVramGB: gb300.vram_gb, numGPUs: 8, strategy: 'expert-parallel', interconnectType: 'nvlink-5', quantization: 'mxfp4' })
    expect(r.perGPU.modelWeights.toNumber()).toBeCloseTo(271.94, 0)
  })

  it('scales an offloaded on-device weight figure by the base share', () => {
    const half = { ...single, modelWeights: single.modelWeights.div(2) }
    const r = calculateMultiGPUVRAM({ singleGPU: half, model: kimi, gpuVramGB: gb300.vram_gb, numGPUs: 8, strategy: 'expert-parallel', interconnectType: 'nvlink-5', quantization: 'mxfp4' })
    expect(r.perGPU.modelWeights.toNumber()).toBeCloseTo(271.94 / 2, 0)
  })
})
```

Match the literal call shape to the existing EP tests in the same file (copy an existing EP test's `calculateMultiGPUVRAM` arguments and add `quantization`). Import `gpusData`/`validateGPUs` and `modelsData`/`validateModels` if missing.

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/engines/multi-gpu.test.ts`
Expected: FAIL (207.x vs 271.94).

- [ ] **Step 3: Implement** — replace the "Same bytes per parameter" block:

```ts
  // The replicated base and the routed experts at their own rates (moeWeightSplit),
  // applied as shares of the on-device weights so offloading still scales them.
  const weightSplit = moeWeightSplit(model, quantization)
  const baseShare = weightSplit
    ? weightSplit.baseGiB.div(weightSplit.baseGiB.add(weightSplit.routedGiB))
    : new Decimal(split.baseB).div(split.baseB + split.routedB)
  const replicatedMemory = singleGPU.modelWeights.mul(baseShare)
  const routedWeights = singleGPU.modelWeights.sub(replicatedMemory)
  const weightsPerGPU = replicatedMemory.add(routedWeights.div(numGPUs))
```

- [ ] **Step 4: Run to verify pass, then mutation-check**

Run: `npx vitest run src/engines && npm run typecheck`
Expected: PASS. Then temporarily replace `moeWeightSplit(model, quantization)` with `null`, re-run `npx vitest run src/engines/multi-gpu.test.ts`, expect FAIL, restore.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add -A src
rtk proxy git commit -m "feat: expert-parallel memory replicates the base at its measured rate"
```

---

### Task 6: Rebase onto main, then MoE decode uses the split

**Files:**
- Modify: `src/engines/performance.ts` (`estimatePerformance`, weight bytes)
- Test: `src/engines/performance.test.ts`

**Interfaces:**
- Consumes: `moeWeightSplit`, `routedTouchedFraction` (Task 4)

- [ ] **Step 0: Rebase** — `fix/offload-host-link` changes `estimatePerformance`. Once it is merged to main: `rtk proxy git fetch origin && rtk proxy git rebase origin/main`, resolve conflicts, `npx vitest run` green before continuing. If it is not merged yet, stop and report BLOCKED with that reason.

- [ ] **Step 1: Failing tests**

```ts
describe('MoE decode reads the base and touched experts at measured rates', () => {
  const models = validateModels(modelsData)
  const kimi = models.find((m) => m.id === 'moonshotai-kimi-k3')
  const b300 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-gb300-nvl72')
  if (!kimi || !b300) throw new Error('fixture')
  const ref = kimi.weight_refs?.mxfp4
  if (!ref) throw new Error('fixture')
  const { high_precision: _hp, ...refNoHp } = ref
  const averaged = { ...kimi, weight_refs: { ...kimi.weight_refs, mxfp4: refNoHp } }
  const run = (m: typeof kimi, batchSize: number, multiGPUResult: MultiGPUVRAMBreakdown | null = null) =>
    estimatePerformance({ model: m, gpu: b300, quantization: 'mxfp4', sequenceLength: 1024, batchSize, multiGPUResult })

  it('batch 1: ~127 GiB read per step instead of ~54, so decode is ~2.3x slower', () => {
    const measured = run(kimi, 1).tokensPerSecond.toNumber()
    const avg = run(averaged, 1).tokensPerSecond.toNumber()
    expect(measured).toBeLessThan(avg * 0.5)
  })

  it('expert parallelism reads the full base on every GPU and 1/N of the touched experts', () => {
    const single = calculateInferenceVRAM({ model: kimi, quantization: 'mxfp4', sequenceLength: 1024, batchSize: 64 })
    const ep = calculateMultiGPUVRAM({ /* same EP call shape as Task 5 */ singleGPU: single, model: kimi, gpuVramGB: b300.vram_gb, numGPUs: 8, strategy: 'expert-parallel', interconnectType: 'nvlink-5', quantization: 'mxfp4' })
    const measured = run(kimi, 64, ep).tokensPerSecond.toNumber()
    const avg = run(averaged, 64, ep).tokensPerSecond.toNumber()
    expect(measured).toBeLessThan(avg)
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/engines/performance.test.ts`
Expected: FAIL (measured equals averaged today).

- [ ] **Step 3: Implement** — in `estimatePerformance`, replace the weight-bytes computation for splittable MoE models:

```ts
  const weightSplit = moeWeightSplit(model, quantization)
  const touched = routedTouchedFraction(model, batchSize)
  const weightBytes =
    weightSplit && touched !== null
      ? weightSplit.baseGiB.add(weightSplit.routedGiB.mul(touched)).mul(BYTES_PER_GB)
      : calculateModelWeightVRAM(decodeParams, quantization, model).mul(BYTES_PER_GB)
```

and the expert-parallel per-GPU branch:

```ts
  const perGPUWeightBytes =
    layout.strategy === 'expert-parallel' && weightSplit && touched !== null
      ? weightSplit.baseGiB
          .add(weightSplit.routedGiB.mul(touched).div(layout.gpusPerStage))
          .mul(BYTES_PER_GB)
      : weightBytes.div(layout.gpusPerStage)
```

Remove `split` / `calculateModelWeightVRAM(split.baseB + ...)` code that this replaces, and any import left unused. Keep the offload logic from main operating on `weightBytes` as merged.

- [ ] **Step 4: Run to verify pass, then mutation-check**

Run: `npx vitest run && npm run typecheck`
Expected: PASS. Temporarily set `const weightSplit = null`, expect the new tests to FAIL, restore.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add src/engines/performance.ts src/engines/performance.test.ts
rtk proxy git commit -m "feat: MoE decode reads the base and touched experts at measured rates"
```

---

### Task 7: UI split label and docs

**Files:**
- Modify: `src/components/layout/ResultsPanel.tsx:~526-540` (weight-source line)
- Test: a component test next to ResultsPanel if one exists for the source line; otherwise a small pure helper `weightSplitLabel(model, format): string | null` in `src/engines/inference.ts` tested in `inference.test.ts`, used by ResultsPanel
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `src/pages/GuidePage.tsx` (or wherever the guide text lives; grep "Weights measured"), `README.md` if it describes refresh:models modes

- [ ] **Step 1: Failing test**

```ts
describe('weightSplitLabel', () => {
  const models = validateModels(modelsData)
  const get = (id: string) => models.find((m) => m.id === id) as Model
  it('labels MoE splits measured or estimated, dense none', () => {
    expect(weightSplitLabel(get('moonshotai-kimi-k3'), 'mxfp4')).toBe('base/expert split measured')
    expect(weightSplitLabel(get('qwen-qwen3-235b-a22b'), 'fp16')).toBe('split estimated (average rate)')
    expect(weightSplitLabel(get('meta-llama-llama-3.1-70b'), 'fp16')).toBeNull()
  })
})
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/engines/inference.test.ts`
Expected: FAIL (`weightSplitLabel` not defined).

- [ ] **Step 3: Implement**

```ts
/** Results-panel wording for how a MoE weight split was obtained; null for dense models. */
export function weightSplitLabel(model: Model, format: QuantizationFormat): string | null {
  const s = moeWeightSplit(model, format)
  if (!s) return null
  return s.measured ? 'base/expert split measured' : 'split estimated (average rate)'
}
```

In ResultsPanel, after the existing measured/estimated text inside the same `<p>`, append ` · {label}` when `weightSplitLabel(selectedModel, quantization)` is non-null (compute it next to `weightSourceRepo`).

Docs:
- `CLAUDE.md`: replace Domain Pitfall 9 with a Key Patterns bullet: "**MoE weights split by measured precision**: MoE weight refs carry `high_precision` (>=16-bit float tensors from the HF dtype summary; F8/U8/I32 are quantized storage). `moeWeightSplit` fills the replicated base from them first; EP memory and MoE decode charge base and experts separately. Without it (GGUF, 16-bit refs, no ref) the single average applies."; add `--split` to the `refresh:models` line.
- `CHANGELOG.md` [Unreleased] Changed: EP memory and MoE decode price base and experts at measured rates (Kimi K3 mxfp4 EP8 ~272 GiB/GPU, was 207; batch-1 decode ~2.3x slower, previously overstated).
- `ARCHITECTURE.md` and the guide: one sentence each.

- [ ] **Step 4: Run full verification**

Run: `npx vitest run && npm run typecheck && ./node_modules/.bin/biome check . && npm run build`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add -A
rtk proxy git commit -m "feat: show whether a MoE weight split is measured; docs"
```
