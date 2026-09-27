# Measured Weight Sizes and Model Auditor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Weight memory matches real Hugging Face checkpoints for every format that has one (measured `weight_refs` per model), with source-derived fallback constants, and `npm run refresh:models` becomes an auditor/drafter that measures and checks that data.

**Architecture:** One engine rule, `effectiveBytesPerParameter(format, model?)`, reads a pinned `weight_refs[format].gib` when present and the constant otherwise; weights and decode both go through it. The auditor is split into pure functions (`scripts/model-audit.ts`, unit-tested with fixtures) and network access (`scripts/hf.ts`), driven by a thin CLI (`scripts/fetch-models.ts`). The data (refs for 54 models) is produced by the auditor and reviewed against the independent 2026-09-26 spike.

**Tech Stack:** TypeScript strict, Zod 4 (`z.partialRecord`), decimal.js, Vitest, tsx, Biome.

**Spec:** `docs/superpowers/specs/2026-09-27-weight-refs-and-model-audit-design.md`

## Global Constraints

- One PR (review decision 2026-09-27).
- `weight_refs?: Partial<Record<QuantizationFormat, { repo: string; gib: number }>>`; `gib` = total weight-file bytes / 1024³, measured, never computed.
- Reference repo order: native release; vendor recipe (`nvidia/*` NVFP4, `RedHatAI/*` FP8 and INT4 w4a16); GGUF `unsloth/*-GGUF` then `bartowski/*-GGUF`; AWQ / GPTQ most-downloaded.
- New constants: `int4` 0.5625, `awq` 0.52, `gptq` 0.52, `gguf-q2_k` 0.366; `nvfp4` stays 0.5625.
- Audit tolerance: exact for integer config fields, 1% for parameter count and `weight_refs` sizes; `--strict` exits 1 on drift.
- The script never writes `models.json`.
- Zod 4: enum-keyed records are exhaustive; use `z.partialRecord` (zod.dev v4 changelog).
- Biome: run `rtk proxy npx biome check --write .` (plain `npx biome format --write` through rtk silently applies nothing).
- Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z
  ```

## Review Focus

1. **Format with no ref on a model that has other refs** (e.g. Gemma 4 with an NVFP4 ref, user picks `int4`) — falls back to the constant, not to another format's size. Test in Task 2.
2. **Custom or URL-restored model** (no `weight_refs`) — constants path, results unchanged by this feature. Test in Task 2.
3. **Gated repo during audit** (no token or no access) — reported "skipped (gated)", audit continues. Test in Task 3.
4. **GGUF repo holding two file sets for one tag** (split shards and a single file) — reported ambiguous, never summed. Test in Task 3.
5. **A pasted size that belongs to another format** — the per-format band test rejects it. Test in Task 5.

---

### Task 1: One quantization-format list and the `weight_refs` schema field

**Files:**
- Modify: `src/utils/schemas.ts` (add `QUANTIZATION_FORMATS`, `weight_refs`)
- Modify: `src/engines/types.ts` (derive `QuantizationFormat`; use the list in `CalculationInputSchema`)
- Test: `src/utils/schemas.test.ts`

**Interfaces:**
- Produces: `QUANTIZATION_FORMATS` (readonly tuple, `src/utils/schemas.ts`); `QuantizationFormat = (typeof QUANTIZATION_FORMATS)[number]` (`src/engines/types.ts`, same name as today); `Model['weight_refs']`.

- [ ] **Step 1: Write the failing tests** (append to `src/utils/schemas.test.ts`, which already imports `ModelSchema`; add `QUANTIZATION_FORMATS` to its import from `./schemas`)

```ts
describe('weight_refs', () => {
  const base = {
    id: 'm',
    name: 'M',
    architecture: 'dense' as const,
    num_parameters_billion: 32.7,
    hidden_size: 5376,
    num_hidden_layers: 60,
    num_attention_heads: 32,
    intermediate_size: 21504,
  }

  it('accepts refs for a subset of formats', () => {
    const r = ModelSchema.safeParse({
      ...base,
      weight_refs: { nvfp4: { repo: 'nvidia/Gemma-4-31B-IT-NVFP4', gib: 30.4 } },
    })
    expect(r.success).toBe(true)
  })

  it('rejects an unknown format key, an empty repo and a non-positive size', () => {
    expect(ModelSchema.safeParse({ ...base, weight_refs: { fp5: { repo: 'x/y', gib: 1 } } }).success).toBe(false)
    expect(ModelSchema.safeParse({ ...base, weight_refs: { fp8: { repo: '', gib: 1 } } }).success).toBe(false)
    expect(ModelSchema.safeParse({ ...base, weight_refs: { fp8: { repo: 'x/y', gib: 0 } } }).success).toBe(false)
  })

  it('lists every quantization format once', () => {
    expect(new Set(QUANTIZATION_FORMATS).size).toBe(QUANTIZATION_FORMATS.length)
    expect(QUANTIZATION_FORMATS).toContain('gguf-q2_k')
    expect(QUANTIZATION_FORMATS).toContain('mxfp4')
  })
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/utils/schemas.test.ts`
Expected: FAIL (`QUANTIZATION_FORMATS` is not exported / `weight_refs` accepted as unknown key → the rejection test fails).

- [ ] **Step 3: Implement**

`src/utils/schemas.ts`, near the top after `MAX_SEQUENCE_LENGTH`:
```ts
/**
 * Every weight quantization format, the single source for the QuantizationFormat
 * type (src/engines/types.ts) and every Zod enum over formats.
 */
export const QUANTIZATION_FORMATS = [
  'fp32',
  'fp16',
  'bf16',
  'fp8',
  'mxfp4',
  'nvfp6',
  'nvfp4',
  'int8',
  'int4',
  'nf4',
  'gptq',
  'awq',
  'gguf-q8_0',
  'gguf-q6_k',
  'gguf-q5_k_s',
  'gguf-q5_k_m',
  'gguf-q5_0',
  'gguf-q4_k_s',
  'gguf-q4_k_m',
  'gguf-q4_0',
  'gguf-q3_k_l',
  'gguf-q3_k_m',
  'gguf-q3_k_s',
  'gguf-q2_k',
] as const
```
In `ModelFields`, after `linear_state_bytes_per_session`:
```ts
  // Measured weight-file size per format from a published checkpoint (GiB = bytes / 1024^3).
  // Recipes differ (what stays 16-bit depends on who quantized it), so each format carries
  // its own reference repo. Absent formats use BYTES_PER_PARAMETER.
  weight_refs: z
    .partialRecord(
      z.enum(QUANTIZATION_FORMATS),
      z.object({ repo: z.string().min(1), gib: z.number().positive() }),
    )
    .optional(),
```
`src/engines/types.ts`: change the import to `import { MAX_SEQUENCE_LENGTH, QUANTIZATION_FORMATS } from '@utils/schemas'`, replace the `QuantizationFormat` union body with
```ts
export type QuantizationFormat = (typeof QUANTIZATION_FORMATS)[number]
```
(keep its doc comment), and in `CalculationInputSchema` replace the literal `quantization: z.enum([...])` list with `quantization: z.enum(QUANTIZATION_FORMATS),`.

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/utils && npm run typecheck`
Expected: PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
rtk proxy npx biome check --write .
git add src/utils/schemas.ts src/utils/schemas.test.ts src/engines/types.ts
git commit -m "feat: weight_refs model field; one quantization format list"
```

---

### Task 2: Engine uses measured sizes; source-derived constants

**Files:**
- Modify: `src/engines/quantization.ts` (`effectiveBytesPerParameter`, `calculateModelWeightVRAM` optional model, `weightSource`)
- Modify: `src/engines/constants.ts` (`int4`, `gptq`, `awq`, `gguf-q2_k` in `BYTES_PER_PARAMETER`, with comments)
- Modify: `src/engines/inference.ts:281`, `src/engines/performance.ts:141,158` (pass `model`)
- Test: `src/engines/quantization.test.ts`, plus existing tests whose absolute values come from the changed constants

**Interfaces:**
- Consumes: `Model['weight_refs']` (Task 1).
- Produces:
  - `effectiveBytesPerParameter(format: QuantizationFormat, model?: Model): Decimal`
  - `calculateModelWeightVRAM(numParametersBillion: number, format: QuantizationFormat, model?: Model): Decimal` (third argument new, optional; existing two-argument calls keep working)
  - `weightSource(model: Model, format: QuantizationFormat): string | null` (the ref repo, or null)

Ruling carried from the plan: the spec says `calculateModelWeightVRAM` "takes the Model"; an optional third argument delivers that for every caller that has a model (inference, decode) without rewriting the LoRA path and ~20 two-argument test calls. Cost if wrong: one extra parameter to remove later.

- [ ] **Step 1: Write the failing tests** (append to `src/engines/quantization.test.ts`; add `effectiveBytesPerParameter`, `weightSource` to its import, and `import type { Model } from '@utils/schemas'`)

```ts
describe('measured weight_refs', () => {
  const gemma: Model = {
    id: 'google-gemma-4-31b',
    name: 'Gemma 4 31B',
    architecture: 'dense',
    num_parameters_billion: 32.7,
    hidden_size: 5376,
    num_hidden_layers: 60,
    num_attention_heads: 32,
    intermediate_size: 21504,
    weight_refs: { nvfp4: { repo: 'nvidia/Gemma-4-31B-IT-NVFP4', gib: 30.4 } },
  }

  it('returns the measured checkpoint size when the format has a ref', () => {
    expect(calculateModelWeightVRAM(32.7, 'nvfp4', gemma).toNumber()).toBeCloseTo(30.4, 9)
  })

  it('scales a parameter subset by the measured bytes per parameter (decode)', () => {
    const half = calculateModelWeightVRAM(32.7 / 2, 'nvfp4', gemma).toNumber()
    expect(half).toBeCloseTo(15.2, 9)
  })

  it('falls back to the constant for a format without a ref', () => {
    expect(effectiveBytesPerParameter('int4', gemma).toNumber()).toBe(0.5625)
    expect(calculateModelWeightVRAM(32.7, 'int4', gemma).toString()).toBe(
      calculateModelWeightVRAM(32.7, 'int4').toString(),
    )
  })

  it('leaves models without refs (custom, URL-restored) on the constants', () => {
    const custom: Model = { ...gemma, weight_refs: undefined }
    expect(calculateModelWeightVRAM(32.7, 'nvfp4', custom).toString()).toBe(
      calculateModelWeightVRAM(32.7, 'nvfp4').toString(),
    )
  })

  it('names the measured source, or null when estimated', () => {
    expect(weightSource(gemma, 'nvfp4')).toBe('nvidia/Gemma-4-31B-IT-NVFP4')
    expect(weightSource(gemma, 'fp8')).toBeNull()
  })
})

describe('source-derived fallback constants', () => {
  it('int4 carries a 16-bit scale per group of 32 (Kimi K2 quantization_config)', () => {
    expect(effectiveBytesPerParameter('int4').toNumber()).toBe((4 + 16 / 32) / 8)
  })

  it('AWQ and GPTQ carry a 16-bit scale and 4-bit zero per group of 128', () => {
    expect(effectiveBytesPerParameter('awq').toNumber()).toBe(0.52)
    expect(effectiveBytesPerParameter('gptq').toNumber()).toBe(0.52)
  })

  it('GGUF Q2_K matches the median published file (2.93 bpp)', () => {
    expect(effectiveBytesPerParameter('gguf-q2_k').toNumber()).toBe(0.366)
  })

  it('NVFP4 is E2M1 plus an FP8 scale per 16 values', () => {
    expect(effectiveBytesPerParameter('nvfp4').toNumber()).toBe((4 + 8 / 16) / 8)
  })
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/engines/quantization.test.ts`
Expected: FAIL (`effectiveBytesPerParameter is not a function`).

- [ ] **Step 3: Implement**

`src/engines/quantization.ts`: add `import type { Model } from '@utils/schemas'`, then replace `calculateModelWeightVRAM` with:
```ts
/**
 * Bytes per parameter for a model in a format: the measured checkpoint when the model
 * has a weight_refs entry for it, else the format constant.
 *
 * Real checkpoints keep some tensors 16-bit and which ones depends on the recipe, so
 * a measured size (spec 2026-09-27) beats any single constant.
 */
export function effectiveBytesPerParameter(format: QuantizationFormat, model?: Model): Decimal {
  const ref = model?.weight_refs?.[format]
  if (ref && model) {
    return new Decimal(ref.gib)
      .mul(BYTES_PER_GB)
      .div(new Decimal(model.num_parameters_billion).mul(1e9))
  }
  return BYTES_PER_PARAMETER[format]
}

/**
 * Calculate model weight VRAM requirement
 *
 * For MoE models pass TOTAL parameters for memory; the decode path passes the active
 * (batched) subset, which scales by the same effective bytes per parameter.
 *
 * @param numParametersBillion - Parameters to size, in billions
 * @param format - Quantization format
 * @param model - When given and it has a weight_refs entry for `format`, the measured
 *   checkpoint sets the bytes per parameter
 * @returns VRAM requirement in GB (GiB) as Decimal
 *
 * @example
 * calculateModelWeightVRAM(7.0, 'fp16') // ~13.04 GB
 * calculateModelWeightVRAM(70.0, 'gptq') // ~33.90 GB (0.52 bytes/param)
 */
export function calculateModelWeightVRAM(
  numParametersBillion: number,
  format: QuantizationFormat,
  model?: Model,
): Decimal {
  return new Decimal(numParametersBillion)
    .mul(1e9)
    .mul(effectiveBytesPerParameter(format, model))
    .div(BYTES_PER_GB)
}

/** The repo a weight figure was measured from, or null when it is an estimate. */
export function weightSource(model: Model, format: QuantizationFormat): string | null {
  return model.weight_refs?.[format]?.repo ?? null
}
```
`src/engines/constants.ts`, in `BYTES_PER_PARAMETER` (and the second `int4: new Decimal(0.5)` table near line 129 only if it is the same table's duplicate for the same purpose — check with `grep -n "int4:" src/engines/constants.ts`; change the one in `BYTES_PER_PARAMETER`, leave a different-purpose table untouched):
```ts
  // 4 bits + one 16-bit scale per group of 32, symmetric (no zero point):
  // Kimi K2 quantization_config {group_size: 32, num_bits: 4, symmetric: true} = 4.5 bpp
  int4: new Decimal(0.5625),
  // 4 bits + 16-bit scale + 4-bit zero point per group of 128 ≈ 4.16 bpp.
  // Checkpoints also keep embeddings / lm_head 16-bit: weight_refs capture that per model.
  gptq: new Decimal(0.52),
  awq: new Decimal(0.52),
```
and
```ts
  // Median of 25 published Q2_K files (unsloth / bartowski, 2026-09-26 spike) ≈ 2.93 bpp:
  // the Q2_K mix stores some tensors in higher-bit types (the pure block is 2.625 bpp).
  'gguf-q2_k': new Decimal(0.366),
```
Update the doc lines above the table that say `gptq: 0.6 bytes (4-bit + 1.2x overhead ...)` and `awq: 0.6 bytes ...` to `0.52 bytes (4-bit + 16-bit scale and 4-bit zero per group of 128)`.

`src/engines/inference.ts:281`: `calculateModelWeightVRAM(model.num_parameters_billion, quantization, model)`.
`src/engines/performance.ts:141`: `calculateModelWeightVRAM(decodeParams, quantization, model)`; line ~158 (the expert-parallel branch): add `, model` as the third argument of that `calculateModelWeightVRAM(` call too.

- [ ] **Step 4: Run the new tests, then the whole suite**

Run: `npx vitest run src/engines/quantization.test.ts` → Expected: PASS.
Run: `npx vitest run 2>&1 | grep -E "FAIL|Tests "` → some existing tests fail on absolute values derived from the old constants. For each failure, confirm the value comes from `int4` / `gptq` / `awq` / `gguf-q2_k`, recompute with the new constant and update the expectation and its comment. Known ones:
  - `quantization.test.ts`: table rows `['int4', 0.5]` → `0.5625`, `['gptq', 0.6]` / `['awq', 0.6]` → `0.52`, `['gguf-q2_k', 0.328]` → `0.366` (comment `// ~2.93 bpp`); the "approximately 0.6 (0.5 * 1.2)" test → `toBeCloseTo(0.52, 2)` with comment "4-bit + group-128 scale and zero"; 70B GPTQ `39.12` → `70e9 × 0.52 / 1024³ = 33.90`; 13B INT4 → `13e9 × 0.5625 / 1024³ = 6.81`; 1.5B INT4 → `0.786`; FP16/GPTQ ratio `2 / 0.6` → `2 / 0.52 = 3.846`.
  - Any other file (e.g. `inference.test.ts`, `inference.integration.test.ts`) asserting a GPTQ/AWQ/INT4/Q2_K absolute size: recompute the same way. Relational assertions (A > B) need no change.
  Also update the docblock example in `src/engines/inference.ts` (`~39.12 GB (70B * 0.6 bytes/param)` → `~33.90 GB (70B * 0.52 bytes/param)`, and its total line accordingly).
Run again until: `Tests  N passed (N)`.

- [ ] **Step 5: Commit**

```bash
rtk proxy npx biome check --write . && npm run typecheck
git add src/engines src/utils
git commit -m "feat: size weights from measured checkpoints; source-derived INT4/AWQ/GPTQ/Q2_K constants"
```

---

### Task 3: Auditor pure functions

**Files:**
- Create: `scripts/model-audit.ts`
- Test: `tests/model-audit.test.ts`

**Interfaces:**
- Consumes: `Model` (`src/utils/schemas.ts`), `QuantizationFormat` (`src/engines/types.ts`).
- Produces (all exported from `scripts/model-audit.ts`):
  - `interface RepoFile { path: string; size: number }`
  - `interface MeasuredFields { hidden_size?: number; num_hidden_layers?: number; num_attention_heads?: number; num_kv_heads?: number; num_experts?: number; num_experts_per_token?: number; context_length?: number }`
  - `interface Drift { field: string; curated: number | undefined; measured: number }`
  - `const GGUF_TAGS: Partial<Record<QuantizationFormat, string>>`
  - `const MEASURED_FORMATS: QuantizationFormat[]` (formats `--measure` looks for)
  - `textConfig(cfg: Record<string, unknown>): Record<string, unknown>`
  - `configFields(cfg: Record<string, unknown>): MeasuredFields`
  - `compareModel(model: Model, fields: MeasuredFields, paramsB: number | null): Drift[]`
  - `weightFiles(files: RepoFile[], format: QuantizationFormat): RepoFile[] | 'ambiguous'`
  - `totalGiB(files: RepoFile[]): number` (rounded to 0.01)
  - `nativeFormat(cfg: Record<string, unknown>, dtypes?: Record<string, number>): QuantizationFormat`
  - `sameModel(candidate: string, baseRepo: string): boolean`
  - `pickReference(format: QuantizationFormat, candidates: string[]): string | null` (candidates already sorted by downloads)
  - `knownGoodGaps(model: Model, cfg: Record<string, unknown>): string[]`
  - `refDrift(format: QuantizationFormat, curatedGiB: number, files: RepoFile[] | null): string | null` (null = within 1%; a message otherwise, including "skipped (gated)" when files is null)

- [ ] **Step 1: Write the failing tests**

```ts
// tests/model-audit.test.ts
import type { Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import {
  compareModel,
  configFields,
  knownGoodGaps,
  nativeFormat,
  pickReference,
  refDrift,
  sameModel,
  totalGiB,
  weightFiles,
} from '../scripts/model-audit'

const GiB = 1024 ** 3
const model: Model = {
  id: 'google-gemma-4-31b',
  name: 'Gemma 4 31B',
  architecture: 'dense',
  num_parameters_billion: 32.7,
  hidden_size: 5376,
  num_hidden_layers: 60,
  num_attention_heads: 32,
  num_kv_heads: 16,
  intermediate_size: 21504,
  context_length: 262144,
}

describe('configFields', () => {
  it('reads multimodal configs through text_config', () => {
    const cfg = {
      text_config: {
        hidden_size: 5376,
        num_hidden_layers: 60,
        num_attention_heads: 32,
        num_key_value_heads: 16,
        max_position_embeddings: 262144,
      },
    }
    expect(configFields(cfg)).toEqual({
      hidden_size: 5376,
      num_hidden_layers: 60,
      num_attention_heads: 32,
      num_kv_heads: 16,
      context_length: 262144,
    })
  })

  it('reads MoE expert counts under their various names', () => {
    expect(configFields({ n_routed_experts: 256, num_experts_per_tok: 8 })).toMatchObject({
      num_experts: 256,
      num_experts_per_token: 8,
    })
  })
})

describe('compareModel', () => {
  it('reports integer fields that differ and parameters off by more than 1%', () => {
    const drift = compareModel(model, { hidden_size: 5376, num_hidden_layers: 62 }, 34)
    expect(drift).toEqual([
      { field: 'num_hidden_layers', curated: 60, measured: 62 },
      { field: 'num_parameters_billion', curated: 32.7, measured: 34 },
    ])
  })

  it('accepts parameters within 1%', () => {
    expect(compareModel(model, {}, 32.9)).toEqual([])
  })
})

describe('weightFiles', () => {
  it('sums safetensors, skipping original/ and metal/ copies and consolidated duplicates', () => {
    const files = [
      { path: 'model-00001-of-00002.safetensors', size: 5 * GiB },
      { path: 'model-00002-of-00002.safetensors', size: 5 * GiB },
      { path: 'consolidated.safetensors', size: 10 * GiB },
      { path: 'original/model.safetensors', size: 10 * GiB },
      { path: 'metal/model.safetensors', size: 10 * GiB },
      { path: 'config.json', size: 1000 },
    ]
    const picked = weightFiles(files, 'fp8')
    expect(picked !== 'ambiguous' && totalGiB(picked)).toBe(10)
  })

  it('matches the exact GGUF tag, not Q2_K_L / Q2_K_XL, and skips mmproj', () => {
    const files = [
      { path: 'Model-Q2_K.gguf', size: 3 * GiB },
      { path: 'Model-Q2_K_L.gguf', size: 4 * GiB },
      { path: 'Model-Q2_K_XL.gguf', size: 5 * GiB },
      { path: 'mmproj-Q2_K.gguf', size: 1 * GiB },
    ]
    const picked = weightFiles(files, 'gguf-q2_k')
    expect(picked !== 'ambiguous' && totalGiB(picked)).toBe(3)
  })

  it('sums split GGUF shards of one set', () => {
    const files = [
      { path: 'Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf', size: 20 * GiB },
      { path: 'Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf', size: 10 * GiB },
    ]
    const picked = weightFiles(files, 'gguf-q4_k_m')
    expect(picked !== 'ambiguous' && totalGiB(picked)).toBe(30)
  })

  it('reports two file sets for one tag as ambiguous', () => {
    const files = [
      { path: 'Model-Q8_0.gguf', size: 30 * GiB },
      { path: 'Q8_0/Model-Q8_0-00001-of-00002.gguf', size: 15 * GiB },
      { path: 'Q8_0/Model-Q8_0-00002-of-00002.gguf', size: 15 * GiB },
    ]
    expect(weightFiles(files, 'gguf-q8_0')).toBe('ambiguous')
  })
})

describe('nativeFormat', () => {
  it('reads the quantization_config', () => {
    expect(nativeFormat({ quantization_config: { quant_method: 'fp8' } })).toBe('fp8')
    expect(nativeFormat({ quantization_config: { quant_method: 'mxfp4' } })).toBe('mxfp4')
    expect(
      nativeFormat({
        quantization_config: {
          quant_method: 'compressed-tensors',
          config_groups: { g: { weights: { num_bits: 4, type: 'int' } } },
        },
      }),
    ).toBe('int4')
    const float4 = (group: number) => ({
      quantization_config: {
        quant_method: 'compressed-tensors',
        config_groups: { g: { weights: { num_bits: 4, type: 'float', group_size: group } } },
      },
    })
    expect(nativeFormat(float4(32))).toBe('mxfp4')
    expect(nativeFormat(float4(16))).toBe('nvfp4')
  })

  it('falls back to the dtype mix, else bf16', () => {
    expect(nativeFormat({}, { F8_E4M3: 600e9, BF16: 40e9 })).toBe('fp8')
    expect(nativeFormat({}, { BF16: 32e9 })).toBe('bf16')
  })
})

describe('reference selection', () => {
  const candidates = [
    'someone/Gemma-4-31B-IT-NVFP4',
    'nvidia/Gemma-4-31B-IT-NVFP4',
    'RedHatAI/gemma-4-31B-it-FP8-block',
    'cyankiwi/gemma-4-31B-it-AWQ-4bit',
    'bartowski/google_gemma-4-31B-it-GGUF',
    'unsloth/gemma-4-31B-it-GGUF',
  ]

  it('prefers the vendor recipe, then unsloth over bartowski for GGUF', () => {
    expect(pickReference('nvfp4', candidates)).toBe('nvidia/Gemma-4-31B-IT-NVFP4')
    expect(pickReference('fp8', candidates)).toBe('RedHatAI/gemma-4-31B-it-FP8-block')
    expect(pickReference('awq', candidates)).toBe('cyankiwi/gemma-4-31B-it-AWQ-4bit')
    expect(pickReference('gguf-q4_k_m', candidates)).toBe('unsloth/gemma-4-31B-it-GGUF')
    expect(pickReference('gptq', candidates)).toBeNull()
  })

  it('matches derivatives of the same model only', () => {
    expect(sameModel('unsloth/gemma-4-31B-it-GGUF', 'google/gemma-4-31B-it')).toBe(true)
    expect(sameModel('unsloth/gemma-4-12B-it-GGUF', 'google/gemma-4-31B-it')).toBe(false)
    expect(
      sameModel('RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8', 'meta-llama/Llama-3.1-8B-Instruct'),
    ).toBe(true)
    expect(sameModel('unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF', 'deepseek-ai/DeepSeek-R1')).toBe(
      false,
    )
  })
})

describe('refDrift', () => {
  it('passes within 1%, reports beyond, and skips gated repos', () => {
    const files = [{ path: 'model.safetensors', size: 30.4 * GiB }]
    expect(refDrift('fp8', 30.5, files)).toBeNull()
    expect(refDrift('fp8', 28, files)).toMatch(/28.*30.4/)
    expect(refDrift('fp8', 30.4, null)).toBe('skipped (gated)')
  })
})

describe('knownGoodGaps', () => {
  it('flags MLA, sliding and linear-attention configs missing their curated field', () => {
    expect(knownGoodGaps(model, { kv_lora_rank: 512 })).toContain('use_mla')
    expect(knownGoodGaps(model, { layer_types: ['sliding_attention', 'full_attention'] })).toContain(
      'kv_sliding_window',
    )
    expect(knownGoodGaps(model, { linear_attn_config: { kda_layers: [1] } })).toContain(
      'linear_state_bytes_per_session',
    )
  })

  it('is silent when the curated fields are present', () => {
    const curated: Model = {
      ...model,
      use_mla: true,
      kv_sliding_elements_per_token: 1,
      kv_sliding_window: 1024,
      linear_state_bytes_per_session: 1,
    }
    expect(
      knownGoodGaps(curated, {
        kv_lora_rank: 512,
        layer_types: ['sliding_attention', 'linear_attention'],
      }),
    ).toEqual([])
  })
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run tests/model-audit.test.ts`
Expected: FAIL, `Failed to resolve import "../scripts/model-audit"`.

- [ ] **Step 3: Implement**

```ts
// scripts/model-audit.ts
/**
 * Pure functions behind `npm run refresh:models`: compare curated models.json entries
 * with Hugging Face configs and checkpoints, and pick / measure reference weight files.
 * No network here (see scripts/hf.ts). Spec:
 * docs/superpowers/specs/2026-09-27-weight-refs-and-model-audit-design.md
 */
import type { QuantizationFormat } from '../src/engines/types'
import type { Model } from '../src/utils/schemas'

export interface RepoFile {
  path: string
  size: number
}

export interface MeasuredFields {
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_kv_heads?: number
  num_experts?: number
  num_experts_per_token?: number
  context_length?: number
}

export interface Drift {
  field: string
  curated: number | undefined
  measured: number
}

type Config = Record<string, unknown>

export const GGUF_TAGS: Partial<Record<QuantizationFormat, string>> = {
  'gguf-q8_0': 'Q8_0',
  'gguf-q6_k': 'Q6_K',
  'gguf-q5_k_m': 'Q5_K_M',
  'gguf-q5_k_s': 'Q5_K_S',
  'gguf-q5_0': 'Q5_0',
  'gguf-q4_k_m': 'Q4_K_M',
  'gguf-q4_k_s': 'Q4_K_S',
  'gguf-q4_0': 'Q4_0',
  'gguf-q3_k_l': 'Q3_K_L',
  'gguf-q3_k_m': 'Q3_K_M',
  'gguf-q3_k_s': 'Q3_K_S',
  'gguf-q2_k': 'Q2_K',
}

/** Formats `--measure` looks for beyond the native release */
export const MEASURED_FORMATS: QuantizationFormat[] = [
  'nvfp4',
  'fp8',
  'int4',
  'awq',
  'gptq',
  ...(Object.keys(GGUF_TAGS) as QuantizationFormat[]),
]

const num = (v: unknown): number | undefined => (typeof v === 'number' ? v : undefined)

export function textConfig(cfg: Config): Config {
  return (cfg.text_config ?? cfg.llm_config ?? cfg.language_config ?? cfg) as Config
}

export function configFields(cfg: Config): MeasuredFields {
  const t = textConfig(cfg)
  const first = (...keys: string[]) => keys.map((k) => num(t[k])).find((v) => v !== undefined)
  const fields: MeasuredFields = {
    hidden_size: first('hidden_size', 'd_model', 'dim'),
    num_hidden_layers: first('num_hidden_layers', 'n_layers', 'num_layers'),
    num_attention_heads: first('num_attention_heads', 'n_heads'),
    num_kv_heads: first('num_key_value_heads', 'n_kv_heads'),
    num_experts: first('n_routed_experts', 'num_experts', 'num_local_experts'),
    num_experts_per_token: first('num_experts_per_tok', 'num_experts_per_token', 'moe_topk'),
    context_length: first('max_position_embeddings') ?? num(cfg.max_position_embeddings),
  }
  return Object.fromEntries(
    Object.entries(fields).filter(([, v]) => v !== undefined),
  ) as MeasuredFields
}

export function compareModel(model: Model, fields: MeasuredFields, paramsB: number | null): Drift[] {
  const drift: Drift[] = []
  for (const [field, measured] of Object.entries(fields) as [keyof MeasuredFields, number][]) {
    const curated = model[field]
    if (curated !== measured) drift.push({ field, curated, measured })
  }
  if (paramsB !== null && Math.abs(paramsB - model.num_parameters_billion) > model.num_parameters_billion * 0.01) {
    drift.push({ field: 'num_parameters_billion', curated: model.num_parameters_billion, measured: paramsB })
  }
  return drift
}

export function weightFiles(files: RepoFile[], format: QuantizationFormat): RepoFile[] | 'ambiguous' {
  const tag = GGUF_TAGS[format]
  if (tag) {
    const re = new RegExp(`(^|[-_./])${tag}(\\.gguf$|-\\d{5}-of-|/)`, 'i')
    const hits = files.filter(
      (f) => f.path.endsWith('.gguf') && !/mmproj|draft|UD-|IQ\d/i.test(f.path) && re.test(f.path),
    )
    // One file set = one name once the shard suffix is removed
    const sets = new Set(hits.map((f) => f.path.replace(/-\d{5}-of-\d{5}\.gguf$/, '.gguf')))
    return sets.size > 1 ? 'ambiguous' : hits
  }
  const st = files.filter((f) => f.path.endsWith('.safetensors') && !/^(original|metal)\//.test(f.path))
  const hf = st.filter((f) => !/consolidated/.test(f.path))
  return hf.length ? hf : st
}

export function totalGiB(files: RepoFile[]): number {
  return Math.round((files.reduce((s, f) => s + f.size, 0) / 1024 ** 3) * 100) / 100
}

export function nativeFormat(cfg: Config, dtypes?: Record<string, number>): QuantizationFormat {
  const q = (cfg.quantization_config ?? textConfig(cfg).quantization_config) as Config | undefined
  if (q) {
    const method = String(q.quant_method ?? '').toLowerCase()
    const s = JSON.stringify(q)
    if (method === 'fp8' || method === 'fbgemm_fp8') return 'fp8'
    if (method === 'mxfp4') return 'mxfp4'
    if (/nvfp4/i.test(s)) return 'nvfp4'
    if (/"num_bits":\s*4/.test(s) && /"type":\s*"int"/.test(s)) return 'int4'
    // 4-bit float: NVFP4 scales per 16 values, MXFP4 per 32 (Kimi K3 native)
    if (/"num_bits":\s*4/.test(s) && /"type":\s*"float"/.test(s)) {
      return /"group_size":\s*16\b/.test(s) ? 'nvfp4' : 'mxfp4'
    }
    if (/"num_bits":\s*8/.test(s) && /"type":\s*"float"/.test(s)) return 'fp8'
  }
  if (dtypes) {
    const total = Object.values(dtypes).reduce((a, b) => a + b, 0)
    const f8 = Object.entries(dtypes)
      .filter(([k]) => /F8/.test(k))
      .reduce((a, [, v]) => a + v, 0)
    if (total > 0 && f8 / total > 0.5) return 'fp8'
  }
  return 'bf16'
}

const QUANT_SUFFIX =
  /-(gguf|fp8(-dynamic|-block)?|nvfp4|fp4|awq(-4bit)?|gptq(-int4)?|int4|w4a16|4bit|dynamic|instruct|it|bf16|mlx)\b/g

const baseName = (repo: string) =>
  (repo.split('/')[1] ?? repo)
    .toLowerCase()
    .replace(/^(google|meta|qwen|mistralai|deepseek-ai|moonshotai)_/, '')
    .replace(/^meta-(?=llama)/, '')
    .replace(QUANT_SUFFIX, '')
    .replace(/-\d{4}$/, '')

export function sameModel(candidate: string, baseRepo: string): boolean {
  return baseName(candidate) === baseName(baseRepo)
}

export function pickReference(format: QuantizationFormat, candidates: string[]): string | null {
  const find = (...res: RegExp[]) => {
    for (const re of res) {
      const hit = candidates.find((c) => re.test(c))
      if (hit) return hit
    }
    return null
  }
  if (GGUF_TAGS[format]) return find(/^unsloth\/.*GGUF$/i, /^bartowski\/.*GGUF$/i)
  switch (format) {
    case 'nvfp4':
      return find(/^nvidia\/.*NVFP4/i, /NVFP4/i)
    case 'fp8':
      return find(/^RedHatAI\/.*FP8/i, /FP8/i)
    case 'int4':
      return find(/^RedHatAI\/.*(w4a16|INT4)/i, /^(?!.*(AWQ|GPTQ)).*(w4a16|INT4)/i)
    case 'awq':
      return find(/AWQ/i)
    case 'gptq':
      return find(/GPTQ/i)
    default:
      return null
  }
}

export function knownGoodGaps(model: Model, cfg: Config): string[] {
  const t = textConfig(cfg)
  const types = JSON.stringify(t.layer_types ?? [])
  const gaps: string[] = []
  if (t.kv_lora_rank && !model.use_mla) gaps.push('use_mla')
  if (/sliding|chunk/.test(types) && !model.kv_sliding_window) gaps.push('kv_sliding_window')
  const linear =
    t.linear_attn_config !== undefined ||
    /linear|conv|mamba/.test(types) ||
    /M/.test(String(t.hybrid_override_pattern ?? ''))
  if (linear && !model.linear_state_bytes_per_session) gaps.push('linear_state_bytes_per_session')
  return gaps
}

export function refDrift(
  format: QuantizationFormat,
  curatedGiB: number,
  files: RepoFile[] | null,
): string | null {
  if (files === null) return 'skipped (gated)'
  const picked = weightFiles(files, format)
  if (picked === 'ambiguous') return `${format}: ambiguous file sets`
  const measured = totalGiB(picked)
  if (Math.abs(measured - curatedGiB) <= curatedGiB * 0.01) return null
  return `${format}: curated ${curatedGiB} GiB, measured ${measured} GiB`
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run tests/model-audit.test.ts && npm run typecheck`
Expected: PASS (19 tests), no type errors (scripts are typechecked by `tsconfig.node.json`).

- [ ] **Step 5: Commit**

```bash
rtk proxy npx biome check --write .
git add scripts/model-audit.ts tests/model-audit.test.ts
git commit -m "feat: model auditor pure functions (config drift, weight files, references)"
```

---

### Task 4: Network layer and CLI (`npm run refresh:models`)

**Files:**
- Create: `scripts/hf.ts`
- Modify (rewrite): `scripts/fetch-models.ts`

**Interfaces:**
- Consumes: everything Task 3 exports.
- Produces (`scripts/hf.ts`): `fetchConfig(repo): Promise<Record<string, unknown> | null>`, `fetchSafetensors(repo): Promise<{ total: number; parameters: Record<string, number> } | null>`, `fetchTree(repo): Promise<RepoFile[] | null>` (null = gated / not found), `searchRepos(term): Promise<string[]>` (sorted by downloads).

- [ ] **Step 1: Write `scripts/hf.ts`**

```ts
// scripts/hf.ts
/** Hugging Face Hub access for the model auditor. Token: HF_TOKEN or ~/.cache/huggingface/token. */
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import type { RepoFile } from './model-audit'

function readToken(): string | undefined {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN
  try {
    return readFileSync(`${homedir()}/.cache/huggingface/token`, 'utf8').trim()
  } catch {
    return undefined
  }
}

const token = readToken()
const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {}

async function getJSON(url: string): Promise<unknown | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const r = await fetch(url, { headers })
    if (r.ok) return r.json()
    if (r.status !== 429) return null
    await new Promise((s) => setTimeout(s, 2000 * (attempt + 1)))
  }
  return null
}

export async function fetchConfig(repo: string): Promise<Record<string, unknown> | null> {
  return (await getJSON(`https://huggingface.co/${repo}/resolve/main/config.json`)) as Record<
    string,
    unknown
  > | null
}

export async function fetchSafetensors(
  repo: string,
): Promise<{ total: number; parameters: Record<string, number> } | null> {
  const info = (await getJSON(`https://huggingface.co/api/models/${repo}?expand[]=safetensors`)) as {
    safetensors?: { total: number; parameters: Record<string, number> }
  } | null
  return info?.safetensors ?? null
}

export async function fetchTree(repo: string): Promise<RepoFile[] | null> {
  const t = await getJSON(`https://huggingface.co/api/models/${repo}/tree/main?recursive=true`)
  if (!Array.isArray(t)) return null
  return t
    .filter((f: { type: string }) => f.type === 'file')
    .map((f: { path: string; size: number; lfs?: { size: number } }) => ({
      path: f.path,
      size: f.lfs?.size ?? f.size,
    }))
}

export async function searchRepos(term: string): Promise<string[]> {
  const r = await getJSON(
    `https://huggingface.co/api/models?search=${encodeURIComponent(term)}&limit=100&sort=downloads`,
  )
  return Array.isArray(r) ? r.map((m: { id: string }) => m.id) : []
}
```

- [ ] **Step 2: Rewrite `scripts/fetch-models.ts` as the CLI**

Keep the existing `MODEL_IDS` roster array unchanged (copy it verbatim from the current file); replace everything else with:
```ts
// scripts/fetch-models.ts
/**
 * npm run refresh:models              audit models.json against Hugging Face (drift report)
 * npm run refresh:models -- --strict  same, exit 1 on any drift
 * npm run refresh:models -- --measure <model-id>   print weight_refs JSON to paste
 * npm run refresh:models -- --draft   draft roster ids missing from models.json
 * Never writes models.json. Spec: docs/superpowers/specs/2026-09-27-weight-refs-and-model-audit-design.md
 */
import { writeFile } from 'node:fs/promises'
import curated from '../src/data/models.json' with { type: 'json' }
import type { QuantizationFormat } from '../src/engines/types'
import type { Model } from '../src/utils/schemas'
import { fetchConfig, fetchSafetensors, fetchTree, searchRepos } from './hf'
import {
  compareModel,
  configFields,
  knownGoodGaps,
  MEASURED_FORMATS,
  nativeFormat,
  pickReference,
  refDrift,
  sameModel,
  totalGiB,
  weightFiles,
} from './model-audit'

const MODEL_IDS = [
  /* copy the existing roster verbatim */
]

const models = curated as Model[]
const repoOf = (m: Model) => (m.hf_url ?? '').replace('https://huggingface.co/', '')

async function audit(strict: boolean) {
  let problems = 0
  for (const m of models) {
    const repo = repoOf(m)
    const [cfg, st] = await Promise.all([fetchConfig(repo), fetchSafetensors(repo)])
    const lines: string[] = []
    if (!cfg) lines.push('config: skipped (gated or missing)')
    else {
      for (const d of compareModel(m, configFields(cfg), st ? st.total / 1e9 : null)) {
        lines.push(`${d.field}: curated ${d.curated}, measured ${d.measured}`)
      }
      for (const g of knownGoodGaps(m, cfg)) lines.push(`missing curated field: ${g}`)
    }
    for (const [format, ref] of Object.entries(m.weight_refs ?? {})) {
      if (!ref) continue
      const msg = refDrift(format as QuantizationFormat, ref.gib, await fetchTree(ref.repo))
      if (msg) lines.push(`weight_refs.${msg}`)
    }
    if (lines.length) {
      problems += lines.filter((l) => !l.includes('skipped')).length
      console.log(`${m.name}\n  ${lines.join('\n  ')}`)
    }
  }
  console.log(problems ? `\n${problems} drift item(s)` : '\nNo drift')
  if (strict && problems) process.exit(1)
}

async function measureRefs(repo: string) {
  const [cfg, st, tree] = await Promise.all([fetchConfig(repo), fetchSafetensors(repo), fetchTree(repo)])
  const refs: Partial<Record<QuantizationFormat, { repo: string; gib: number }>> = {}
  const native = nativeFormat(cfg ?? {}, st?.parameters)
  const own = tree ? weightFiles(tree, native) : null
  if (own && own !== 'ambiguous' && own.length) refs[native] = { repo, gib: totalGiB(own) }
  const candidates = (await searchRepos(repo.split('/')[1] ?? repo)).filter(
    (c) => c !== repo && sameModel(c, repo),
  )
  const trees = new Map<string, Awaited<ReturnType<typeof fetchTree>>>()
  for (const format of MEASURED_FORMATS) {
    if (refs[format]) continue
    const ref = pickReference(format, candidates)
    if (!ref) continue
    if (!trees.has(ref)) trees.set(ref, await fetchTree(ref))
    const files = trees.get(ref)
    if (!files) continue
    const picked = weightFiles(files, format)
    if (picked === 'ambiguous') {
      console.warn(`WARN ${ref} ${format}: ambiguous file sets, skipped`)
      continue
    }
    if (picked.length) refs[format] = { repo: ref, gib: totalGiB(picked) }
  }
  return refs
}

async function measure(id: string) {
  const m = models.find((x) => x.id === id)
  if (!m) throw new Error(`unknown model id ${id}`)
  console.log(JSON.stringify({ [id]: await measureRefs(repoOf(m)) }, null, 2))
}

async function draft() {
  const known = new Set(models.map(repoOf))
  const drafts = []
  for (const repo of MODEL_IDS.filter((r) => !known.has(r))) {
    const cfg = await fetchConfig(repo)
    const st = await fetchSafetensors(repo)
    if (!cfg) {
      console.warn(`WARN ${repo}: skipped (gated or missing)`)
      continue
    }
    const f = configFields(cfg)
    drafts.push({
      id: repo.replace('/', '-').toLowerCase(),
      name: repo.split('/')[1],
      architecture: f.num_experts ? 'moe' : 'dense',
      num_parameters_billion: st ? Math.round((st.total / 1e9) * 10) / 10 : undefined,
      ...f,
      hf_url: `https://huggingface.co/${repo}`,
      weight_refs: await measureRefs(repo),
    })
  }
  await writeFile('src/data/models-fetched.json', `${JSON.stringify(drafts, null, 2)}\n`)
  console.log(`Wrote ${drafts.length} draft(s) to src/data/models-fetched.json; review and merge by hand.`)
}

const args = process.argv.slice(2)
const at = args.indexOf('--measure')
if (at >= 0) await measure(args[at + 1] ?? '')
else if (args.includes('--draft')) await draft()
else await audit(args.includes('--strict'))
```

- [ ] **Step 3: Typecheck and smoke-run against the live Hub**

Run: `npm run typecheck` → Expected: no errors.
Run: `npx tsx scripts/fetch-models.ts --measure google-gemma-4-31b`
Expected: JSON with at least `"bf16"` (the original repo, ~61 GiB), `"nvfp4": { "repo": "nvidia/Gemma-4-31B-IT-NVFP4", "gib": ~30.4 }`, and `"fp8"` from a `RedHatAI/` repo.
Run: `npx tsx scripts/fetch-models.ts --measure moonshotai-kimi-k3`
Expected: `"mxfp4": { "repo": "moonshotai/Kimi-K3", "gib": ~1453.7 }` (native).
Run: `npm run refresh:models 2>&1 | tail -20`
Expected: a report ending in `N drift item(s)` or `No drift`, no crash; gated models show `skipped (gated or missing)`. Record surprising drift lines in the ledger; fixing curated data is not part of this task.

- [ ] **Step 4: Commit**

```bash
rtk proxy npx biome check --write .
git add scripts/hf.ts scripts/fetch-models.ts
git commit -m "feat: refresh:models audits models.json and measures weight_refs"
```

---

### Task 5: Measure `weight_refs` for all 54 models; integrity and corpus tests

**Files:**
- Modify: `src/data/models.json` (add `weight_refs` to entries)
- Test: `src/utils/models.test.ts`

**Interfaces:**
- Consumes: `--measure` (Task 4), `calculateInferenceVRAM` (engine, uses Task 2's model-aware weights).

- [ ] **Step 1: Write the failing tests** (append inside `describe('Model Database Validation', ...)` in `src/utils/models.test.ts`; add `import { calculateInferenceVRAM } from '@engines/inference'` and `import { BYTES_PER_PARAMETER } from '@engines/constants'`)

```ts
  it('gives every weight ref a size plausible for its format', () => {
    // Implied bytes per parameter; a size pasted under the wrong format falls outside.
    const band = (format: string): [number, number] => {
      if (format.startsWith('gguf-')) {
        const c = BYTES_PER_PARAMETER[format as keyof typeof BYTES_PER_PARAMETER].toNumber()
        return [c * 0.6, c * 1.4]
      }
      const bands: Record<string, [number, number]> = {
        nvfp4: [0.5, 1.0],
        mxfp4: [0.5, 1.0],
        int4: [0.5, 1.0],
        awq: [0.5, 1.0],
        gptq: [0.5, 1.0],
        fp8: [0.95, 1.3],
        bf16: [1.9, 2.1],
        fp16: [1.9, 2.1],
      }
      return bands[format] ?? [0, Number.POSITIVE_INFINITY]
    }
    let refs = 0
    for (const m of modelsData) {
      const w = (m as { weight_refs?: Record<string, { repo: string; gib: number }> }).weight_refs
      for (const [format, ref] of Object.entries(w ?? {})) {
        refs++
        expect(ref.repo, `${m.id} ${format}`).toMatch(/^[\w.-]+\/[\w.-]+$/)
        const bpp = (ref.gib * 1024 ** 3) / (m.num_parameters_billion * 1e9)
        const [lo, hi] = band(format)
        expect(bpp, `${m.id} ${format} ${bpp.toFixed(3)} bytes/param`).toBeGreaterThanOrEqual(lo)
        expect(bpp, `${m.id} ${format} ${bpp.toFixed(3)} bytes/param`).toBeLessThanOrEqual(hi)
      }
    }
    expect(refs).toBeGreaterThan(100)
  })

  it('matches the independent 2026-09-26 checkpoint measurements through the engine', () => {
    // [model id, format, reference repo, file GiB measured by the spike]
    const ANCHORS: [string, string, string, number][] = [
      ['google-gemma-4-31b', 'nvfp4', 'nvidia/Gemma-4-31B-IT-NVFP4', 30.4],
      ['moonshotai-kimi-k3', 'mxfp4', 'moonshotai/Kimi-K3', 1453.7],
      ['meta-llama-llama-3.1-8b', 'fp8', 'RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8', 8.5],
      ['deepseek-r1', 'gguf-q2_k', 'unsloth/DeepSeek-R1-GGUF', 227.3],
      ['qwen-qwen3.8-27b', 'awq', 'cyankiwi/Qwen3.8-27B-AWQ-INT4', 19.6],
    ]
    for (const [id, format, repo, fileGiB] of ANCHORS) {
      const m = validateModels(modelsData).find((x) => x.id === id)
      expect(m?.weight_refs?.[format as never]?.repo, `${id} ${format}`).toBe(repo)
      const weights = calculateInferenceVRAM({
        model: m as never,
        quantization: format as never,
        sequenceLength: 4096,
        batchSize: 1,
      }).modelWeights.toNumber()
      expect(Math.abs(weights - fileGiB) / fileGiB, `${id} ${format}`).toBeLessThan(0.01)
    }
  })
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/utils/models.test.ts`
Expected: FAIL (`expected 0 to be greater than 100` and anchor repo `undefined`).

- [ ] **Step 3: Measure all models**

Run (network, ~10-20 min; sequential to respect rate limits):
```bash
S=/private/tmp/claude-501/-Users-fjacquet-Projects-llmvram/529e3bfe-2166-47b0-8bd8-b6609ece1842/scratchpad
node -e 'console.log(require("./src/data/models.json").map(m=>m.id).join("\n"))' > $S/ids.txt
: > $S/refs.jsonl
while read id; do npx tsx scripts/fetch-models.ts --measure "$id" 2>>$S/refs.warn | tr -d '\n' >> $S/refs.jsonl; echo >> $S/refs.jsonl; done < $S/ids.txt
wc -l $S/refs.jsonl; cat $S/refs.warn | head
```
Expected: 54 lines, each a `{ "<id>": { ... } }` object (possibly `{}` for gated models with no public derivatives).

- [ ] **Step 4: Review against the spike, then merge into models.json**

Compare every measured ref with the spike's `matrix.json` (`$S/matrix.json`, same repos for most cells):
```bash
node -e '
const S=process.argv[1]; const fs=require("fs");
const refs=Object.assign({},...fs.readFileSync(S+"/refs.jsonl","utf8").trim().split("\n").filter(Boolean).map(l=>JSON.parse(l)));
const mx=JSON.parse(fs.readFileSync(S+"/matrix.json","utf8"));
for (const r of mx) for (const c of r.cells) { const f=refs[r.id]?.[c.format]; if (!f) continue;
  const same=c.repo.startsWith(f.repo); const d=Math.abs(f.gib-c.fileGiB)/c.fileGiB;
  if (same && d>0.01) console.log("DIFF", r.id, c.format, f.repo, f.gib, "spike", c.fileGiB) }
console.log("models with refs:", Object.values(refs).filter(v=>Object.keys(v).length).length)
' $S
```
Expected: no `DIFF` lines for identical repos (anything listed is a file-selection bug — fix `weightFiles` / `pickReference` with a new test in `tests/model-audit.test.ts` first, then re-measure that model). Spot-check refs for formats whose repo differs from the spike's choice by opening the repo on Hugging Face.

Merge (one-off, not committed):
```bash
node -e '
const S=process.argv[1]; const fs=require("fs");
const refs=Object.assign({},...fs.readFileSync(S+"/refs.jsonl","utf8").trim().split("\n").filter(Boolean).map(l=>JSON.parse(l)));
const p="src/data/models.json"; const ms=JSON.parse(fs.readFileSync(p,"utf8"));
for (const m of ms) if (refs[m.id] && Object.keys(refs[m.id]).length) m.weight_refs=refs[m.id];
fs.writeFileSync(p, JSON.stringify(ms,null,2)+"\n");
' $S
rtk proxy npx biome format --write src/data/models.json
git diff --stat src/data/models.json
```
Expected: only `weight_refs` blocks added (`git diff src/data/models.json | grep "^[-+]" | grep -v weight_refs` shows only brace/comma lines and the refs content).

- [ ] **Step 5: Run to verify the tests pass**

Run: `npx vitest run src/utils/models.test.ts`
Expected: PASS. If a band test fails, the named model/format is either a file-selection bug (fix with a test, re-measure) or a genuinely unusual checkpoint (drop that single ref, ledger a ruling). If an anchor's repo differs because `pickReference` chose another publisher, replace that anchor with another spike cell whose repo matches the committed ref (ledger a ruling).

- [ ] **Step 6: Commit**

```bash
npx vitest run 2>&1 | grep -E "FAIL|Tests "
git add src/data/models.json src/utils/models.test.ts
git commit -m "data: measured weight_refs for all models; integrity and corpus tests"
```

---

### Task 6: Source line in the results, docs, end-to-end check

**Files:**
- Modify: `src/components/layout/ResultsPanel.tsx` (line after `<MemoryBreakdownTable ... />`)
- Modify: `CHANGELOG.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `README.md`, `src/components/guide/GuidePage.tsx`

**Interfaces:**
- Consumes: `weightSource` (Task 2); store `selectedModel`, `quantization` (already destructured in `ResultsPanel`).

- [ ] **Step 1: Add the source line**

In `ResultsPanel.tsx`: `import { weightSource } from '@engines/quantization'`; directly after `<MemoryBreakdownTable breakdown={displayBreakdown} />`:
```tsx
            {(() => {
              const source = weightSource(selectedModel, quantization)
              return (
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {source ? (
                    <>
                      Weights measured from{' '}
                      <a
                        href={`https://huggingface.co/${source}`}
                        target="_blank"
                        rel="noreferrer"
                        className="underline"
                      >
                        {source}
                      </a>
                    </>
                  ) : (
                    `Weights estimated: no reference checkpoint for ${quantization}`
                  )}
                </p>
              )
            })()}
```

- [ ] **Step 2: Docs**

- `CHANGELOG.md`: add `## [Unreleased]` at the top (above `## [1.11.0]`) with:
  - `### Added`: `- Measured weight sizes: each model carries weight_refs per format ({repo, gib}) measured from published checkpoints (native release, NVIDIA NVFP4, RedHatAI FP8/INT4, unsloth/bartowski GGUF, AWQ/GPTQ). The engine uses them for memory and decode; the results say whether weights are measured or estimated. Corrects under-counts of up to 44% (Gemma 4 31B NVFP4) where checkpoints keep tensors in 16-bit.`
  - `- npm run refresh:models audits models.json against Hugging Face (config via text_config, exact safetensors counts, weight_refs drift, missing MLA / sliding / linear-state fields), measures weight_refs (--measure <id>) and drafts new roster entries (--draft). It never writes models.json.`
  - `### Changed`: `- Fallback bytes per parameter from the format definitions: INT4 0.5625 (16-bit scale per group of 32), AWQ/GPTQ 0.52 (scale and zero per group of 128, was 0.6), GGUF Q2_K 0.366 (median published file, was 0.328).`
- `CLAUDE.md`: in Commands, replace the `refresh:models` line with `npm run refresh:models   # Audit models.json vs Hugging Face; --measure <id> prints weight_refs; --draft drafts new roster ids (never writes models.json)`. In Key Patterns, after the KV known-good line, add: `- **Weight sizes are measured per format**: weight_refs[format] = {repo, gib} from the reference checkpoint (native, nvidia NVFP4, RedHatAI FP8/INT4, unsloth→bartowski GGUF, most-downloaded AWQ/GPTQ), produced by \`refresh:models --measure\`. effectiveBytesPerParameter uses it for memory and decode; BYTES_PER_PARAMETER is only the fallback.`
- `ARCHITECTURE.md`, Quantization Engine section: replace the `**Formula:**` line and the paragraph under it with:
  ```markdown
  **Formula:** `weight_GB = params × effectiveBytesPerParameter(format, model) / 1024³`

  `effectiveBytesPerParameter` returns the measured `weight_refs[format].gib × 1024³ / (params × 1e9)` when the model has a reference checkpoint for the format (what stays 16-bit depends on the recipe, so each format is measured), else `BYTES_PER_PARAMETER[format]` (INT4 0.5625, AWQ/GPTQ 0.52, NVFP4 0.5625, GGUF from measured bpp).
  ```
- `README.md`, Performance/quantization feature bullet list: add `- **Measured weights**: weight sizes come from published checkpoints per format where one exists, and the results say whether a figure is measured or estimated`.
- `GuidePage.tsx`, next to the Model Weights / quantization explanation (find it with `grep -n "Quantization" src/components/guide/GuidePage.tsx | head`): a sentence `Where a published checkpoint exists for the chosen format, the weight size is measured from it (the results name the repository); otherwise it is estimated from the format&apos;s bytes per parameter.`

- [ ] **Step 3: Full verification**

```bash
rtk proxy npx biome check --write . && rtk proxy npx biome check .
npx vitest run
npm run typecheck
npm run build
```
Expected: Biome clean, all tests pass, no type errors, build succeeds.

- [ ] **Step 4: End-to-end in the dev server**

Run `npx vite --port 5199 --strictPort` in the background; open `http://localhost:5199/llmvram/`; in the console (or Playwright `browser_evaluate`):
```js
const { useUIStore } = await import('/llmvram/src/store/uiStore.ts')
const models = (await import('/llmvram/src/data/models.json')).default
const gpus = (await import('/llmvram/src/data/gpus.json')).default
const s = useUIStore.getState()
s.setSelectedModel(models.find((m) => m.id === 'google-gemma-4-31b'))
s.setSelectedGPU(gpus.find((g) => g.id === 'nvidia-h200-141gb'))
s.setQuantization('nvfp4')
```
Expected: Model Weights ≈ 30.4 GB and the line "Weights measured from nvidia/Gemma-4-31B-IT-NVFP4". Then `s.setQuantization('nf4')`: the line reads "Weights estimated: no reference checkpoint for nf4". Stop the server; delete `.playwright-mcp/` if created.

- [ ] **Step 5: Commit**

```bash
git add src/components/layout/ResultsPanel.tsx CHANGELOG.md CLAUDE.md ARCHITECTURE.md README.md src/components/guide/GuidePage.tsx
git commit -m "feat: show whether weights are measured or estimated; docs"
```
