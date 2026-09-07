# Long-Context Support up to 1M+ Tokens — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the calculator express and correctly model sequence lengths up to 10,485,760 tokens, fixing the three calculation errors that only become visible at long context.

**Architecture:** Three engine changes plus a data field. Activation memory stops scaling with the full context window and is bounded by a prefill chunk. TTFT stops being a constant and becomes a compute-bound prefill model with a linear and a quadratic term. Decode throughput switches from total to active parameters for MoE models, backed by a new explicit `active_parameters_billion` field with a derived fallback. The UI then raises its caps and gains a native-context marker with a non-blocking warning.

**Tech Stack:** React 19, TypeScript strict (`noUncheckedIndexedAccess`), Vite 7, Zustand, Zod, decimal.js, Vitest + jsdom, Biome.

**Spec:** `docs/superpowers/specs/2026-09-07-long-context-1m-design.md`

## Global Constraints

- **Biome formatting:** 2-space indent, single quotes, **no semicolons** (ASI), 100-char line width. Unused imports and variables are **errors**.
- **All money math uses `decimal.js`.** Engine functions return `Decimal`, never `number`, except where an existing signature already returns `number` (e.g. `calculateMoEActiveParams`).
- **Engines stay pure.** No React, no DOM, no store access in `src/engines/`.
- **`src/data/models.json` must remain sorted alphabetically by `name`** after any edit.
- **MoE weight VRAM always uses total parameters.** Only throughput and prefill FLOPs use active parameters.
- **Never write model data from memory.** Every `active_parameters_billion` value is either encoded in the model's own name or verified against its Hugging Face model card / `config.json`.
- **Do not edit `src/**/*.js` or `src/**/*.d.ts`** — gitignored build artifacts (`.gitignore` lines 43–44).
- **Test commands:** `npx vitest run <path>` for a single file (never bare `npm test`, which watches). `npm run lint` and `npm run typecheck` are wrapped by the rtk hook; if they misbehave use `rtk proxy npm run lint` / `rtk proxy npm run typecheck`.
- **Coverage targets:** 75% lines/functions/branches/statements on `src/engines/` and `src/utils/`.

## File Structure

**Created:** none. This change extends existing modules.

**Modified:**

| File | Responsibility after the change |
|---|---|
| `src/engines/constants.ts` | Adds `PREFILL_CHUNK_TOKENS`, `PREFILL_MFU` |
| `src/utils/schemas.ts` | Adds `MAX_SEQUENCE_LENGTH` (leaf module, no engine imports) and `active_parameters_billion` on `ModelSchema`; raises `TrainingInputSchema` cap |
| `src/engines/types.ts` | Raises `CalculationInputSchema` cap; extends `PerformanceEstimate` |
| `src/engines/inference.ts` | Bounds activations by the prefill chunk; three-tier `calculateMoEActiveParams` |
| `src/engines/performance.ts` | Prefill/TTFT model; active-parameter roofline |
| `src/workers/calculation.worker.ts` | Threads `sequenceLength` into `estimatePerformance`; serializes the new fields |
| `src/hooks/useInferenceCalculation.ts` | Same, plus reconstructs the new fields from the worker |
| `src/components/layout/ResultsPanel.tsx` | Prefill display, adaptive TTFT formatting, correct ms into the comparison snapshot |
| `src/components/inputs/SequenceLengthInput.tsx` | Dynamic max, 1M presets, native-context marker, warning badge |
| `src/store/comparisonStore.ts` | Unit comment corrected to match the value actually stored |
| `src/data/models.json` | `active_parameters_billion` on 33 MoE entries |
| `scripts/fetch-models.ts` | Divergence warning on refresh |
| `CHANGELOG.md`, `README.md`, `src/components/guide/GuidePage.tsx` | Documentation |

**Dependency note:** `performance.ts` gains an import of `calculateMoEActiveParams` from `./inference`. `inference.ts` does not import `performance.ts`, so there is no cycle. `schemas.ts` imports only `zod` and stays a leaf; `types.ts` will import `MAX_SEQUENCE_LENGTH` from `@utils/schemas`, which is acyclic because `schemas.ts` imports nothing from `engines/`.

---

### Task 1: Bound activation memory by the prefill chunk

**Files:**
- Modify: `src/engines/constants.ts` (append after `FLASH_ATTENTION_LONG_THRESHOLD`, ~line 374)
- Modify: `src/engines/inference.ts:82-101` (`calculateActivationMemory`)
- Test: `src/engines/inference.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `PREFILL_CHUNK_TOKENS: number` exported from `src/engines/constants.ts`. `calculateActivationMemory(model: Model, sequenceLength: number, batchSize: number): Decimal` — signature unchanged.

- [ ] **Step 1: Write the failing test**

Append to `src/engines/inference.test.ts` inside the existing `describe('calculateActivationMemory', ...)` block (if that block does not exist, add it at the end of the file):

```ts
describe('calculateActivationMemory - prefill chunk bound', () => {
  it('is unchanged at or below the prefill chunk', () => {
    const at4k = calculateActivationMemory(llama7b, 4096, 1)
    const at8k = calculateActivationMemory(llama7b, 8192, 1)

    // 1 * 4096 * 11008 * 4 / 1024^3
    expect(at4k.toNumber()).toBeCloseTo((4096 * 11008 * 4) / 1024 ** 3, 6)
    expect(at8k.toNumber()).toBeCloseTo((8192 * 11008 * 4) / 1024 ** 3, 6)
  })

  it('plateaus above the prefill chunk instead of growing with the context window', () => {
    const at8k = calculateActivationMemory(llama7b, 8192, 1)
    const at128k = calculateActivationMemory(llama7b, 131072, 1)
    const at1m = calculateActivationMemory(llama7b, 1048576, 1)

    expect(at128k.toString()).toBe(at8k.toString())
    expect(at1m.toString()).toBe(at8k.toString())
  })

  it('still scales with batch size above the chunk', () => {
    const batch1 = calculateActivationMemory(llama7b, 1048576, 1)
    const batch4 = calculateActivationMemory(llama7b, 1048576, 4)

    expect(batch4.div(batch1).toNumber()).toBeCloseTo(4, 9)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engines/inference.test.ts -t 'prefill chunk bound'`
Expected: FAIL — "plateaus above the prefill chunk" fails because `at1m` is 128× `at8k`.

- [ ] **Step 3: Add the constant**

Append to `src/engines/constants.ts`:

```ts
/**
 * Prefill chunk size in tokens
 *
 * Inference engines process a long prompt in chunks rather than in one pass, so peak
 * activation memory is bounded by the chunk size, NOT by the context window. Without
 * this bound a 1M-token context reports tens of GB of activations, which is fiction.
 *
 * Value is vLLM's default `max_num_batched_tokens` for UsageContext.LLM_CLASS on GPUs
 * below 70 GiB (16384 above; 2048 for the OpenAI API server context).
 *
 * Reference: https://docs.vllm.ai/en/stable/configuration/optimization
 */
export const PREFILL_CHUNK_TOKENS = 8192
```

- [ ] **Step 4: Bound the activation formula**

In `src/engines/inference.ts`, extend the `constants` import to include `PREFILL_CHUNK_TOKENS`:

```ts
import {
  BYTES_PER_GB,
  FRAMEWORK_OVERHEAD_GB,
  PER_GPU_FRAMEWORK_OVERHEAD_GB,
  PREFILL_CHUNK_TOKENS,
} from './constants'
```

Then replace the final calculation block of `calculateActivationMemory`:

```ts
  // Peak activations are bounded by the prefill chunk, not the context window.
  // Decode activations are one token wide; prefill is processed PREFILL_CHUNK_TOKENS
  // at a time, so activations plateau once the prompt exceeds one chunk.
  const activeTokens = Math.min(sequenceLength, PREFILL_CHUNK_TOKENS)

  // batch * chunk_tokens * intermediate_size * 4
  // The factor 4 is 2 bytes (bf16 activations) x ~2 live buffers per layer.
  // NOT FP32 storage, despite what this comment used to claim.
  const activationBytes = new Decimal(batchSize)
    .mul(activeTokens)
    .mul(effectiveIntermediateSize)
    .mul(4)

  return activationBytes.div(BYTES_PER_GB)
```

Also update the JSDoc above the function: replace the `Formula: batch_size * sequenceLength * intermediate_size * 4 / BYTES_PER_GB` line and the "factor of 4 is for FP32 activation storage" sentence with:

```
 * Formula: batch * min(sequenceLength, PREFILL_CHUNK_TOKENS) * intermediateSize * 4 / BYTES_PER_GB
 * The factor of 4 is 2 bytes (bf16) x ~2 live buffers per layer, not FP32 storage.
```

- [ ] **Step 5: Run the new tests**

Run: `npx vitest run src/engines/inference.test.ts -t 'prefill chunk bound'`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the whole inference suite for regressions**

Run: `npx vitest run src/engines/inference.test.ts src/engines/inference.integration.test.ts`
Expected: PASS. Only one pre-existing case runs above 8,192 tokens (`inference.integration.test.ts:243`, "long context: 131K tokens") and its assertions are lower bounds and ratios (`kvCacheRatio > 0.3`) that only strengthen when activations shrink. If it fails, the ratio moved the wrong way — stop and investigate rather than loosening the assertion.

- [ ] **Step 7: Commit**

```bash
rtk git add src/engines/constants.ts src/engines/inference.ts src/engines/inference.test.ts
rtk git commit -m "fix(engines): bound activation memory by the prefill chunk

Activations scaled with the full context window, reporting 68 GB at 1M tokens
for a 27B dense model. Real engines chunk prefill, so peak activations plateau
at PREFILL_CHUNK_TOKENS (8192, vLLM's default max_num_batched_tokens).

The x4 factor is preserved and re-documented as 2 bytes x ~2 live buffers
rather than FP32, so no VRAM figure at or below 8192 tokens changes.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Explicit MoE active parameters with a derived fallback

**Files:**
- Modify: `src/utils/schemas.ts` (`ModelSchema`, after `num_experts_per_token`, ~line 75)
- Modify: `src/engines/inference.ts:35-50` (`calculateMoEActiveParams`)
- Test: `src/engines/inference.test.ts`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `Model.active_parameters_billion?: number`. `calculateMoEActiveParams(model: Model): number` — signature unchanged, three-tier behaviour.

- [ ] **Step 1: Write the failing test**

Append to `src/engines/inference.test.ts`:

```ts
describe('calculateMoEActiveParams - three-tier resolution', () => {
  // Qwen3.6 35B A3B shape: 40 layers, 256 experts (8 active), hidden 2048,
  // per-expert intermediate 512, 36B total.
  const qwen35bA3b: Model = {
    id: 'test-qwen-35b-a3b',
    name: 'Test Qwen 35B A3B',
    architecture: 'moe',
    num_parameters_billion: 36,
    hidden_size: 2048,
    num_hidden_layers: 40,
    num_attention_heads: 16,
    num_kv_heads: 2,
    intermediate_size: 512,
    num_experts: 256,
    num_experts_per_token: 8,
  }

  it('tier 1: uses active_parameters_billion when present', () => {
    const withExplicit: Model = { ...qwen35bA3b, active_parameters_billion: 3 }
    expect(calculateMoEActiveParams(withExplicit)).toBe(3)
  })

  it('tier 2: derives from per-expert dimensions when the field is absent', () => {
    // expertParams = 40 * 256 * 3 * 2048 * 512 / 1e9 = 32.21225472
    // nonExpert    = 36 - 32.21225472 = 3.78774528
    // active       = 3.78774528 + 32.21225472 * (8 / 256) = 4.7942...
    expect(calculateMoEActiveParams(qwen35bA3b)).toBeCloseTo(4.7942, 3)
  })

  it('tier 2: never returns more than the total parameter count', () => {
    // Bad data: derived expert params (32.2B) exceed the declared total, AND the active
    // ratio is high enough that expertParams * ratio alone would still overshoot.
    // Both clamps have to fire: nonExpert floors at 0, then the sum caps at the total.
    // (A low ratio like 8/256 would pass without exercising the outer clamp at all.)
    const inconsistent: Model = {
      ...qwen35bA3b,
      num_parameters_billion: 10,
      num_experts_per_token: 128,
    }
    expect(calculateMoEActiveParams(inconsistent)).toBe(10)
  })

  it('tier 3: dense models return the full parameter count', () => {
    expect(calculateMoEActiveParams(llama7b)).toBe(llama7b.num_parameters_billion)
  })

  it('tier 3: MoE with missing expert fields returns the full parameter count', () => {
    const incomplete: Model = { ...qwen35bA3b, num_experts: undefined }
    expect(calculateMoEActiveParams(incomplete)).toBe(36)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engines/inference.test.ts -t 'three-tier resolution'`
Expected: FAIL — tier 1 fails to typecheck (`active_parameters_billion` not on `Model`), and tier 2 returns 8.1 from the old 20/80 heuristic.

- [ ] **Step 3: Add the schema field**

In `src/utils/schemas.ts`, inside `ModelSchema`, directly after `num_experts_per_token`:

```ts
  // Active parameters per token for MoE models (e.g. 3 for a 36B "A3B" model).
  // Used for decode throughput and prefill FLOPs only — weight VRAM always uses
  // num_parameters_billion, because every expert must be resident.
  // Absent means "derive it"; see calculateMoEActiveParams tier 2.
  active_parameters_billion: z.number().positive().optional(),
```

- [ ] **Step 4: Replace the heuristic with three tiers**

In `src/engines/inference.ts`, replace the body of `calculateMoEActiveParams`:

```ts
export function calculateMoEActiveParams(model: Model): number {
  // Tier 1: explicit value verified from the model card
  if (model.active_parameters_billion) {
    return model.active_parameters_billion
  }

  // Tier 3: dense model, or MoE fields incomplete
  if (model.architecture === 'dense' || !model.num_experts || !model.num_experts_per_token) {
    return model.num_parameters_billion
  }

  // Tier 2: derive from stored dimensions. Our MoE entries store the PER-EXPERT
  // intermediate_size (Qwen3.6 35B A3B: 512, not 17408), so expert parameters are
  // layers x experts x 3 projections (gate, up, down) x hidden x per-expert intermediate.
  const expertParams = new Decimal(model.num_hidden_layers)
    .mul(model.num_experts)
    .mul(3)
    .mul(model.hidden_size)
    .mul(model.intermediate_size)
    .div(1e9)

  const total = new Decimal(model.num_parameters_billion)
  // Guard against inconsistent data where the derivation exceeds the declared total
  const nonExpertParams = Decimal.max(total.sub(expertParams), 0)
  const activeRatio = new Decimal(model.num_experts_per_token).div(model.num_experts)

  return Decimal.min(nonExpertParams.add(expertParams.mul(activeRatio)), total).toNumber()
}
```

Update the JSDoc above it: the old block documents a "20% shared, 80% expert" split and a Mixtral example returning ~18.68. Replace the `@example` block with:

```
 * @example
 * ```ts
 * // Tier 1 - explicit field wins
 * calculateMoEActiveParams({ ...qwen35bA3b, active_parameters_billion: 3 }) // 3
 *
 * // Tier 2 - derived from per-expert dimensions
 * calculateMoEActiveParams(qwen35bA3b) // ~4.79
 *
 * // Tier 3 - dense model
 * calculateMoEActiveParams(llama70b) // 70.0
 * ```
```

- [ ] **Step 5: Run the new tests**

Run: `npx vitest run src/engines/inference.test.ts -t 'three-tier resolution'`
Expected: PASS (5 tests).

- [ ] **Step 6: Fix any pre-existing MoE activation assertions**

Run: `npx vitest run src/engines/inference.test.ts src/engines/inference.integration.test.ts`

`calculateActivationMemory` scales `intermediate_size` by `activeParams / total`, so any existing test asserting an exact MoE activation figure shifts when tier 2 replaces the 20/80 heuristic. For each failure, recompute the expected value from the new ratio and update it — do **not** relax the assertion to a range.

- [ ] **Step 7: Typecheck**

Run: `rtk proxy npm run typecheck`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
rtk git add src/utils/schemas.ts src/engines/inference.ts src/engines/inference.test.ts
rtk git commit -m "feat(engines): explicit MoE active parameters with derived fallback

The flat 20/80 shared/expert heuristic returned 8.1B for a model literally
named A3B. Replace it with three tiers: an explicit active_parameters_billion
field, a derivation from the stored per-expert dimensions (4.79B for the same
model), then the total parameter count for dense models.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Prefill/TTFT model and active-parameter throughput

This task changes a public signature, so every call site moves with it. Splitting it would leave the build red between tasks.

**Files:**
- Modify: `src/engines/constants.ts` (append `PREFILL_MFU`)
- Modify: `src/engines/types.ts:70-81` (`PerformanceEstimate`)
- Modify: `src/engines/performance.ts` (whole `estimatePerformance` body and `PerformanceParams`)
- Modify: `src/workers/calculation.worker.ts` (response type + both call sites)
- Modify: `src/hooks/useInferenceCalculation.ts` (`reconstructPerformanceEstimate` + sync fallback call)
- Modify: `src/components/layout/ResultsPanel.tsx:323` (snapshot units)
- Modify: `src/store/comparisonStore.ts:34` (unit comment)
- Test: `src/engines/performance.test.ts`, `src/store/comparisonStore.test.ts`

**Interfaces:**
- Consumes: `calculateMoEActiveParams` from Task 2.
- Produces:
  - `PerformanceParams` gains a **required** `sequenceLength: number`.
  - `PerformanceEstimate` gains `prefillSeconds: Decimal | null`, `prefillBottleneck: 'linear' | 'attention'`, `prefillEstimateDegraded: boolean`.
  - `PREFILL_MFU: Decimal` from `src/engines/constants.ts`.

- [ ] **Step 1: Write the failing tests**

Append to `src/engines/performance.test.ts`:

```ts
describe('estimatePerformance - prefill model', () => {
  it('TTFT grows with sequence length', () => {
    const short = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })
    const long = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 131072,
      batchSize: 1,
    })

    expect(long.timeToFirstToken.greaterThan(short.timeToFirstToken)).toBe(true)
  })

  it('grows super-linearly once the quadratic attention term dominates', () => {
    const at256k = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 262144,
      batchSize: 1,
    })
    const at512k = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 524288,
      batchSize: 1,
    })

    const ratio = at512k.prefillSeconds?.div(at256k.prefillSeconds ?? 1).toNumber() ?? 0
    expect(ratio).toBeGreaterThan(2)
  })

  it('reports the linear term as the bottleneck at short context', () => {
    const result = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })

    expect(result.prefillBottleneck).toBe('linear')
    expect(result.prefillEstimateDegraded).toBe(false)
  })

  it('reports the attention term as the bottleneck at 1M context', () => {
    const result = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1048576,
      batchSize: 1,
    })

    expect(result.prefillBottleneck).toBe('attention')
  })

  it('degrades gracefully when the GPU has no FLOPS data', () => {
    const noFlopsGPU: GPU = { ...h100_80gb_sxm, fp16_tflops: undefined, fp32_tflops: undefined }
    const result = estimatePerformance({
      model: llama7b,
      gpu: noFlopsGPU,
      quantization: 'fp16',
      sequenceLength: 131072,
      batchSize: 1,
    })

    expect(result.prefillSeconds).toBeNull()
    expect(result.prefillEstimateDegraded).toBe(true)
    expect(result.timeToFirstToken.isFinite()).toBe(true)
    expect(result.timeToFirstToken.greaterThan(0)).toBe(true)
  })
})

describe('estimatePerformance - MoE active parameters', () => {
  const moe36bA3b: Model = {
    id: 'test-moe-36b-a3b',
    name: 'Test MoE 36B A3B',
    architecture: 'moe',
    num_parameters_billion: 36,
    hidden_size: 2048,
    num_hidden_layers: 40,
    num_attention_heads: 16,
    num_kv_heads: 2,
    intermediate_size: 512,
    num_experts: 256,
    num_experts_per_token: 8,
    active_parameters_billion: 3,
  }

  it('decode throughput uses active parameters, not the total', () => {
    const result = estimatePerformance({
      model: moe36bA3b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })

    // bandwidth 3350 GB/s / (3B active x 2 bytes) — orders of magnitude above
    // the 3350e9 / (36e9 x 2) = ~46 tok/s the total-parameter formula gave.
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(400)
  })

  it('decode throughput respects weight quantization', () => {
    const fp16 = estimatePerformance({
      model: moe36bA3b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })
    const int4 = estimatePerformance({
      model: moe36bA3b,
      gpu: h100_80gb_sxm,
      quantization: 'gptq',
      sequenceLength: 1024,
      batchSize: 1,
    })

    // Fewer bytes read per token means more tokens per second
    expect(int4.tokensPerSecond.greaterThan(fp16.tokensPerSecond)).toBe(true)
  })
})
```

`llama7b` may not yet exist as a fixture in `performance.test.ts`. If it does not, add it next to `h100_80gb_sxm`:

```ts
const llama7b: Model = {
  id: 'test-llama-7b',
  name: 'Test Llama 7B',
  architecture: 'dense',
  num_parameters_billion: 7,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  intermediate_size: 11008,
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engines/performance.test.ts -t 'prefill model'`
Expected: FAIL — `sequenceLength` is not a valid property on `PerformanceParams`, and `prefillSeconds` does not exist.

- [ ] **Step 3: Add the MFU constant**

Append to `src/engines/constants.ts`:

```ts
/**
 * Model FLOPs Utilization during prefill
 *
 * Prefill is compute-bound and reaches far higher utilization than decode: the
 * literature reports 40-60% MFU for prefill against 1-5% for batch-1 decode.
 * 0.45 sits inside that range.
 *
 * Used as: prefillSeconds = prefillFLOPs / (gpuFLOPS * PREFILL_MFU)
 */
export const PREFILL_MFU = new Decimal(0.45)
```

- [ ] **Step 4: Extend the result type**

In `src/engines/types.ts`, replace the `PerformanceEstimate` interface:

```ts
export interface PerformanceEstimate {
  /** Throughput during decoding phase (tokens/sec) */
  tokensPerSecond: Decimal
  /** Latency to the first output token, in seconds — multiply by 1000 for ms display */
  timeToFirstToken: Decimal
  /** Prompt-processing time in seconds; null when the GPU has no FLOPS data */
  prefillSeconds: Decimal | null
  /** Which prefill term dominates: the linear 2*N*T term or the quadratic attention term */
  prefillBottleneck: 'linear' | 'attention'
  /** True when TTFT fell back to the pre-prefill-model heuristic for lack of FLOPS data */
  prefillEstimateDegraded: boolean
  /** True if performance limited by TFLOPS (small batch, short sequence) */
  isComputeBound: boolean
  /** True if performance limited by memory bandwidth (large batch, long sequence) */
  isMemoryBound: boolean
  /** Primary bottleneck classification */
  bottleneck: 'compute' | 'memory' | 'balanced'
}
```

- [ ] **Step 5: Rewrite the engine**

In `src/engines/performance.ts`:

Imports — add `PREFILL_MFU` and `calculateMoEActiveParams`:

```ts
import { BYTES_PER_GB, PREFILL_MFU } from './constants'
import { calculateMoEActiveParams } from './inference'
import { calculateModelWeightVRAM } from './quantization'
```

`PerformanceParams` — add the required field:

```ts
  /** Number of concurrent sequences (batch size) */
  batchSize: number
  /** Prompt length in tokens — drives prefill time and therefore TTFT */
  sequenceLength: number
```

Body — destructure `sequenceLength`, then replace steps 1–3:

```ts
  const { model, gpu, quantization, sequenceLength, batchSize, multiGPUResult } = params

  // 1. Bytes read per decode token. MoE decode touches only the active experts, so
  //    this uses active params — but it must still route through the quantization
  //    helper, because bytes depend on precision. Never hardcode `x 2` here.
  const activeParams = calculateMoEActiveParams(model)
  const modelSizeGB = calculateModelWeightVRAM(activeParams, quantization)
  const modelSizeBytes = modelSizeGB.mul(BYTES_PER_GB)

  // 2. Memory-bound tokens/sec (dominant for LLM inference)
  const bandwidthBytesPerSec = new Decimal(gpu.memory_bandwidth_gbps).mul(1e9)
  const memoryBoundTPS = bandwidthBytesPerSec.div(modelSizeBytes).mul(batchSize)

  // 3. Compute-bound tokens/sec. FLOPs are precision-independent: ~2 FLOPs per
  //    active parameter (one multiply, one add).
  const flopsPerToken = new Decimal(activeParams).mul(2e9)
```

Leave steps 4, 4b and 5 (roofline min, multi-GPU scaling, bottleneck classification) exactly as they are.

Replace step 6 (`const timeToFirstToken = new Decimal(1).div(tokensPerSecond.mul(0.5))`) with:

```ts
  // 6. Prefill model. Prefill is compute-bound — a different roofline regime from the
  //    bandwidth-bound decode above. Two terms:
  //      linear:    2 * activeParams * T          (precision-independent FLOPs)
  //      attention: 2 * layers * T^2 * hidden     (1/2 * 4 * T^2 * D * L, causal)
  //    Batch is NOT applied: TTFT is a per-request latency for one sequence of T tokens.
  const promptTokens = new Decimal(sequenceLength)
  const linearFLOPs = new Decimal(activeParams).mul(2e9).mul(promptTokens)
  const attentionFLOPs = new Decimal(2)
    .mul(model.num_hidden_layers)
    .mul(promptTokens.pow(2))
    .mul(model.hidden_size)

  const prefillBottleneck: 'linear' | 'attention' = attentionFLOPs.greaterThan(linearFLOPs)
    ? 'attention'
    : 'linear'

  let prefillSeconds: Decimal | null = null
  let prefillEstimateDegraded = false
  let timeToFirstToken: Decimal

  if (gpu.fp16_tflops !== undefined || gpu.fp32_tflops !== undefined) {
    const gpuTFLOPS = gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0
    let effectiveFLOPS = new Decimal(gpuTFLOPS).mul(1e12).mul(PREFILL_MFU)

    if (multiGPUResult && multiGPUResult.numGPUs > 1) {
      effectiveFLOPS = effectiveFLOPS
        .mul(multiGPUResult.numGPUs)
        .mul(multiGPUResult.scalingEfficiency)
    }

    prefillSeconds = linearFLOPs.add(attentionFLOPs).div(effectiveFLOPS)
    timeToFirstToken = prefillSeconds.add(new Decimal(1).div(tokensPerSecond))
  } else {
    // No FLOPS data: prefill time is not computable. Fall back to the previous
    // heuristic rather than returning Infinity, and mark the estimate degraded.
    prefillEstimateDegraded = true
    timeToFirstToken = new Decimal(1).div(tokensPerSecond.mul(0.5))
  }

  return {
    tokensPerSecond,
    timeToFirstToken,
    prefillSeconds,
    prefillBottleneck,
    prefillEstimateDegraded,
    isMemoryBound,
    isComputeBound,
    bottleneck,
  }
```

Update the function's JSDoc: the paragraph claiming "**TTFT (Time To First Token)** is estimated as 2x slower than decode" is now false. Replace it with a description of the two-term prefill model, and update the `@example` TTFT figure comment to `// perf.prefillBottleneck === 'linear'`.

- [ ] **Step 6: Run the engine tests**

Run: `npx vitest run src/engines/performance.test.ts`
Expected: the new tests PASS. Pre-existing tests that construct `estimatePerformance({...})` without `sequenceLength` now fail to typecheck — add `sequenceLength: 2048` to each, which keeps their decode assertions in the same regime. Pre-existing tests asserting an exact `timeToFirstToken` must be recomputed against the new formula.

- [ ] **Step 7: Thread sequenceLength through the worker**

In `src/workers/calculation.worker.ts`:

Add `sequenceLength` to the `estimatePerformance` call (step 4 of the handler):

```ts
      const performance = estimatePerformance({
        model,
        gpu,
        quantization,
        sequenceLength,
        batchSize,
        multiGPUResult,
      })
```

Extend `CalculationSuccessResponse['payload']['performance']`:

```ts
    performance: {
      tokensPerSecond: string
      timeToFirstToken: string
      prefillSeconds: string | null
      prefillBottleneck: 'linear' | 'attention'
      prefillEstimateDegraded: boolean
      isComputeBound: boolean
      isMemoryBound: boolean
      bottleneck: 'compute' | 'memory' | 'balanced'
    }
```

And extend the serialization block:

```ts
          performance: {
            tokensPerSecond: performance.tokensPerSecond.toString(),
            timeToFirstToken: performance.timeToFirstToken.toString(),
            prefillSeconds: performance.prefillSeconds?.toString() ?? null,
            prefillBottleneck: performance.prefillBottleneck,
            prefillEstimateDegraded: performance.prefillEstimateDegraded,
            isComputeBound: performance.isComputeBound,
            isMemoryBound: performance.isMemoryBound,
            bottleneck: performance.bottleneck,
          },
```

- [ ] **Step 8: Thread it through the hook**

In `src/hooks/useInferenceCalculation.ts`, replace `reconstructPerformanceEstimate`:

```ts
function reconstructPerformanceEstimate(serialized: {
  tokensPerSecond: string
  timeToFirstToken: string
  prefillSeconds: string | null
  prefillBottleneck: 'linear' | 'attention'
  prefillEstimateDegraded: boolean
  isComputeBound: boolean
  isMemoryBound: boolean
  bottleneck: 'compute' | 'memory' | 'balanced'
}): PerformanceEstimate {
  return {
    tokensPerSecond: new Decimal(serialized.tokensPerSecond),
    timeToFirstToken: new Decimal(serialized.timeToFirstToken),
    prefillSeconds:
      serialized.prefillSeconds === null ? null : new Decimal(serialized.prefillSeconds),
    prefillBottleneck: serialized.prefillBottleneck,
    prefillEstimateDegraded: serialized.prefillEstimateDegraded,
    isComputeBound: serialized.isComputeBound,
    isMemoryBound: serialized.isMemoryBound,
    bottleneck: serialized.bottleneck,
  }
}
```

And in the sync fallback path, add `sequenceLength` to the `estimatePerformance` call:

```ts
        const performance = performanceModule.estimatePerformance({
          model,
          gpu: effectiveGPU,
          quantization,
          sequenceLength,
          batchSize,
          multiGPUResult: multiGPU,
        })
```

- [ ] **Step 9: Fix the comparison-store unit bug**

`comparisonStore.ts:34` declares milliseconds; `ResultsPanel.tsx:323` writes seconds; `ComparisonColumn.tsx:300` renders the value as ms. Comparison TTFT is therefore 1000× too small.

First add the failing test to `src/store/comparisonStore.test.ts`:

The file already defines `makeSnapshot(label)` at module scope, returning a full snapshot with `results.timeToFirstToken: 22`. Add, inside the existing `describe('comparisonStore', ...)`:

```ts
  it('stores TTFT in milliseconds, not seconds', () => {
    const base = makeSnapshot('ttft units')
    useComparisonStore.getState().addSnapshot({
      ...base,
      results: { ...base.results, timeToFirstToken: 527 },
    })

    const stored = useComparisonStore.getState().snapshots[0]
    expect(stored?.results.timeToFirstToken).toBe(527)
  })
```

The store performs no conversion, so this test documents the contract that `ResultsPanel` must satisfy; the real fix is the `.mul(1000)` below.

Then fix the producer in `src/components/layout/ResultsPanel.tsx:323`:

```ts
        timeToFirstToken: result.performance.timeToFirstToken.mul(1000).toNumber(),
```

And tighten the comment in `src/store/comparisonStore.ts:34`:

```ts
    timeToFirstToken: number // ms — producers must multiply the engine's seconds by 1000
```

`src/components/comparison/ComparisonColumn.tsx:300` renders the field as
`{snapshot.results.timeToFirstToken.toFixed(0)} ms` and becomes correct once the producer
is fixed. Leave it unchanged.

- [ ] **Step 10: Full verification**

Run: `npx vitest run`
Expected: PASS.

Run: `rtk proxy npm run typecheck`
Expected: no errors.

Run: `rtk proxy npm run lint`
Expected: no errors.

- [ ] **Step 11: Commit**

```bash
rtk git add src/engines/constants.ts src/engines/types.ts src/engines/performance.ts \
  src/engines/performance.test.ts src/workers/calculation.worker.ts \
  src/hooks/useInferenceCalculation.ts src/components/layout/ResultsPanel.tsx \
  src/store/comparisonStore.ts src/store/comparisonStore.test.ts
rtk git commit -m "fix(engines): model prefill for TTFT and use active params for throughput

TTFT was 1/(tokensPerSecond * 0.5) — no sequence dependence at all, so a 1M
token prompt reported the same latency as a 10 token one. Replace it with a
compute-bound prefill model: a linear 2*N*T term plus a causal quadratic
attention term, divided by gpuFLOPS * PREFILL_MFU.

Decode throughput divided bandwidth by TOTAL parameters, making every MoE
model report roughly its expert-count ratio too slow — 3.8 tok/s instead of
~45 for Qwen3.6 35B A3B. It now uses active parameters, still routed through
calculateModelWeightVRAM so weight quantization is respected.

Also fixes a unit bug found while tracing TTFT: ResultsPanel wrote seconds
into a comparison-snapshot field declared and rendered as milliseconds.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Raise the sequence-length caps

**Files:**
- Modify: `src/utils/schemas.ts` (add `MAX_SEQUENCE_LENGTH`; `TrainingInputSchema.sequenceLength` at line 119)
- Modify: `src/engines/types.ts:87,92` (`CalculationInputSchema.sequenceLength` and its doc comment)
- Test: `src/store/urlSerializer.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `MAX_SEQUENCE_LENGTH: number` (= 10_485_760) exported from `src/utils/schemas.ts`.

- [ ] **Step 1: Write the failing test**

`src/store/urlSerializer.test.ts` builds its ~30-field state inline inside the first round-trip test. Hoist that object literal to module scope as `const baseState = { ... }`, keeping the per-field `as const` annotations it already carries, and have the original test pass `baseState` instead of its inline copy. Then append:

```ts
describe('sequence length round-trip at long context', () => {
  it('round-trips 1M tokens', () => {
    const deserialized = deserializeFromURL(
      serializeToURL({ ...baseState, sequenceLength: 1048576 }),
    )
    expect(deserialized?.sl).toBe(1048576)
  })

  it('round-trips the maximum sequence length', () => {
    const deserialized = deserializeFromURL(
      serializeToURL({ ...baseState, sequenceLength: 10485760 }),
    )
    expect(deserialized?.sl).toBe(10485760)
  })
})
```

Note the deserialized shape uses the short URL keys (`sl`, not `sequenceLength`) — this matches the assertions already in that file.

Also append to `src/engines/kv-cache.test.ts`:

```ts
it('scales linearly to 1M tokens', () => {
  const at128k = calculateKVCacheVRAM({
    model: llama70bGQA,
    sequenceLength: 131072,
    batchSize: 1,
    kvPrecision: 'fp16',
  })
  const at1m = calculateKVCacheVRAM({
    model: llama70bGQA,
    sequenceLength: 1048576,
    batchSize: 1,
    kvPrecision: 'fp16',
  })

  expect(at1m.div(at128k).toNumber()).toBeCloseTo(8, 9)
  expect(at1m.toNumber()).toBeGreaterThan(100)
})
```

`llama70bGQA` is the fixture name that file already uses (80 layers, hidden 8192, 8 KV heads of 64). Expected magnitudes: ~40 GiB at 128K, ~320 GiB at 1M — KV cache dwarfs the fp16 weights at this scale, which is the point of the assertion.

- [ ] **Step 2: Run tests to verify they pass or fail**

Run: `npx vitest run src/store/urlSerializer.test.ts src/engines/kv-cache.test.ts`
Expected: the KV-cache test PASSES immediately (the engine has no cap); the URL round-trip tests fail **only if** the serializer validates against a capped schema. If they already pass, that is the correct outcome — they are regression locks, keep them.

- [ ] **Step 3: Add the shared constant**

At the top of `src/utils/schemas.ts`, after the zod import:

```ts
/**
 * Maximum expressible sequence length, in tokens
 *
 * Set by the largest context window in the model database (Llama 4 Scout,
 * 10,485,760). This is the bound on what the calculator can *express*, not on what
 * is advisable — the UI marks each model's native context separately and warns
 * rather than clamping, because RoPE/YaRN extension beyond native context is a real
 * workload.
 */
export const MAX_SEQUENCE_LENGTH = 10_485_760
```

- [ ] **Step 4: Raise both caps**

In `src/utils/schemas.ts`, `TrainingInputSchema` (line ~119):

```ts
  /** Sequence length for training */
  sequenceLength: z.number().int().min(512).max(MAX_SEQUENCE_LENGTH),
```

In `src/engines/types.ts`, add the import and raise `CalculationInputSchema` (line ~92):

```ts
import { MAX_SEQUENCE_LENGTH } from '@utils/schemas'
```

```ts
  /** Maximum sequence length (prompt + generation) */
  sequenceLength: z.number().int().min(512).max(MAX_SEQUENCE_LENGTH),
```

And update the doc comment at line 87 from `- Sequence length: 512 to 131072 (128K context)` to:

```
 * - Sequence length: 512 to 10,485,760 (10M context — the largest in the model database)
```

`schemas.ts` imports only `zod`, so `types.ts → schemas.ts` introduces no cycle.

Do **not** touch `src/utils/schemas.js` or `src/utils/schemas.d.ts` — gitignored build output.

- [ ] **Step 5: Verify**

Run: `npx vitest run src/store/urlSerializer.test.ts src/engines/kv-cache.test.ts`
Expected: PASS.

Run: `rtk proxy npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
rtk git add src/utils/schemas.ts src/engines/types.ts src/store/urlSerializer.test.ts \
  src/engines/kv-cache.test.ts
rtk git commit -m "feat(schemas): raise sequence length cap to 10,485,760 tokens

Nine models in the database already declare a context window of 1M or more,
and Llama 4 Scout declares 10,485,760, but both sequence-length schemas capped
at 131,072. Introduce a shared MAX_SEQUENCE_LENGTH and apply it to
CalculationInputSchema and TrainingInputSchema.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Sequence-length slider — dynamic maximum, native-context marker, warning

**Files:**
- Modify: `src/components/inputs/SequenceLengthInput.tsx` (whole file)
- Test: `src/components/inputs/SequenceLengthInput.test.tsx` (create if absent)

**Interfaces:**
- Consumes: `MAX_SEQUENCE_LENGTH` from Task 4; `selectedModel` from `useUIStore` (field exists, `uiStore.ts:39`).
- Produces: no exported API beyond the component.

- [ ] **Step 1: Write the failing test**

Create or extend `src/components/inputs/SequenceLengthInput.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { useUIStore } from '@store/uiStore'
import { SequenceLengthInput } from './SequenceLengthInput'

const model128k = {
  id: 'm-128k',
  name: 'Model 128K',
  architecture: 'dense' as const,
  num_parameters_billion: 7,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  intermediate_size: 11008,
  context_length: 131072,
}

describe('SequenceLengthInput', () => {
  beforeEach(() => {
    useUIStore.setState({ selectedModel: model128k, sequenceLength: 4096 })
  })

  it('formats 1M tokens as M, not as 1024K', () => {
    useUIStore.setState({ sequenceLength: 1048576 })
    render(<SequenceLengthInput />)
    expect(screen.getByText(/1M tokens/)).toBeInTheDocument()
    expect(screen.queryByText(/1024K/)).not.toBeInTheDocument()
  })

  it('offers a 1M preset', () => {
    render(<SequenceLengthInput />)
    expect(screen.getByRole('button', { name: '1M' })).toBeInTheDocument()
  })

  it('warns when the sequence exceeds the model native context', () => {
    useUIStore.setState({ sequenceLength: 262144 })
    render(<SequenceLengthInput />)
    expect(screen.getByText(/beyond native context/i)).toBeInTheDocument()
  })

  it('does not warn at or below the model native context', () => {
    useUIStore.setState({ sequenceLength: 131072 })
    render(<SequenceLengthInput />)
    expect(screen.queryByText(/beyond native context/i)).not.toBeInTheDocument()
  })

  it('extends its range for a model whose native context exceeds 1M', () => {
    useUIStore.setState({
      selectedModel: { ...model128k, id: 'm-10m', context_length: 10485760 },
    })
    render(<SequenceLengthInput />)
    const slider = screen.getByRole('slider', { name: /sequence length/i })
    expect(Number(slider.getAttribute('max'))).toBeCloseTo(Math.log2(10485760), 3)
  })
})
```

If the store's `persist` middleware breaks under jsdom, mock it with `vi.hoisted()` + `vi.mock()` to build a plain store — see the pattern already used elsewhere in this repo's store tests.

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/inputs/SequenceLengthInput.test.tsx`
Expected: FAIL — no 1M preset, `1024K tokens` rendered, no warning text, slider max fixed at 17.

- [ ] **Step 3: Rewrite the component**

Replace the top of `src/components/inputs/SequenceLengthInput.tsx` down to the end of `formatValue`:

```tsx
import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'
import { MAX_SEQUENCE_LENGTH } from '@utils/schemas'

// Preset values for quick selection
const PRESETS = [
  { value: 512, label: '512' },
  { value: 2048, label: '2K' },
  { value: 8192, label: '8K' },
  { value: 32768, label: '32K' },
  { value: 131072, label: '128K' },
  { value: 262144, label: '256K' },
  { value: 524288, label: '512K' },
  { value: 1048576, label: '1M' },
]

// Slider is log2-scaled. The floor is fixed; the ceiling is 1M for almost every model,
// extended only for a model whose native context is larger (Llama 4 Scout at 10M).
const MIN_LOG = 9 // log2(512)
const DEFAULT_MAX_TOKENS = 1_048_576

function formatTokens(value: number): string {
  if (value >= 1_048_576) {
    const mValue = value / 1_048_576
    return `${mValue.toFixed(mValue % 1 === 0 ? 0 : 1)}M`
  }
  if (value >= 1024) {
    const kValue = value / 1024
    return `${kValue.toFixed(kValue % 1 === 0 ? 0 : 1)}K`
  }
  return value.toLocaleString()
}

export function SequenceLengthInput() {
  const { sequenceLength, setSequenceLength, selectedModel } = useUIStore()

  const nativeContext = selectedModel?.context_length
  const maxTokens = Math.min(
    Math.max(DEFAULT_MAX_TOKENS, nativeContext ?? 0),
    MAX_SEQUENCE_LENGTH,
  )
  const maxLog = Math.log2(maxTokens)

  // The user's value is never rewritten — not on model change, not when it exceeds the
  // model's native context. RoPE/YaRN extension is a real workload, custom models carry
  // no context_length at all, and silently clamping would destroy a shared URL.
  const sliderValue = Math.min(Math.log2(sequenceLength), maxLog)
  const exceedsNative = nativeContext !== undefined && sequenceLength > nativeContext

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const logValue = Number.parseFloat(e.target.value)
    setSequenceLength(Math.round(2 ** logValue))
  }

  const formatValue = (value: number): string => `${formatTokens(value)} tokens`

  // Position of the native-context marker along the log2 track, as a percentage
  const nativeMarkerPercent =
    nativeContext !== undefined && nativeContext >= 512 && nativeContext <= maxTokens
      ? ((Math.log2(nativeContext) - MIN_LOG) / (maxLog - MIN_LOG)) * 100
      : null
```

Then, in the JSX:

- Replace the hardcoded right-hand label `<span ...>128K</span>` with `{formatTokens(maxTokens)}`.
- Change the range input's `max={MAX_LOG}` to `max={maxLog}`.
- Wrap the range input in a relatively-positioned container and add the marker before it:

```tsx
        <div className="relative">
          {nativeMarkerPercent !== null && (
            <div
              className="absolute top-0 h-2 w-0.5 bg-gray-500 dark:bg-gray-300 pointer-events-none"
              style={{ left: `${nativeMarkerPercent}%` }}
              aria-hidden="true"
              title={`Native context: ${formatTokens(nativeContext ?? 0)}`}
            />
          )}
          <input
            type="range"
            id="sequence-length"
            min={MIN_LOG}
            max={maxLog}
            step={0.1}
            value={sliderValue}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
            aria-label="Sequence length"
            aria-valuetext={formatValue(sequenceLength)}
          />
        </div>
```

- Add the warning badge immediately after the preset button row:

```tsx
      {exceedsNative && (
        <p
          role="status"
          className="text-xs text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-md px-2 py-1.5"
        >
          Beyond native context ({formatTokens(nativeContext ?? 0)}) — requires RoPE scaling
          / YaRN. The estimate still computes.
        </p>
      )}
```

- Update the `InfoTip` text to mention the 1M range: `"... Common values: 2K (chat), 8K (documents), 128K-1M (long context). Values above the model's native context require RoPE scaling."`

Delete the now-unused `MAX_LOG` constant — Biome treats unused variables as errors.

- [ ] **Step 4: Run the component tests**

Run: `npx vitest run src/components/inputs/SequenceLengthInput.test.tsx`
Expected: PASS (5 tests).

- [ ] **Step 5: Lint and typecheck**

Run: `rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
rtk git add src/components/inputs/SequenceLengthInput.tsx \
  src/components/inputs/SequenceLengthInput.test.tsx
rtk git commit -m "feat(ui): sequence slider up to 1M with native-context marker

Slider maximum becomes max(1M, model context_length), presets gain 256K/512K/1M,
and 1048576 now renders as 1M rather than 1024K. The selected model's native
context appears as a marker on the track, and exceeding it shows a warning
instead of clamping — RoPE/YaRN extension is a real workload, custom models
have no context_length, and clamping on model change would destroy a shared URL.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Surface prefill time in the results panel

**Files:**
- Modify: `src/components/layout/ResultsPanel.tsx:584-615` (Performance Estimate grid)
- Modify: `src/utils/exportPptx.ts:339,350` (TTFT formatting)

**Interfaces:**
- Consumes: `prefillSeconds`, `prefillBottleneck`, `prefillEstimateDegraded` from Task 3.
- Produces: no exported API.

- [ ] **Step 1: Add the formatting helper**

`ResultsPanel.tsx` does not currently import `Decimal` — add the type import at the top:

```tsx
import type Decimal from 'decimal.js'
```

Then, near the other helpers:

```tsx
/**
 * Format a duration given in seconds.
 *
 * A 1M-token prefill takes tens of seconds; rendering that as "45230.0 ms" is
 * unreadable, so switch units at one second.
 */
function formatDuration(seconds: Decimal): string {
  const ms = seconds.mul(1000)
  if (ms.lessThan(1000)) return `${ms.toFixed(1)} ms`
  return `${seconds.toFixed(2)} s`
}
```

- [ ] **Step 2: Use it for TTFT and add the prefill tile**

Replace the "Time to First Token" tile body:

```tsx
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {formatDuration(result.performance.timeToFirstToken)}
                </p>
                {result.performance.prefillEstimateDegraded && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    rough estimate — no FLOPS data for this GPU
                  </p>
                )}
```

Add a fourth tile after "Bottleneck" (and change the grid class from `sm:grid-cols-3` to `sm:grid-cols-4`):

```tsx
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Prompt Processing</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {result.performance.prefillSeconds
                    ? formatDuration(result.performance.prefillSeconds)
                    : 'n/a'}
                </p>
                {result.performance.prefillSeconds && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {result.performance.prefillBottleneck === 'attention'
                      ? 'attention-dominated (quadratic)'
                      : 'weight-dominated (linear)'}
                  </p>
                )}
              </div>
```

Apply `formatDuration` to the per-user TTFT block too (it currently ends with `.mul(1000).toFixed(1)} ms`):

```tsx
                      {formatDuration(
                        result.performance.timeToFirstToken
                          .mul(concurrentUsers)
                          .div(batchSize),
                      )}
```

- [ ] **Step 3: Match the export**

In `src/utils/exportPptx.ts`, replace the fixed-ms TTFT line (~339):

```ts
  const ttftSeconds = performance.timeToFirstToken
  const ttftLabel = ttftSeconds.mul(1000).lessThan(1000)
    ? `${ttftSeconds.mul(1000).toFixed(1)} ms`
    : `${ttftSeconds.toFixed(2)} s`
```

and use `ttftLabel` at the two places that currently interpolate `ttftMs`.

- [ ] **Step 4: Verify in the browser**

Run: `npm run dev`

Check by hand: select a dense model with a GPU that has `fp16_tflops`, set the sequence to 1M, and confirm the panel shows a multi-second "Prompt Processing" value labelled *attention-dominated*, and that at 2K it shows a sub-second value labelled *weight-dominated*.

**Expect MoE throughput to be short of its final value at this point.** Task 7 has not run yet, so every MoE model is still on the tier-2 derivation: Qwen3.6 35B A3B resolves to 4.79B active, giving roughly 350 tok/s rather than the ~558 it will show once its verified `active_parameters_billion: 3` lands. That gap is expected here and is not a defect in the prefill work.

- [ ] **Step 5: Verify the suite**

Run: `npx vitest run && rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: PASS, no errors.

- [ ] **Step 6: Commit**

```bash
rtk git add src/components/layout/ResultsPanel.tsx src/utils/exportPptx.ts
rtk git commit -m "feat(ui): show prompt-processing time and its dominant term

TTFT rendered as fixed milliseconds, which reads as '45230.0 ms' at 1M context.
Switch to adaptive units and add a Prompt Processing tile showing whether the
linear weight term or the quadratic attention term dominates — the distinction
that makes long context interesting.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Populate `active_parameters_billion` for the 33 MoE models

This is a data task. Correctness comes from verification, not from recall.

**Files:**
- Modify: `src/data/models.json` (33 MoE entries)
- Modify: `scripts/fetch-models.ts`

**Interfaces:**
- Consumes: the `active_parameters_billion` schema field from Task 2.
- Produces: verified data; no code API.

- [ ] **Step 1: Add the 11 values encoded in model names**

These need no lookup — the model's own name states the active parameter count. Add `active_parameters_billion` to each entry in `src/data/models.json`:

| `name` | `active_parameters_billion` |
|---|---|
| Gemma 4 26B A4B | 4 |
| Kimi Linear 48B A3B | 3 |
| Llama 4 Maverick 17B 128E | 17 |
| Llama 4 Scout 17B 16E | 17 |
| Nemotron 3 Nano 30B A3B | 3 |
| Nemotron 3 Super 120B A12B | 12 |
| Nemotron 3 Ultra 550B A55B | 55 |
| Nemotron 3.5 Lightning 30B A3B | 3 |
| Qwen3 235B A22B | 22 |
| Qwen3.6 35B A3B | 3 |
| Qwen3.8 2.4T A95B | 95 |

- [ ] **Step 2: Verify the remaining 22 one at a time**

For each model below, open its `hf_url` (already present on the entry) via the Hugging Face MCP — `hub_repo_details` for the model card, `hf_fs` to `cat` its `config.json` — and read the active parameter count from the card. Where the card does not state it, compute it from `config.json` using `num_experts_per_tok`, `num_local_experts` (or the model's equivalent field names) and `moe_intermediate_size`, and record in the commit message which models were computed rather than quoted.

DeepSeek R1 · DeepSeek V4 Flash · DeepSeek V4 Pro · GLM 4.7 · GLM 4.7 Flash · GLM 5.2 · GPT OSS 20B · GPT OSS 120B · Kimi K2 Instruct · Kimi K2 Thinking · Kimi K2.5 · Kimi K2.6 · Kimi K2.7 Code · Kimi K3 · Ling 3.0 Tiny · Ling 3.0 Flash · MiniMax M2.1 · MiniMax M2.5 · MiniMax M2.7 · MiniMax M3 · Mistral Large 3 675B · Mistral Small 4 119B

**When verification fails** — some of these (Kimi K3 at 2,779.9B, Qwen3.8 2.4T, DeepSeek V4 Pro) may have no reachable public `config.json` — leave `active_parameters_billion` **absent** for that model and let the tier-2 derivation handle it. Do not guess. Do not block the other entries. List the models left on tier 2 in the commit message.

**Also report, but do not change:** `Mistral Small 4 119B` has `num_kv_heads: 32` against `num_attention_heads: 32`, meaning no GQA reduction at all — suspect for a recent Mistral. Its `config.json` is open anyway during this step. Note the actual value in the commit message; changing it is a separate decision.

- [ ] **Step 3: Re-sort and validate**

`src/data/models.json` must stay sorted alphabetically by `name`:

```bash
npx tsx -e "const fs=require('fs');const p='src/data/models.json';const m=JSON.parse(fs.readFileSync(p,'utf8'));m.sort((a,b)=>a.name.localeCompare(b.name));fs.writeFileSync(p,JSON.stringify(m,null,2)+'\n')"
```

Then confirm every entry still parses against the schema:

```bash
npx tsx -e "import('./src/utils/schemas.ts').then(async s=>{const m=(await import('./src/data/models.json',{with:{type:'json'}})).default;s.validateModels(m);console.log('ok',m.length)})"
```

Expected: `ok 54`.

- [ ] **Step 4: Add the refresh-script divergence check**

In `scripts/fetch-models.ts`, after a model is built, warn when a stored value and the derivation disagree:

```ts
/**
 * Warn when a curated active_parameters_billion diverges from what the model's own
 * dimensions imply. Catches a stale hand-entered value on refresh without ever
 * overwriting a verified one.
 */
function checkActiveParamsConsistency(model: Model): void {
  if (!model.active_parameters_billion) return
  if (!model.num_experts || !model.num_experts_per_token) return

  const expertParams =
    (model.num_hidden_layers * model.num_experts * 3 * model.hidden_size * model.intermediate_size) /
    1e9
  const nonExpert = Math.max(model.num_parameters_billion - expertParams, 0)
  const derived = nonExpert + expertParams * (model.num_experts_per_token / model.num_experts)

  const divergence = Math.abs(derived - model.active_parameters_billion) / model.active_parameters_billion
  if (divergence > 0.25) {
    console.warn(
      `WARN ${model.name}: active_parameters_billion=${model.active_parameters_billion} ` +
        `but dimensions imply ~${derived.toFixed(1)}B (${(divergence * 100).toFixed(0)}% apart)`,
    )
  }
}
```

Call it for each fetched model, and carry `active_parameters_billion` through from any existing entry so a refresh does not drop verified values.

- [ ] **Step 5: Verify**

Run: `npx vitest run && rtk proxy npm run lint && rtk proxy npm run typecheck`
Expected: PASS, no errors.

**Re-check any exact-value MoE activation assertion here.** `calculateActivationMemory` scales `intermediate_size` by `activeParams / total`, so an MoE activation figure moves **twice** across this plan: once in Task 2 when the 20/80 heuristic became the tier-2 derivation, and again here when tier 1 takes over for the models given a verified value. A test Task 2 just corrected may need correcting again — recompute from the new ratio rather than widening the assertion.

Then spot-check the engine end to end:

```bash
npx tsx -e "
import('./src/engines/inference.ts').then(async i => {
  const models = (await import('./src/data/models.json', { with: { type: 'json' } })).default
  const m = models.find(x => x.id === 'qwen-qwen3.6-35b-a3b')
  console.log('active:', i.calculateMoEActiveParams(m))
})"
```

Expected: `active: 3` (the value entered in Step 1), not 4.79 and certainly not 8.1.

- [ ] **Step 6: Commit**

```bash
rtk git add src/data/models.json scripts/fetch-models.ts
rtk git commit -m "data(models): add verified active parameter counts for MoE models

11 values are stated in the model names themselves; the rest were read from
each model's Hugging Face card or computed from its config.json. Models whose
config was unreachable are left without the field and fall back to the derived
estimate.

Adds a refresh-time warning when a curated value diverges from what the model's
own dimensions imply by more than 25%.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Documentation

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `src/components/guide/GuidePage.tsx` (TTFT copy at lines ~392, ~561, ~584)

**Interfaces:**
- Consumes: everything above.
- Produces: no code API.

- [ ] **Step 1: Fix the in-app guide**

Three places in `src/components/guide/GuidePage.tsx` describe TTFT. The characterization that prefill is "2× slower than decode" is now false. Rewrite them to say that TTFT is dominated by prompt processing, that it grows with prompt length, and that beyond roughly 100K tokens the quadratic attention term overtakes the linear weight term. Also note that the sequence-length control now reaches 1M and that going beyond a model's native context requires RoPE scaling.

- [ ] **Step 2: Update the changelog**

Add a new section at the top of `CHANGELOG.md`, matching the file's existing heading style:

```markdown
### Added

- Sequence lengths up to 10,485,760 tokens, with 256K / 512K / 1M presets, a marker for
  the selected model's native context, and a warning (never a clamp) when the requested
  context exceeds it.
- `active_parameters_billion` on MoE models, verified per model.
- Prompt-processing time in the results panel, labelled by whether the linear weight term
  or the quadratic attention term dominates.

### Fixed

- Activation memory scaled with the full context window, reporting 68 GB at 1M tokens for
  a 27B dense model. It is now bounded by the prefill chunk. **VRAM figures above 8,192
  tokens decrease; figures at or below 8,192 are unchanged.**
- Decode throughput divided memory bandwidth by total rather than active parameters,
  making every MoE model report roughly its expert ratio too slow. **MoE tokens/sec
  figures increase, by up to ~9x.**
- Time to first token had no dependence on sequence length. **All TTFT figures change.**
- The comparison view stored seconds in a field rendered as milliseconds, showing 0.53 s
  as "1 ms".
```

- [ ] **Step 3: Update the README**

Wherever the README states the supported context range or describes the performance estimates, update it for the 1M+ range and the prefill model. If it carries a model or GPU count, re-check it (54 models, 24 GPUs — unchanged by this work).

- [ ] **Step 4: Final full verification**

```bash
npx vitest run
rtk proxy npm run typecheck
rtk proxy npm run lint
npm run build
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
rtk git add CHANGELOG.md README.md src/components/guide/GuidePage.tsx
rtk git commit -m "docs: long-context support and the value changes it brings

Records which displayed figures move: activations above 8K, MoE tokens/sec,
every TTFT, and the comparison view's TTFT units.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Known remaining gaps

Called out in the spec as non-goals; do not fix them in this plan.

- `FRAMEWORK_OVERHEAD_GB` stays a flat 1.0 GB, so a ~13.6 GB VRAM gap with
  apxml.com/tools/vram-calculator remains open on the reference configuration.
- `FLASH_ATTENTION_LONG_THRESHOLD = 8192` puts 8K and 1M in the same retention bucket
  (training path only).
- `Mistral Small 4 119B`'s GQA ratio of 1.0 is reported by Task 7, not corrected.
