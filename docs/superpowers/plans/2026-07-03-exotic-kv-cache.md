# Exotic-Attention KV-Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KV-cache VRAM estimates exact for the 16 exotic-attention models (MLA, mamba/linear hybrids, explicit head_dim) via one optional schema field and one engine branch.

**Architecture:** `ModelSchema` gains optional `kv_cache_elements_per_token` (total cached elements per token across all layers). `calculateKVCacheVRAM` uses it directly when present (`elements × seq × batch × precisionBytes`); otherwise the existing GQA formula runs unchanged. Data values are hand-curated from live HuggingFace config.json (verified 2026-07-03) and set on 16 models in `models.json`. Also fixes one data error found during verification: Nemotron Ultra has 108 layers, not 128.

**Tech Stack:** TypeScript strict, Zod, decimal.js, Vitest, Biome.

## Global Constraints

- New schema field: `kv_cache_elements_per_token: z.number().int().positive().optional()` — exact name everywhere.
- Engine override formula: `elements × sequenceLength × batchSize × precisionBytes ÷ BYTES_PER_GB`. The existing GQA path must be byte-identical for models without the field.
- The 16 data values are HF-config-verified and must be used verbatim (table in Task 2).
- `num_kv_heads` values in models.json are NOT changed — the new field takes precedence in the engine only.
- Biome style: 2-space indent, single quotes, no semicolons, 100-char width. `models.json`: 2-space indent + trailing newline; array stays sorted by `name` (no names change in this plan, so order is untouched).
- Environment quirk: if `npm run lint` / `npm run typecheck` / `npx vitest run` fails to launch with an rtk JSON-parse error, prefix with `rtk proxy` (e.g. `rtk proxy npm run lint`) — wrapper quirk, not a real failure.
- Known pre-existing state: `npx vitest run` currently passes 270/270 across 18 files.

---

### Task 1: Schema field + engine override branch (TDD)

**Files:**
- Modify: `src/utils/schemas.ts:63-66` (add field after `num_kv_heads`)
- Modify: `src/engines/kv-cache.ts:41-70` (add override branch + docstring note)
- Test: `src/engines/kv-cache.test.ts` (add fixture + 3 tests)

**Interfaces:**
- Consumes: existing `calculateKVCacheVRAM(params: {model, sequenceLength, batchSize, kvPrecision}): Decimal`, `KV_PRECISION_BYTES` (fp16=2.0, fp8=1.0, int8=1.0, int4=0.5), `BYTES_PER_GB` (1024³).
- Produces: `Model` type now has optional `kv_cache_elements_per_token?: number` (Zod-inferred — Task 2's data and tests rely on this exact field name).

- [ ] **Step 1: Write the failing tests**

In `src/engines/kv-cache.test.ts`, add after the `mqaModel` fixture (before the `describe`):

```typescript
const mlaModel: Model = {
  id: 'test-mla-deepseek-r1',
  name: 'Test DeepSeek R1 (MLA)',
  architecture: 'moe',
  num_parameters_billion: 671,
  hidden_size: 7168,
  num_hidden_layers: 61,
  num_attention_heads: 128,
  num_kv_heads: 128, // nominal dummy value — override field takes precedence
  intermediate_size: 18432,
  num_experts: 256,
  num_experts_per_token: 8,
  // MLA: 61 layers × (kv_lora_rank 512 + qk_rope_head_dim 64) = 35136
  kv_cache_elements_per_token: 35136,
}
```

Add inside the `describe('calculateKVCacheVRAM', ...)` block, before its closing `})`:

```typescript
  it('uses kv_cache_elements_per_token override when present (MLA exact value)', () => {
    const result = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    // 35136 × 4096 × 1 × 2 bytes / 1024³ = 0.26806640625 GB exactly
    expect(result.toNumber()).toBe(0.26806640625)
  })

  it('override is much smaller than the nominal GQA formula for MLA models', () => {
    const nominal: Model = { ...mlaModel, kv_cache_elements_per_token: undefined }
    const overridden = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    const gqaFormula = calculateKVCacheVRAM({
      model: nominal,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    // GQA formula: 2×61×7168 = 874496 elem/token vs MLA 35136 → ~24.9× reduction
    expect(gqaFormula.div(overridden).toNumber()).toBeCloseTo(874496 / 35136, 3)
  })

  it('override path applies KV precision scaling (int8 = half of fp16)', () => {
    const fp16 = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 2,
      kvPrecision: 'fp16',
    })
    const int8 = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 2,
      kvPrecision: 'int8',
    })
    expect(int8.mul(2).toNumber()).toBeCloseTo(fp16.toNumber(), 10)
  })
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engines/kv-cache.test.ts`
Expected: the first new test FAILS (override ignored → GQA formula returns ~6.67 GB, not 0.268), the ratio test FAILS (ratio 1). Note: TypeScript may not error on the unknown fixture field at runtime (vitest transpiles without typecheck); the failures are assertion failures.

- [ ] **Step 3: Add the schema field**

In `src/utils/schemas.ts`, after the `num_kv_heads` line (line 64), add:

```typescript
  // Exotic-attention override: total cached elements per token across ALL layers
  // (MLA latent dims, hybrid attention-layers-only, explicit head_dim). When present,
  // the KV-cache engine uses it directly instead of the GQA formula.
  kv_cache_elements_per_token: z.number().int().positive().optional(),
```

- [ ] **Step 4: Add the engine override branch**

In `src/engines/kv-cache.ts`, at the top of the function body (immediately after the destructuring line `const { model, sequenceLength, batchSize, kvPrecision } = params`), insert:

```typescript
  // Exotic-attention override (MLA, mamba/linear hybrids, explicit head_dim):
  // the model declares its exact per-token cache size; precision applies identically.
  // Constant-state memory (mamba SSM / linear-attention state) does not scale with
  // sequence length and is intentionally excluded.
  if (model.kv_cache_elements_per_token) {
    return new Decimal(model.kv_cache_elements_per_token)
      .mul(sequenceLength)
      .mul(batchSize)
      .mul(KV_PRECISION_BYTES[kvPrecision])
      .div(BYTES_PER_GB)
  }
```

(The existing `precisionBytes` const below stays for the GQA path.) Also extend the function docstring's parameter note: change the `@param params.model` line to end with `..., optionally num_kv_heads, kv_cache_elements_per_token)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/engines/kv-cache.test.ts`
Expected: PASS — all pre-existing tests (GQA/MQA/fallback/precision) plus the 3 new ones.

- [ ] **Step 6: Typecheck, lint, full suite**

Run: `npm run typecheck` → no errors.
Run: `npm run lint` → no errors (`npm run lint:fix` if fixable issues, then re-run).
Run: `npx vitest run` → all tests pass (270 pre-existing + 3 new = 273).

- [ ] **Step 7: Commit**

```bash
git add src/utils/schemas.ts src/engines/kv-cache.ts src/engines/kv-cache.test.ts
git commit -m "feat(engine): kv_cache_elements_per_token override for exotic attention"
```

---

### Task 2: Set verified values on 16 models + fix Nemotron Ultra layer count (TDD)

**Files:**
- Modify: `src/data/models.json` (16 entries gain the field; 1 entry's `num_hidden_layers` corrected)
- Test: `src/utils/models.test.ts` (add invariant test)

**Interfaces:**
- Consumes: `kv_cache_elements_per_token` field from Task 1 (schema accepts it; `validateModels` in the existing test enforces it).
- Produces: final data consumed by the app; no later task depends on specifics.

The canonical 16 values (HF-config-verified 2026-07-03 — use verbatim):

| id | value |
|---|---|
| deepseek-r1 | 35136 |
| deepseek-v4-flash | 24768 |
| deepseek-v4-pro | 35136 |
| moonshotai-kimi-k2-thinking | 35136 |
| moonshotai-kimi-k2-instruct | 35136 |
| moonshotai-kimi-k2.5 | 35136 |
| moonshotai-kimi-linear-48b-a3b | 4032 |
| zai-org-glm-5.2 | 44928 |
| minimax-m3 | 61440 |
| minimax-m2.1 | 126976 |
| minimax-m2.5 | 126976 |
| minimax-m2.7 | 126976 |
| nvidia-nemotron-3-nano-4b | 8192 |
| nvidia-nemotron-3-nano-30b-a3b | 3072 |
| nvidia-nemotron-3-super-120b-a12b | 4096 |
| nvidia-nemotron-3-ultra-550b-a55b | 6144 |

- [ ] **Step 1: Write the failing invariant test**

In `src/utils/models.test.ts`, add immediately before the final closing `})` of the `describe` block:

```typescript
  it('sets kv_cache_elements_per_token on exactly the exotic-attention models', () => {
    const EXOTIC_KV: Record<string, number> = {
      'deepseek-r1': 35136,
      'deepseek-v4-flash': 24768,
      'deepseek-v4-pro': 35136,
      'moonshotai-kimi-k2-thinking': 35136,
      'moonshotai-kimi-k2-instruct': 35136,
      'moonshotai-kimi-k2.5': 35136,
      'moonshotai-kimi-linear-48b-a3b': 4032,
      'zai-org-glm-5.2': 44928,
      'minimax-m3': 61440,
      'minimax-m2.1': 126976,
      'minimax-m2.5': 126976,
      'minimax-m2.7': 126976,
      'nvidia-nemotron-3-nano-4b': 8192,
      'nvidia-nemotron-3-nano-30b-a3b': 3072,
      'nvidia-nemotron-3-super-120b-a12b': 4096,
      'nvidia-nemotron-3-ultra-550b-a55b': 6144,
    }
    for (const [id, expected] of Object.entries(EXOTIC_KV)) {
      const model = modelsData.find((m) => m.id === id) as
        | { kv_cache_elements_per_token?: number }
        | undefined
      expect(model, `model ${id} missing from database`).toBeDefined()
      expect(model?.kv_cache_elements_per_token, `wrong value for ${id}`).toBe(expected)
    }
    const withField = modelsData.filter(
      (m) => (m as { kv_cache_elements_per_token?: number }).kv_cache_elements_per_token,
    )
    expect(withField.length).toBe(Object.keys(EXOTIC_KV).length)
  })

  it('stores the corrected Nemotron Ultra layer count (108, not 128)', () => {
    const ultra = modelsData.find((m) => m.id === 'nvidia-nemotron-3-ultra-550b-a55b')
    expect(ultra?.num_hidden_layers).toBe(108)
  })
```

(The `as` casts are needed because the JSON import's inferred type lacks the optional field until the data carries it.)

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/utils/models.test.ts`
Expected: FAIL — `wrong value for deepseek-r1` (undefined) and the Ultra test (128 ≠ 108).

- [ ] **Step 3: Write the transform script**

Create `/private/tmp/claude-501/-Users-fjacquet-Projects-llmvram/c3125bf3-21c2-43e1-b95d-c1352221f087/scratchpad/set-kv-elements.mjs`:

```javascript
import { readFile, writeFile } from 'node:fs/promises'

const PATH = 'src/data/models.json'

const KV = {
  'deepseek-r1': 35136,
  'deepseek-v4-flash': 24768,
  'deepseek-v4-pro': 35136,
  'moonshotai-kimi-k2-thinking': 35136,
  'moonshotai-kimi-k2-instruct': 35136,
  'moonshotai-kimi-k2.5': 35136,
  'moonshotai-kimi-linear-48b-a3b': 4032,
  'zai-org-glm-5.2': 44928,
  'minimax-m3': 61440,
  'minimax-m2.1': 126976,
  'minimax-m2.5': 126976,
  'minimax-m2.7': 126976,
  'nvidia-nemotron-3-nano-4b': 8192,
  'nvidia-nemotron-3-nano-30b-a3b': 3072,
  'nvidia-nemotron-3-super-120b-a12b': 4096,
  'nvidia-nemotron-3-ultra-550b-a55b': 6144,
}

const models = JSON.parse(await readFile(PATH, 'utf8'))
let applied = 0

for (const m of models) {
  if (Object.hasOwn(KV, m.id)) {
    m.kv_cache_elements_per_token = KV[m.id]
    applied++
  }
  if (m.id === 'nvidia-nemotron-3-ultra-550b-a55b') {
    if (m.num_hidden_layers !== 128 && m.num_hidden_layers !== 108) {
      throw new Error(`Unexpected Ultra layer count: ${m.num_hidden_layers}`)
    }
    m.num_hidden_layers = 108 // layers_block_type has 108 entries (48 mamba + 48 moe + 12 attn)
  }
}

if (applied !== Object.keys(KV).length) {
  throw new Error(`Applied ${applied}, expected ${Object.keys(KV).length}. Check ids.`)
}

await writeFile(PATH, `${JSON.stringify(models, null, 2)}\n`)
console.log(`OK: set kv_cache_elements_per_token on ${applied} models; Ultra layers -> 108`)
```

- [ ] **Step 4: Run the transform**

Run: `node "/private/tmp/claude-501/-Users-fjacquet-Projects-llmvram/c3125bf3-21c2-43e1-b95d-c1352221f087/scratchpad/set-kv-elements.mjs"`
Expected: `OK: set kv_cache_elements_per_token on 16 models; Ultra layers -> 108`
If it throws, STOP and check id spellings — do not hand-edit around the guard.

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/utils/models.test.ts`
Expected: PASS (all, including the two new tests — Zod validation also confirms the new field parses).

- [ ] **Step 6: Typecheck, lint, full suite**

Run: `npm run typecheck` → no errors.
Run: `npm run lint` → no errors.
Run: `npx vitest run` → all pass (275 total: 273 from Task 1 + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/data/models.json src/utils/models.test.ts
git commit -m "feat(data): exact KV-cache elements for 16 exotic-attention models"
```

---

### Task 3: Fetch-script comment + CHANGELOG

**Files:**
- Modify: `scripts/fetch-models.ts:4-8` (extend the NOTE comment)
- Modify: `CHANGELOG.md` (add to `## [Unreleased]`)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed later.

- [ ] **Step 1: Extend the fetch-script comment**

In `scripts/fetch-models.ts`, the `MODEL_IDS` comment block currently ends with:

```typescript
// and hand-curate num_parameters_billion / MoE fields against models.json.
```

Replace that line with:

```typescript
// and hand-curate num_parameters_billion / MoE fields against models.json.
// `kv_cache_elements_per_token` is also hand-curated (MLA latent dims, hybrid
// layer mixes, and explicit head_dim are not derivable from top-level config
// fields) — this script never writes it.
```

- [ ] **Step 2: Add the CHANGELOG entry**

In `CHANGELOG.md`, replace the line `## [Unreleased]` with:

```markdown
## [Unreleased]

### Added

- Exact KV-cache VRAM math for exotic-attention models via a new
  `kv_cache_elements_per_token` model field (verified from HuggingFace `config.json` on
  2026-07-03): MLA (DeepSeek R1/V4, Kimi K2 family, GLM 5.2), linear/mamba hybrids
  (Kimi Linear, Nemotron 3 family), and explicit-head_dim GQA (MiniMax M2.x/M3) —
  16 models. Headline: DeepSeek R1 KV cache at 4K context drops from ~6.7 GB
  (25× overestimate) to ~0.27 GB.

### Fixed

- Nemotron 3 Ultra 550B A55B layer count corrected to 108 (48 mamba + 48 MoE +
  12 attention per `layers_block_type`); previously stored as 128.
```

- [ ] **Step 3: Typecheck + lint (script file changed)**

Run: `npm run typecheck` → no errors.
Run: `npm run lint` → no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch-models.ts CHANGELOG.md
git commit -m "docs: record exotic KV-cache math and Nemotron Ultra layer fix"
```

---

## Self-Review

**Spec coverage:** schema field → Task 1 Step 3; engine branch → Task 1 Step 4; docstring note → Task 1 Step 4; 16 data values verbatim → Task 2 (table + script literal match the spec table); Ultra 128→108 → Task 2 Steps 1/3; engine tests (exact value 0.26806640625, precision scaling, fallback untouched) → Task 1 Step 1 (fallback covered by pre-existing tests remaining green); data invariant test (exact values, field on no other model) → Task 2 Step 1; fetch-script comment → Task 3 Step 1; error handling = Zod-only → no code needed; out-of-scope items (UI, multi-GPU, CSA/HCA modeling) → not present anywhere. ✓

**Placeholder scan:** none — every step carries exact code/commands. ✓

**Type consistency:** field name `kv_cache_elements_per_token` identical in schema (Task 1), engine (Task 1), fixture (Task 1), test map + script (Task 2), comment (Task 3). Test-count arithmetic: 270 baseline + 3 (Task 1) + 2 (Task 2) = 275. ✓

## Notes for the executor

- Do NOT alter any `num_kv_heads` value in models.json — nominal values stay; the engine override wins.
- The models.json array order must not change (no `name` fields change; the script mutates in place and re-serializes in the same order).
- If any test outside `kv-cache.test.ts` / `models.test.ts` fails, stop and report — nothing else should be affected.
