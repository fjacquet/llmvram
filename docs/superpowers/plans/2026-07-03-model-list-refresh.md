# Current-Generation Model List Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh `src/data/models.json` to the July-2026 generation — add 13 verified current-gen flagship LLMs, retire 26 obsolete entries, re-sort alphabetically — and update the dependent test, fetch script, and docs.

**Architecture:** Pure data + docs change. `src/data/models.json` is the single source of truth imported by the app. Each entry is a plain object validated by the Zod `ModelSchema` in `src/utils/schemas.ts`. No engine or component code changes. The only test file that reads `models.json` is `src/utils/models.test.ts`; engine tests use their own inline fixtures and are unaffected.

**Tech Stack:** TypeScript (strict), Vitest, Zod, Biome. Node for the deterministic data transform.

## Global Constraints

- Entries in `src/data/models.json` MUST be sorted by `name` using code-unit order (the comparator `(a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0)`), matching the existing file. Uppercase sorts before lowercase.
- `id` convention: `{org}-{model-slug-lowercased}` (e.g. `qwen-qwen3.6-27b`). All `id`s MUST be unique.
- For MoE models, `num_parameters_billion` is the TOTAL parameter count (all experts), and `num_experts` + `num_experts_per_token` MUST be set.
- Every entry MUST satisfy `ModelSchema`: required `id`, `name`, `architecture` (`'dense'|'moe'`), `num_parameters_billion` (>0), `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size` (all positive ints); optional `num_kv_heads`, `num_experts`, `num_experts_per_token`, `context_length`, `license`, `hf_url`.
- `num_kv_heads` MUST be `<= num_attention_heads`.
- `models.json` is written with 2-space indentation and a trailing newline.
- All config field values in this plan were read from live HuggingFace `config.json` on 2026-07-03; do not alter them.
- Biome: 2-space indent, single quotes, no semicolons, 100-char width. Run `npm run lint:fix` after code edits.

---

## Canonical data: the 13 new entries

This exact JSON is used verbatim in Task 2. (Order here is grouped by family; the transform script re-sorts the whole file.)

```json
[
  { "id": "google-gemma-4-31b", "name": "Gemma 4 31B", "architecture": "dense", "num_parameters_billion": 32.7, "hidden_size": 5376, "num_hidden_layers": 60, "num_attention_heads": 32, "num_kv_heads": 16, "intermediate_size": 21504, "context_length": 262144, "license": "Apache-2.0", "hf_url": "https://huggingface.co/google/gemma-4-31B-it" },
  { "id": "google-gemma-4-12b", "name": "Gemma 4 12B", "architecture": "dense", "num_parameters_billion": 12.0, "hidden_size": 3840, "num_hidden_layers": 48, "num_attention_heads": 16, "num_kv_heads": 8, "intermediate_size": 15360, "context_length": 262144, "license": "Apache-2.0", "hf_url": "https://huggingface.co/google/gemma-4-12B-it" },
  { "id": "google-gemma-4-26b-a4b", "name": "Gemma 4 26B A4B", "architecture": "moe", "num_parameters_billion": 26.5, "hidden_size": 2816, "num_hidden_layers": 30, "num_attention_heads": 16, "num_kv_heads": 8, "intermediate_size": 2112, "num_experts": 128, "num_experts_per_token": 8, "context_length": 262144, "license": "Apache-2.0", "hf_url": "https://huggingface.co/google/gemma-4-26B-A4B-it" },
  { "id": "qwen-qwen3.6-27b", "name": "Qwen3.6 27B", "architecture": "dense", "num_parameters_billion": 27.8, "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24, "num_kv_heads": 4, "intermediate_size": 17408, "context_length": 262144, "license": "Apache-2.0", "hf_url": "https://huggingface.co/Qwen/Qwen3.6-27B" },
  { "id": "qwen-qwen3.6-35b-a3b", "name": "Qwen3.6 35B A3B", "architecture": "moe", "num_parameters_billion": 36.0, "hidden_size": 2048, "num_hidden_layers": 40, "num_attention_heads": 16, "num_kv_heads": 2, "intermediate_size": 512, "num_experts": 256, "num_experts_per_token": 8, "context_length": 262144, "license": "Apache-2.0", "hf_url": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B" },
  { "id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "architecture": "moe", "num_parameters_billion": 158.1, "hidden_size": 4096, "num_hidden_layers": 43, "num_attention_heads": 64, "num_kv_heads": 1, "intermediate_size": 2048, "num_experts": 256, "num_experts_per_token": 6, "context_length": 1048576, "license": "MIT", "hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash" },
  { "id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "architecture": "moe", "num_parameters_billion": 861.6, "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 128, "num_kv_heads": 1, "intermediate_size": 3072, "num_experts": 384, "num_experts_per_token": 6, "context_length": 1048576, "license": "MIT", "hf_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro" },
  { "id": "zai-org-glm-5.2", "name": "GLM 5.2", "architecture": "moe", "num_parameters_billion": 753.3, "hidden_size": 6144, "num_hidden_layers": 78, "num_attention_heads": 64, "num_kv_heads": 64, "intermediate_size": 12288, "num_experts": 256, "num_experts_per_token": 8, "context_length": 1048576, "license": "MIT", "hf_url": "https://huggingface.co/zai-org/GLM-5.2" },
  { "id": "moonshotai-kimi-k2-thinking", "name": "Kimi K2 Thinking", "architecture": "moe", "num_parameters_billion": 1058.1, "hidden_size": 7168, "num_hidden_layers": 61, "num_attention_heads": 64, "num_kv_heads": 64, "intermediate_size": 18432, "num_experts": 384, "num_experts_per_token": 8, "context_length": 262144, "license": "Modified MIT", "hf_url": "https://huggingface.co/moonshotai/Kimi-K2-Thinking" },
  { "id": "moonshotai-kimi-linear-48b-a3b", "name": "Kimi Linear 48B A3B", "architecture": "moe", "num_parameters_billion": 49.1, "hidden_size": 2304, "num_hidden_layers": 27, "num_attention_heads": 32, "num_kv_heads": 32, "intermediate_size": 9216, "num_experts": 256, "num_experts_per_token": 8, "context_length": 1048576, "license": "MIT", "hf_url": "https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct" },
  { "id": "mistralai-mistral-medium-3.5-128b", "name": "Mistral Medium 3.5 128B", "architecture": "dense", "num_parameters_billion": 127.7, "hidden_size": 12288, "num_hidden_layers": 88, "num_attention_heads": 96, "num_kv_heads": 8, "intermediate_size": 28672, "context_length": 262144, "license": "Mistral Research License", "hf_url": "https://huggingface.co/mistralai/Mistral-Medium-3.5-128B" },
  { "id": "minimax-m3", "name": "MiniMax M3", "architecture": "moe", "num_parameters_billion": 427.0, "hidden_size": 6144, "num_hidden_layers": 60, "num_attention_heads": 64, "num_kv_heads": 4, "intermediate_size": 12288, "num_experts": 128, "num_experts_per_token": 4, "context_length": 1048576, "license": "MiniMax", "hf_url": "https://huggingface.co/MiniMaxAI/MiniMax-M3" },
  { "id": "nvidia-nemotron-3-ultra-550b-a55b", "name": "Nemotron 3 Ultra 550B A55B", "architecture": "moe", "num_parameters_billion": 560.5, "hidden_size": 8192, "num_hidden_layers": 128, "num_attention_heads": 64, "num_kv_heads": 2, "intermediate_size": 5120, "num_experts": 512, "num_experts_per_token": 22, "context_length": 262144, "license": "other", "hf_url": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16" }
]
```

## Canonical data: the 26 names to retire

```
LLaMA 2 7B, LLaMA 2 13B, LLaMA 2 70B,
Mistral 7B v0.1, Mistral 7B v0.3,
Mixtral 8x7B v0.1, Mixtral 8x7B Instruct v0.1, Mixtral 8x22B v0.1,
Gemma 2B, Gemma 7B, Gemma 2 9B,
Phi 3 Mini 4K, Phi 3 Small 8K, Phi 3 Medium 4K, Phi 3.5 Mini Instruct,
MPT 30B, Falcon 40B, Yi 34B, LLaMA 3 8B,
DeepSeek Coder 33B, DeepSeek V2 Lite, DeepSeek Coder V2 Lite,
Command-R, Command-R Plus,
Nemotron Llama 3.1 Nano 8B, Nemotron Llama 3.3 Super 49B
```

---

### Task 1: Update the model-database test to the new roster

**Files:**
- Modify: `src/utils/models.test.ts`

**Interfaces:**
- Consumes: `modelsData` (array of `Model`) imported from `@data/models.json`; `ModelSchema`, `validateModels` from `./schemas`.
- Produces: nothing consumed by later tasks. This task redefines the invariants Task 2's data must satisfy.

Rationale: the current test hard-codes presence of retired families (LLaMA 2, Mixtral, Command-R, Phi). After Task 2 those disappear, so those assertions must be re-pointed to current families now, giving us a red suite that Task 2 turns green. We also add an alphabetical-sort invariant (a project rule currently untested) and re-point the MoE-pitfall and GQA fixtures to current models.

- [ ] **Step 1: Replace the four retired-family presence tests**

In `src/utils/models.test.ts`, replace this block:

```typescript
  it('should include LLaMA 2 variants', () => {
    const llama2Models = modelsData.filter((m) => m.name.includes('LLaMA 2'))
    expect(llama2Models.length).toBeGreaterThanOrEqual(3)
  })

  it('should include LLaMA 3 variants', () => {
    const llama3Models = modelsData.filter((m) => m.name.includes('LLaMA 3'))
    expect(llama3Models.length).toBeGreaterThanOrEqual(3)
  })
```

with:

```typescript
  it('should include Gemma 4 variants', () => {
    const gemma4Models = modelsData.filter((m) => m.name.includes('Gemma 4'))
    expect(gemma4Models.length).toBeGreaterThanOrEqual(3)
  })

  it('should include LLaMA 3 / Llama 4 variants', () => {
    const llamaModels = modelsData.filter(
      (m) => m.name.includes('LLaMA 3') || m.name.includes('Llama 4'),
    )
    expect(llamaModels.length).toBeGreaterThanOrEqual(3)
  })
```

- [ ] **Step 2: Re-point the MoE-pitfall test to Qwen3.6 35B A3B**

Replace this block:

```typescript
  it('should include Mixtral MoE models with correct total parameters', () => {
    const mixtral8x7b = modelsData.find((m) => m.id.includes('mixtral-8x7b'))
    expect(mixtral8x7b).toBeDefined()
    expect(mixtral8x7b?.architecture).toBe('moe')
    // Must be 46.7B (total), NOT 13B (active) - research pitfall #1
    expect(mixtral8x7b?.num_parameters_billion).toBeCloseTo(46.7, 1)
    expect(mixtral8x7b?.num_experts).toBe(8)
    expect(mixtral8x7b?.num_experts_per_token).toBe(2)
  })
```

with:

```typescript
  it('should store MoE models with TOTAL parameters, not active', () => {
    const qwenMoe = modelsData.find((m) => m.id === 'qwen-qwen3.6-35b-a3b')
    expect(qwenMoe).toBeDefined()
    expect(qwenMoe?.architecture).toBe('moe')
    // 36.0B total (all experts), NOT ~3B active - research pitfall #1
    expect(qwenMoe?.num_parameters_billion).toBeCloseTo(36.0, 1)
    expect(qwenMoe?.num_experts).toBe(256)
    expect(qwenMoe?.num_experts_per_token).toBe(8)
  })
```

- [ ] **Step 3: Replace the Phi and Command-R presence tests with current families**

Replace this block:

```typescript
  it('should include Phi models', () => {
    const phiModels = modelsData.filter((m) => m.name.includes('Phi'))
    expect(phiModels.length).toBeGreaterThanOrEqual(2)
  })
```

with:

```typescript
  it('should include Kimi models', () => {
    const kimiModels = modelsData.filter((m) => m.name.includes('Kimi'))
    expect(kimiModels.length).toBeGreaterThanOrEqual(2)
  })
```

and replace this block:

```typescript
  it('should include Command-R models', () => {
    const commandRModels = modelsData.filter((m) => m.name.includes('Command-R'))
    expect(commandRModels.length).toBeGreaterThanOrEqual(2)
  })
```

with:

```typescript
  it('should include GLM models', () => {
    const glmModels = modelsData.filter((m) => m.name.includes('GLM'))
    expect(glmModels.length).toBeGreaterThanOrEqual(2)
  })
```

- [ ] **Step 4: Re-point the GQA test away from retired Mistral 7B**

Replace this block:

```typescript
  it('should specify num_kv_heads for GQA models', () => {
    const llama3_8b = modelsData.find((m) => m.id.includes('llama-3.1-8b'))
    const mistral7b = modelsData.find((m) => m.id.includes('mistral-7b'))

    // LLaMA 3.1 and Mistral use GQA with num_kv_heads < num_attention_heads
    if (llama3_8b) {
      expect(llama3_8b.num_kv_heads).toBeDefined()
      expect(llama3_8b.num_kv_heads).toBeLessThan(llama3_8b.num_attention_heads)
    }
    if (mistral7b) {
      expect(mistral7b.num_kv_heads).toBeDefined()
      expect(mistral7b.num_kv_heads).toBeLessThan(mistral7b.num_attention_heads)
    }
  })
```

with:

```typescript
  it('should specify num_kv_heads for GQA models', () => {
    const llama3_8b = modelsData.find((m) => m.id.includes('llama-3.1-8b'))
    const qwen27b = modelsData.find((m) => m.id === 'qwen-qwen3.6-27b')

    // LLaMA 3.1 and Qwen3.6 use GQA with num_kv_heads < num_attention_heads
    if (llama3_8b) {
      expect(llama3_8b.num_kv_heads).toBeDefined()
      expect(llama3_8b.num_kv_heads).toBeLessThan(llama3_8b.num_attention_heads)
    }
    if (qwen27b) {
      expect(qwen27b.num_kv_heads).toBeDefined()
      expect(qwen27b.num_kv_heads).toBeLessThan(qwen27b.num_attention_heads)
    }
  })
```

- [ ] **Step 5: Add an alphabetical-sort invariant**

Immediately before the final closing `})` of the `describe(...)` block, add:

```typescript
  it('should be sorted alphabetically by name (code-unit order)', () => {
    const names = modelsData.map((m) => m.name)
    const sorted = [...names].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
    expect(names).toEqual(sorted)
  })
```

- [ ] **Step 6: Run the test to verify it now fails (red)**

Run: `npx vitest run src/utils/models.test.ts`
Expected: FAIL — e.g. `should include Gemma 4 variants` (0 found) and `should store MoE models with TOTAL parameters` (`qwen-qwen3.6-35b-a3b` undefined), because the data has not been updated yet.

- [ ] **Step 7: Commit**

```bash
git add src/utils/models.test.ts
git commit -m "test(models): re-point database tests to current-gen roster"
```

---

### Task 2: Refresh `models.json` (retire 26, add 13, re-sort)

**Files:**
- Modify: `src/data/models.json`
- Test: `src/utils/models.test.ts` (from Task 1)

**Interfaces:**
- Consumes: the test invariants from Task 1.
- Produces: a 52-entry `models.json` importable as before. New `id`s that later tasks reference: `qwen-qwen3.6-27b`, `qwen-qwen3.6-35b-a3b`, `deepseek-v4-flash`, `deepseek-v4-pro`, `zai-org-glm-5.2`, `moonshotai-kimi-k2-thinking`, `moonshotai-kimi-linear-48b-a3b`, `google-gemma-4-31b`, `google-gemma-4-12b`, `google-gemma-4-26b-a4b`, `mistralai-mistral-medium-3.5-128b`, `minimax-m3`, `nvidia-nemotron-3-ultra-550b-a55b`.

- [ ] **Step 1: Write the transform script to the scratchpad**

Create `/private/tmp/claude-501/-Users-fjacquet-Projects-llmvram/c3125bf3-21c2-43e1-b95d-c1352221f087/scratchpad/refresh-models.mjs` with this exact content:

```javascript
import { readFile, writeFile } from 'node:fs/promises'

const PATH = 'src/data/models.json'

const RETIRE = new Set([
  'LLaMA 2 7B', 'LLaMA 2 13B', 'LLaMA 2 70B',
  'Mistral 7B v0.1', 'Mistral 7B v0.3',
  'Mixtral 8x7B v0.1', 'Mixtral 8x7B Instruct v0.1', 'Mixtral 8x22B v0.1',
  'Gemma 2B', 'Gemma 7B', 'Gemma 2 9B',
  'Phi 3 Mini 4K', 'Phi 3 Small 8K', 'Phi 3 Medium 4K', 'Phi 3.5 Mini Instruct',
  'MPT 30B', 'Falcon 40B', 'Yi 34B', 'LLaMA 3 8B',
  'DeepSeek Coder 33B', 'DeepSeek V2 Lite', 'DeepSeek Coder V2 Lite',
  'Command-R', 'Command-R Plus',
  'Nemotron Llama 3.1 Nano 8B', 'Nemotron Llama 3.3 Super 49B',
])

const ADD = [
  { id: 'google-gemma-4-31b', name: 'Gemma 4 31B', architecture: 'dense', num_parameters_billion: 32.7, hidden_size: 5376, num_hidden_layers: 60, num_attention_heads: 32, num_kv_heads: 16, intermediate_size: 21504, context_length: 262144, license: 'Apache-2.0', hf_url: 'https://huggingface.co/google/gemma-4-31B-it' },
  { id: 'google-gemma-4-12b', name: 'Gemma 4 12B', architecture: 'dense', num_parameters_billion: 12.0, hidden_size: 3840, num_hidden_layers: 48, num_attention_heads: 16, num_kv_heads: 8, intermediate_size: 15360, context_length: 262144, license: 'Apache-2.0', hf_url: 'https://huggingface.co/google/gemma-4-12B-it' },
  { id: 'google-gemma-4-26b-a4b', name: 'Gemma 4 26B A4B', architecture: 'moe', num_parameters_billion: 26.5, hidden_size: 2816, num_hidden_layers: 30, num_attention_heads: 16, num_kv_heads: 8, intermediate_size: 2112, num_experts: 128, num_experts_per_token: 8, context_length: 262144, license: 'Apache-2.0', hf_url: 'https://huggingface.co/google/gemma-4-26B-A4B-it' },
  { id: 'qwen-qwen3.6-27b', name: 'Qwen3.6 27B', architecture: 'dense', num_parameters_billion: 27.8, hidden_size: 5120, num_hidden_layers: 64, num_attention_heads: 24, num_kv_heads: 4, intermediate_size: 17408, context_length: 262144, license: 'Apache-2.0', hf_url: 'https://huggingface.co/Qwen/Qwen3.6-27B' },
  { id: 'qwen-qwen3.6-35b-a3b', name: 'Qwen3.6 35B A3B', architecture: 'moe', num_parameters_billion: 36.0, hidden_size: 2048, num_hidden_layers: 40, num_attention_heads: 16, num_kv_heads: 2, intermediate_size: 512, num_experts: 256, num_experts_per_token: 8, context_length: 262144, license: 'Apache-2.0', hf_url: 'https://huggingface.co/Qwen/Qwen3.6-35B-A3B' },
  { id: 'deepseek-v4-flash', name: 'DeepSeek V4 Flash', architecture: 'moe', num_parameters_billion: 158.1, hidden_size: 4096, num_hidden_layers: 43, num_attention_heads: 64, num_kv_heads: 1, intermediate_size: 2048, num_experts: 256, num_experts_per_token: 6, context_length: 1048576, license: 'MIT', hf_url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash' },
  { id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro', architecture: 'moe', num_parameters_billion: 861.6, hidden_size: 7168, num_hidden_layers: 61, num_attention_heads: 128, num_kv_heads: 1, intermediate_size: 3072, num_experts: 384, num_experts_per_token: 6, context_length: 1048576, license: 'MIT', hf_url: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro' },
  { id: 'zai-org-glm-5.2', name: 'GLM 5.2', architecture: 'moe', num_parameters_billion: 753.3, hidden_size: 6144, num_hidden_layers: 78, num_attention_heads: 64, num_kv_heads: 64, intermediate_size: 12288, num_experts: 256, num_experts_per_token: 8, context_length: 1048576, license: 'MIT', hf_url: 'https://huggingface.co/zai-org/GLM-5.2' },
  { id: 'moonshotai-kimi-k2-thinking', name: 'Kimi K2 Thinking', architecture: 'moe', num_parameters_billion: 1058.1, hidden_size: 7168, num_hidden_layers: 61, num_attention_heads: 64, num_kv_heads: 64, intermediate_size: 18432, num_experts: 384, num_experts_per_token: 8, context_length: 262144, license: 'Modified MIT', hf_url: 'https://huggingface.co/moonshotai/Kimi-K2-Thinking' },
  { id: 'moonshotai-kimi-linear-48b-a3b', name: 'Kimi Linear 48B A3B', architecture: 'moe', num_parameters_billion: 49.1, hidden_size: 2304, num_hidden_layers: 27, num_attention_heads: 32, num_kv_heads: 32, intermediate_size: 9216, num_experts: 256, num_experts_per_token: 8, context_length: 1048576, license: 'MIT', hf_url: 'https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct' },
  { id: 'mistralai-mistral-medium-3.5-128b', name: 'Mistral Medium 3.5 128B', architecture: 'dense', num_parameters_billion: 127.7, hidden_size: 12288, num_hidden_layers: 88, num_attention_heads: 96, num_kv_heads: 8, intermediate_size: 28672, context_length: 262144, license: 'Mistral Research License', hf_url: 'https://huggingface.co/mistralai/Mistral-Medium-3.5-128B' },
  { id: 'minimax-m3', name: 'MiniMax M3', architecture: 'moe', num_parameters_billion: 427.0, hidden_size: 6144, num_hidden_layers: 60, num_attention_heads: 64, num_kv_heads: 4, intermediate_size: 12288, num_experts: 128, num_experts_per_token: 4, context_length: 1048576, license: 'MiniMax', hf_url: 'https://huggingface.co/MiniMaxAI/MiniMax-M3' },
  { id: 'nvidia-nemotron-3-ultra-550b-a55b', name: 'Nemotron 3 Ultra 550B A55B', architecture: 'moe', num_parameters_billion: 560.5, hidden_size: 8192, num_hidden_layers: 128, num_attention_heads: 64, num_kv_heads: 2, intermediate_size: 5120, num_experts: 512, num_experts_per_token: 22, context_length: 262144, license: 'other', hf_url: 'https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16' },
]

const original = JSON.parse(await readFile(PATH, 'utf8'))
const kept = original.filter((m) => !RETIRE.has(m.name))

// Guard: every retired name must have matched exactly one entry.
const removed = original.length - kept.length
if (removed !== RETIRE.size) {
  throw new Error(`Expected to remove ${RETIRE.size} entries but removed ${removed}. Check name spellings.`)
}

const next = [...kept, ...ADD].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))

// Guard: unique ids.
const ids = next.map((m) => m.id)
if (new Set(ids).size !== ids.length) {
  throw new Error('Duplicate ids detected after merge.')
}

await writeFile(PATH, `${JSON.stringify(next, null, 2)}\n`)
console.log(`OK: ${original.length} -> ${next.length} (removed ${removed}, added ${ADD.length})`)
```

- [ ] **Step 2: Run the transform**

Run: `node "/private/tmp/claude-501/-Users-fjacquet-Projects-llmvram/c3125bf3-21c2-43e1-b95d-c1352221f087/scratchpad/refresh-models.mjs"`
Expected: `OK: 65 -> 52 (removed 26, added 13)`
If it throws instead, STOP and fix the name spellings — do not hand-edit around the guard.

- [ ] **Step 3: Run the model tests to verify they pass (green)**

Run: `npx vitest run src/utils/models.test.ts`
Expected: PASS (all tests, including the new Gemma 4 / Kimi / GLM presence, the Qwen3.6 MoE-total check, and the alphabetical-sort invariant).

- [ ] **Step 4: Typecheck and lint**

Run: `npm run typecheck`
Expected: no errors.
Run: `npm run lint`
Expected: no errors. If Biome flags `models.json` formatting, run `npm run format` and re-run `npm run lint`.

- [ ] **Step 5: Full test suite (guard against unexpected coupling)**

Run: `npx vitest run`
Expected: PASS. Engine tests use inline fixtures and are unaffected; this confirms nothing else imported a retired model.

- [ ] **Step 6: Commit**

```bash
git add src/data/models.json
git commit -m "feat(models): refresh to July-2026 generation (retire 26, add 13, 52 total)"
```

---

### Task 3: Update the fetch-script `MODEL_IDS` to current-gen repos

**Files:**
- Modify: `scripts/fetch-models.ts:4-56` (the `MODEL_IDS` array and its comment)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing consumed by later tasks. Keeps `npm run refresh:models` aligned with the curated roster.

- [ ] **Step 1: Replace the `MODEL_IDS` array**

In `scripts/fetch-models.ts`, replace the entire block from `// Model IDs to fetch (matches the 30+ models from Plan 03)` through the closing `]` of `MODEL_IDS` with:

```typescript
// Model IDs to fetch — current-generation curated roster (2026-07-03 refresh).
// NOTE: multimodal models (Gemma 4, Qwen3.6, MiniMax M3, Mistral 3) expose the
// transformer fields under config.json `text_config`, which this script does not read;
// and several models are gated or custom-code. Treat fetched output as a starting point
// and hand-curate num_parameters_billion / MoE fields against models.json.
const MODEL_IDS = [
  // Gemma 4 (Google)
  'google/gemma-4-31B-it',
  'google/gemma-4-12B-it',
  'google/gemma-4-26B-A4B-it',

  // Qwen3.6 (Alibaba)
  'Qwen/Qwen3.6-27B',
  'Qwen/Qwen3.6-35B-A3B',

  // DeepSeek V4
  'deepseek-ai/DeepSeek-V4-Flash',
  'deepseek-ai/DeepSeek-V4-Pro',

  // GLM (Z.ai)
  'zai-org/GLM-5.2',

  // Kimi (Moonshot)
  'moonshotai/Kimi-K2-Thinking',
  'moonshotai/Kimi-Linear-48B-A3B-Instruct',

  // Mistral
  'mistralai/Mistral-Medium-3.5-128B',

  // MiniMax
  'MiniMaxAI/MiniMax-M3',

  // NVIDIA Nemotron
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
]
```

- [ ] **Step 2: Typecheck and lint the script**

Run: `npm run typecheck`
Expected: no errors.
Run: `npm run lint`
Expected: no errors (run `npm run lint:fix` if Biome reports fixable issues, then re-run).

- [ ] **Step 3: Commit**

```bash
git add scripts/fetch-models.ts
git commit -m "chore(scripts): point fetch-models MODEL_IDS at current-gen repos"
```

---

### Task 4: Update docs (CHANGELOG, README) and memory

**Files:**
- Modify: `CHANGELOG.md` (the `## [Unreleased]` section)
- Modify: `README.md:43` (model-count + family list)
- Modify: `/Users/fjacquet/.claude/projects/-Users-fjacquet-Projects-llmvram/memory/MEMORY.md` (current counts line)

**Interfaces:**
- Consumes: the final roster from Task 2.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add a CHANGELOG entry under `## [Unreleased]`**

In `CHANGELOG.md`, replace the line `## [Unreleased]` with:

```markdown
## [Unreleased]

### Added

- Current-generation models to the database (verified from HuggingFace `config.json` on
  2026-07-03): Gemma 4 (31B, 12B, 26B A4B), Qwen3.6 (27B, 35B A3B), DeepSeek V4 (Flash,
  Pro), GLM 5.2, Kimi K2 Thinking, Kimi Linear 48B A3B, Mistral Medium 3.5 128B,
  MiniMax M3, and Nemotron 3 Ultra 550B A55B (13 models).

### Removed

- Obsolete (2023–early-2024) models from the database: LLaMA 2 (7B/13B/70B), Mistral 7B
  (v0.1/v0.3), Mixtral (8x7B ×2, 8x22B), Gemma 2B/7B, Gemma 2 9B, Phi 3 (Mini/Small/
  Medium) and Phi 3.5, MPT 30B, Falcon 40B, Yi 34B, LLaMA 3 8B, DeepSeek Coder 33B,
  DeepSeek V2 Lite, DeepSeek Coder V2 Lite, Command-R / Command-R Plus, and the
  Llama-based Nemotron mids (3.1 Nano 8B, 3.3 Super 49B) — 26 models. Model count: 52.
```

- [ ] **Step 2: Update the README model-database line**

In `README.md`, replace line 43:

```markdown
64 curated models including LLaMA 2/3/4, Mistral, Mixtral, Qwen, DeepSeek, Gemma, Phi, Command-R, Falcon, Yi, GLM, Kimi K2/K2.5, GPT OSS, Nemotron, MiniMax, and more. Plus custom model input.
```

with:

```markdown
52 curated models including Gemma 4, Qwen3.6 / Qwen2.5, LLaMA 3.x / Llama 4, DeepSeek V4 / R1, GLM 5.2 / 4.7, Kimi K2 / K2 Thinking / Linear, Mistral Large 3 / Medium 3.5, MiniMax M2.x / M3, Nemotron 3, GPT OSS, and more. Plus custom model input.
```

- [ ] **Step 3: Update the memory counts line**

In `/Users/fjacquet/.claude/projects/-Users-fjacquet-Projects-llmvram/memory/MEMORY.md`, replace the line:

```markdown
- Current counts: 65 models, 22 GPUs (as of v1.4.1, 2026-06-03)
```

with:

```markdown
- Current counts: 52 models, 22 GPUs (2026-07-03 generational refresh: retired 26 obsolete, added 13 current-gen — Gemma 4, Qwen3.6, DeepSeek V4, GLM 5.2, Kimi K2 Thinking/Linear, Mistral Medium 3.5, MiniMax M3, Nemotron 3 Ultra 550B)
```

- [ ] **Step 4: Verify docs reference the correct count**

Run: `grep -n "52 curated" README.md && grep -n "52 models" "/Users/fjacquet/.claude/projects/-Users-fjacquet-Projects-llmvram/memory/MEMORY.md"`
Expected: one match in each file.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md README.md
git commit -m "docs: record July-2026 model refresh (52 models)"
```

(The memory file lives outside the repo; it is updated but not committed here.)

---

## Self-Review

**Spec coverage:**
- Add 13 current-gen models → Task 2 (data in the canonical JSON block). ✓
- Retire 26 obsolete → Task 2 (RETIRE set) + guard. ✓
- Re-sort alphabetically → Task 2 sort + Task 1 sort-invariant test. ✓
- Keep entries accurate / verified → values copied from live config.json into the canonical block; not re-derived. ✓
- Refresh fetch pipeline → Task 3. ✓
- CHANGELOG / README / MEMORY → Task 4. ✓
- Spec "testing/validation" (vitest models test, typecheck, lint, full suite) → Task 2 Steps 3–5. ✓
- Spec noted the models.test.ts conflict → Task 1 handles it. ✓
- Optional models (Phi-4, Gemma 4 E4B, LiquidAI) → intentionally NOT added (they were explicitly optional and left out); the Phi presence test is removed accordingly. Nemotron Llama Ultra 253B → kept (per recommendation), so it is absent from the RETIRE set. ✓

**Placeholder scan:** No TBD/TODO; every code/data step shows exact content. ✓

**Type/name consistency:** New `id`s referenced in Task 1 tests (`qwen-qwen3.6-35b-a3b`, `qwen-qwen3.6-27b`) exactly match the `id`s in the Task 2 canonical JSON. Retire names in Task 1's rationale match the RETIRE set in Task 2. Field names match `ModelSchema`. ✓

## Notes for the executor

- The exotic-attention entries (DeepSeek V4 kv=1 MLA, GLM 5.2 sparse attn, Kimi K2 Thinking MLA, Kimi Linear linear-attn, MiniMax M3 lightning-attn, Nemotron 3 Ultra mamba-hybrid) are stored with **nominal** `num_kv_heads` from config, matching how DeepSeek R1 / Kimi K2 / Nemotron 3 Super already sit in the list. KV-cache VRAM for these is a known approximation; accurate math is a separate future engine task — do NOT change engine code here.
- If the full suite (Task 2 Step 5) surfaces a failure in a non-`models.test.ts` file, that means something imported a retired model unexpectedly; stop and report rather than editing engine tests.
