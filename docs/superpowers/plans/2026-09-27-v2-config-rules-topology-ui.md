# v2.0 Configuration Rules, GPU Topology, Simplified UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One pure rule table decides every input dependency (corrected with a visible notice), GPU data describes each card's real topology, multi-node prefill/decode is computed from bytes over the fabric, and the UI shows essentials plus a verdict with everything else one click away.

**Architecture:** `src/engines/config-rules.ts` holds a rule table driving `normalizeConfig` (fixpoint, mode-gated, idempotent) and `allowedOptions`. The Zustand store funnels every config action through one `commit()` that normalizes and writes a `pendingNotice`; URL restore builds the whole config and commits once. GPU rows gain `unified_memory` and `nvlink_bridge`; `resolveInterconnect(gpu, groupSize)` picks the bridge only while the group fits it. `fabric.ts` replaces the efficiency heuristic with `fabricHopSeconds` and a GPipe microbatch fill. The UI reads `allowedOptions` so impossible values are not offered, and collapses advanced inputs and result details into native `<details>`.

**Tech Stack:** React 19, TypeScript 7 strict (`noUncheckedIndexedAccess`), Zustand 5, Zod 4, decimal.js, Vitest 5 + Testing Library (jsdom), Biome, sonner, tsx scripts.

**Spec:** `docs/superpowers/specs/2026-09-27-v2-config-rules-topology-ui-design.md` (ADRs 0004, 0005, 0006, 0007 in `docs/adr/`)

## Global Constraints

- Engines are pure: `src/engines/**` imports no React, DOM, sonner or store code. `config-rules.ts` may import `@utils/schemas` (types, `MAX_SEQUENCE_LENGTH`) and `@utils/gpuLimits` (`maxGPUsFor`), as other engines already do.
- Biome style: 2-space indent, single quotes, no semicolons, 100-column lines; unused imports/variables are errors. Run `npm run lint:fix` before each commit.
- A PostToolUse format hook strips imports that are unused at edit time. Add every import **in the same edit** as its first use, never in a separate earlier edit.
- Test fixtures come from real data: `validateModels(modelsData)` / `validateGPUs(gpusData)` looked up by id. Never add a hand-written GPU or model literal. A schema-validated derived row (`validateGPU({ ...findGPU(id), field })`) is allowed only where the plan says so, with the comment given.
- Test files are typechecked by nothing (`tsconfig.app.json` excludes them). Build fixtures by calling real factories; update every caller the plan lists when a signature changes.
- `src/data/gpus.json` is generated: edit `scripts/fetch-gpus.ts`, then `npm run refresh:gpus && npx biome format --write src/data/gpus.json`. Never hand-edit the JSON.
- Database ids are never renamed. `nvidia-h200-141gb` becomes the SXM product; the NVL card gets the new id `nvidia-h200-nvl-141gb`.
- `models.json` stays sorted by name (codepoint order); this plan does not touch it.
- `fp16_tflops` is dense. New or changed values: H100 PCIe 756 (datasheet 1,513 is sparse), H200 NVL 835 (datasheet 1,671 is sparse).
- Notice texts are the spec's English strings verbatim (Section 1 tables); titles are `Adjusted for {GPU/model/mode}` for actions and `Shared link adjusted` for a restored link.
- Soft rules (W1-W8) warn and keep the value; only hard rules (R1-R14) correct.
- The version bump is NOT part of this PR (release PR later). Do not touch `package.json` "version".
- Commit messages end with exactly these two trailer lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z`
- If `rtk git` misbehaves (fabricated output, blocked in a worktree), use `rtk proxy git ...`.
- Single-file test runs: `npx vitest run <path>`. Never `npm test` (watch mode hangs).

## Review Focus

1. **A shared link whose model cannot be resolved** (unknown id, no custom params) carrying `ss: expert-parallel, ng: 6`: restore must not throw, keeps EP/6 while no model is selected, and the first model pick then applies R2 then R14 in one notice (Task 1a null-model test, Task 2b restore test).
2. **Non-integer, negative and out-of-range numbers in a hand-edited link** (`ng: 2.5`, `ng: -3`, `bs: 0`, `sl: 100`, `nn: 12`, `cu: 0`): the link must open (schema no longer rejects it), every value lands on an integer inside its bound, and one "Shared link adjusted" notice lists each fix (Task 2b hostile-link test).
3. **The same correction twice in a row** (select vLLM, switch to training, back to inference, switch to training again): each action must toast again, not be swallowed as "unchanged state" (Task 2a notice-id test, Task 5 hook test).
4. **Enabling offloading on a unified-memory GPU while the target is still `cpu-ram`**: the checkbox must turn offloading on with target NVMe (action intent), not be silently switched off again by R6, which would make the checkbox look dead (Task 2a test).
5. **PDF export when the user had manually opened or closed Advanced/Details, including when capture fails**: every `<details>` is open during capture and each returns to its exact previous state afterwards (Task 6b test).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/engines/config-rules.ts` | Create | Rule table, `normalizeConfig`, TP-degree validity, `allowedOptions`, `softWarnings`, `buildNotice`, `DEFAULT_RULE_CONFIG` |
| `src/engines/config-rules.test.ts` | Create | Per-rule 3-path table, properties, R14 degree table, mode gating, options, warnings, notices |
| `src/store/uiStore.ts` | Modify | `UIConfig`, `DEFAULT_UI_CONFIG`, `commit()`, `pendingNotice`, `restoreConfig`, action intents |
| `src/store/uiStore.test.ts` | Modify | Real-store tests for notices, cascades, restore |
| `src/store/urlSerializer.ts` | Modify | Loosened numeric keys, `io` key, custom-GPU fields, `urlStateToConfig` |
| `src/store/urlSerializer.test.ts` | Modify | Round-trip of every key, pure restore mapping |
| `src/hooks/useURLSync.ts` | Modify | Deserialize, map, one `restoreConfig` call |
| `src/hooks/useConfigNotices.tsx` | Create | Toast the store's `pendingNotice`, then clear it |
| `src/hooks/useAllowedOptions.ts` | Create | `allowedOptions` bound to the store |
| `src/engines/kv-tier.ts` | Modify | Remove `resetTierForGPU`; comment points at R4 |
| `src/utils/gpuLimits.ts` | Modify | Comment: clamping is a rule with a notice now |
| `src/utils/schemas.ts` | Modify | `unified_memory`, `nvlink_bridge`, `nvlink-3` |
| `src/types/gpu.ts` | Modify | `CustomGPUInput.unified_memory` passthrough |
| `scripts/fetch-gpus.ts` | Modify | Per-card topology corrections, H200 NVL row |
| `src/data/gpus.json` | Regenerate | Output of `npm run refresh:gpus` |
| `src/engines/types.ts` | Modify | `nvlink-3`, `ethernet-200g`, `FabricSpec` without `classFactor`, `interNodeGBps` |
| `src/engines/constants.ts` | Modify | `INTERCONNECT_SPECS['nvlink-3']`, label |
| `src/engines/multi-gpu.ts` | Modify | `resolveInterconnect(gpu, groupSize)`, `interconnectLabel`, `interNodeGBps: 0` |
| `src/engines/fabric.ts` | Modify | Hop time, eta, vLLM chunk size, microbatch fill, 200GbE preset |
| `src/engines/multi-node.ts` | Modify | Section 3b composition |
| `src/engines/performance.ts` | Modify | Bridge-aware link, decode hop, amortized prefill hop |
| `src/engines/index.ts` | Modify | Barrel follows `fabric.ts` exports |
| `src/engines/frameworks.ts` | Modify | TGI labelled archived |
| `src/utils/perfLabels.ts` | Create | `firstTokenLabel(batchSize)` |
| `src/hooks/useInferenceCalculation.ts`, `src/workers/calculation.worker.ts` | Modify | `sequenceLength` to multi-node, new breakdown field, override drops bridge |
| `src/components/inputs/*` | Modify | Options from `allowedOptions`; labels |
| `src/components/layout/InputPanel.tsx` | Modify | Essential / Advanced layout |
| `src/components/layout/advancedChanges.ts` | Create | Count non-default advanced settings |
| `src/components/layout/ResultsPanel.tsx` | Modify | Verdict, warnings, collapsed details, soft warnings |
| `src/components/outputs/VerdictBlock.tsx` | Create | Fit, decode, first token, sessions |
| `src/hooks/useResultExports.ts` | Modify | PDF expands all `<details>` before capture |
| `src/utils/exportPptx.ts` | Modify | TTFT label, GPUs-per-replica row |
| `src/components/guide/GuidePage.tsx`, `CHANGELOG.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `README.md` | Modify | Docs (Task 7) |

Task order follows spec Section 7: 1 (1a, 1b) rules; 2 (2a, 2b) store and URL; 3 (3a, 3b) GPU data and interconnect; 4 (4a, 4b) multi-node model; 5 UI options and notices; 6 (6a, 6b) composition tests then layout; 7 docs.

---

### Task 1a: Rule table and `normalizeConfig`

**Files:**
- Modify: `src/utils/schemas.ts:106-107` (add `unified_memory` before `tier`)
- Create: `src/engines/config-rules.ts`
- Test: `src/engines/config-rules.test.ts`

**Interfaces:**
- Consumes: `maxGPUsFor(gpu: GPU | null): number` (`src/utils/gpuLimits.ts`); `splitMoEParams(model: Model)` (`src/engines/inference.ts:141`); `clampKVTier`, `graceLinkGBps`, `DEFAULT_KV_TIER`, `KVTierSettings` (`src/engines/kv-tier.ts`); `FRAMEWORK_PRESETS`, `FrameworkPreset` (`src/engines/frameworks.ts`); `INTERCONNECT_LABELS`, `MAX_CONCURRENT_USERS` (`src/engines/constants.ts`); `MAX_SEQUENCE_LENGTH` (`src/utils/schemas.ts`).
- Produces (all exported from `src/engines/config-rules.ts`):
  - `type ConfigMode = 'inference' | 'training'`
  - `interface RuleConfig { mode: ConfigMode; numGPUs: number; numNodes: number; shardingStrategy: ShardingStrategy; batchSize: number; concurrentUsers: number; sequenceLength: number; loraRank: number; gradientAccumulationSteps: number; offloadingEnabled: boolean; offloadTarget: OffloadTarget; offloadPercentage: number; offloadLayers: number; kvCacheOffload: boolean; offloadHostCapacityGB: number | null; kvTier: KVTierSettings; interconnectOverride: string | null; frameworkPreset: FrameworkPreset; cpuOffloadOptimizer: boolean }`
  - `const DEFAULT_RULE_CONFIG: RuleConfig` (the store's current defaults)
  - `function pickRuleConfig(source: RuleConfig): RuleConfig`
  - `type RuleId = 'R1' | 'R2' | 'R3' | 'R4' | 'R5' | 'R6' | 'R7' | 'R8' | 'R9' | 'R10' | 'R12' | 'R14'`
  - `type CorrectionSubject = 'gpu' | 'model' | 'mode' | 'preset' | 'offload' | 'range'`
  - `interface Correction { rule: RuleId; field: keyof RuleConfig; from: unknown; to: unknown; subject: CorrectionSubject; message: string }`
  - `interface RuleContext { model: Model | null; gpu: GPU | null }`
  - `interface Rule { id: RuleId; modes: readonly ConfigMode[]; check: (config: RuleConfig, ctx: RuleContext) => Correction[] }`
  - `const RULES: readonly Rule[]`
  - `const MAX_NORMALIZE_PASSES = 4`
  - `interface NormalizeResult<C extends RuleConfig> { config: C; corrections: Correction[]; passes: number }`
  - `function normalizeConfig<C extends RuleConfig>(config: C, model: Model | null, gpu: GPU | null, rules?: readonly Rule[]): NormalizeResult<C>`
  - `function isValidTPDegree(model: Model, t: number): boolean`
  - `function validTPDegrees(model: Model, max: number): number[]`
  - `function supportsCpuOffload(preset: FrameworkPreset): boolean`
- Schema: `GPUSchema.unified_memory?: boolean` (data values arrive in Task 3a).

- [ ] **Step 1: Add `unified_memory` to `GPUSchema`**

In `src/utils/schemas.ts`, insert before `// Classification` (line 106):

```ts
  /**
   * CPU and GPU share one memory pool (Apple Silicon, GB10 DGX Spark). The only
   * source for "no separate host memory" (config-rules R6): never infer it from
   * `interconnect === 'unified'` or from the tier.
   */
  unified_memory: z.boolean().optional(),

```

- [ ] **Step 2: Write the failing tests**

Create `src/engines/config-rules.test.ts`:

```ts
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { type GPU, type Model, validateGPU, validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import {
  type Correction,
  DEFAULT_RULE_CONFIG,
  isValidTPDegree,
  MAX_NORMALIZE_PASSES,
  normalizeConfig,
  RULES,
  type RuleConfig,
  type RuleId,
  validTPDegrees,
} from './config-rules'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)

function findModel(id: string): Model {
  const m = models.find((x) => x.id === id)
  if (!m) throw new Error(`fixture model not found in models.json: ${id}`)
  return m
}
function findGPU(id: string): GPU {
  const g = gpus.find((x) => x.id === id)
  if (!g) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return g
}

const L70 = findModel('meta-llama-llama-3.1-70b') // 64 heads, 8 KV heads, 80 layers
const L8 = findModel('meta-llama-llama-3.1-8b') // 32 layers
const DSR1 = findModel('deepseek-r1') // MoE
const KIMI_K3 = findModel('moonshotai-kimi-k3') // 96 heads, 96 KV heads
const GEMMA_1B = findModel('google-gemma-3-1b') // 4 heads, 1 KV head
const H100 = findGPU('nvidia-h100-80gb-sxm') // max 8
const NVL72 = findGPU('nvidia-gb300-nvl72') // max 72, Grace host
const M3 = findGPU('apple-m3-ultra') // max 1
// Derived rows, validated by the real schema. UNIFIED_M3: Task 3a sets
// unified_memory in the data and replaces this with M3. WITH_OPTIONS: after
// Task 3a no database GPU carries interconnect_options, so R5 is only reachable
// through a row like this one.
const UNIFIED_M3 = validateGPU({ ...M3, unified_memory: true })
const WITH_OPTIONS = validateGPU({ ...H100, interconnect_options: ['nvlink-4', 'pcie-5'] })

const cfg = (patch: Partial<RuleConfig> = {}): RuleConfig => ({ ...DEFAULT_RULE_CONFIG, ...patch })

function ruleFixes(result: { corrections: Correction[] }, rule: RuleId): Correction[] {
  return result.corrections.filter((c) => c.rule === rule)
}

interface RuleCase {
  rule: RuleId
  path: 'field' | 'dependency' | 'link'
  name: string
  config: Partial<RuleConfig>
  model: Model | null
  gpu: GPU | null
  /** For the dependency path: the same config is valid here (no correction for this rule) */
  validUnder?: { model: Model | null; gpu: GPU | null; mode?: RuleConfig['mode'] }
  expected: Partial<RuleConfig>
}

const CASES: RuleCase[] = [
  // R1: numGPUs in [1, max_gpus_per_node]
  { rule: 'R1', path: 'field', name: '9 on an 8-GPU board', config: { numGPUs: 9 }, model: L70, gpu: H100, expected: { numGPUs: 8 } },
  { rule: 'R1', path: 'dependency', name: '8 then a 1-GPU part', config: { numGPUs: 8 }, model: L70, gpu: M3, validUnder: { model: L70, gpu: H100 }, expected: { numGPUs: 1 } },
  { rule: 'R1', path: 'link', name: 'negative count', config: { numGPUs: -3 }, model: L70, gpu: H100, expected: { numGPUs: 1 } },
  // R2: expert-parallel only for a splittable MoE
  { rule: 'R2', path: 'field', name: 'EP on a dense model', config: { shardingStrategy: 'expert-parallel', numGPUs: 8 }, model: L70, gpu: H100, expected: { shardingStrategy: 'tensor-parallel' } },
  { rule: 'R2', path: 'dependency', name: 'EP then a dense model', config: { shardingStrategy: 'expert-parallel', numGPUs: 8 }, model: L70, gpu: H100, validUnder: { model: DSR1, gpu: H100 }, expected: { shardingStrategy: 'tensor-parallel' } },
  { rule: 'R2', path: 'link', name: 'dense + EP at 6 cascades into R14', config: { shardingStrategy: 'expert-parallel', numGPUs: 6 }, model: L70, gpu: H100, expected: { shardingStrategy: 'tensor-parallel', numGPUs: 4 } },
  // R3: host-grace only on a Grace host
  { rule: 'R3', path: 'field', name: 'host-grace on HGX', config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } } },
  { rule: 'R3', path: 'dependency', name: 'host-grace then HGX', config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' } }, model: L70, gpu: H100, validUnder: { model: L70, gpu: NVL72 }, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } } },
  { rule: 'R3', path: 'link', name: 'host-grace with a 500% active share', config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 5 } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none', activeShare: 1 } } },
  // R4: KV tier bounds, host capacity positive or null (no model/GPU dependency: live in both modes)
  { rule: 'R4', path: 'field', name: 'active share 0', config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network', activeShare: 0 } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network', activeShare: 0.01 } } },
  { rule: 'R4', path: 'dependency', name: 'still live in training mode', config: { mode: 'training', kvTier: { ...DEFAULT_KV_TIER, burstSeconds: 0 } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, burstSeconds: 1 } } },
  { rule: 'R4', path: 'link', name: 'host capacity 0', config: { offloadHostCapacityGB: 0 }, model: L70, gpu: H100, expected: { offloadHostCapacityGB: null } },
  // R5: interconnect override must be one of the GPU's options
  { rule: 'R5', path: 'field', name: 'override on a GPU without options', config: { interconnectOverride: 'pcie-5' }, model: L70, gpu: H100, expected: { interconnectOverride: null } },
  { rule: 'R5', path: 'dependency', name: 'valid override then another GPU', config: { interconnectOverride: 'pcie-5' }, model: L70, gpu: H100, validUnder: { model: L70, gpu: WITH_OPTIONS }, expected: { interconnectOverride: null } },
  { rule: 'R5', path: 'link', name: 'garbage override', config: { interconnectOverride: 'nvlink-9' }, model: L70, gpu: WITH_OPTIONS, expected: { interconnectOverride: null } },
  // R6: unified memory has no separate host
  { rule: 'R6', path: 'field', name: 'cpu-ram offload on unified memory', config: { offloadingEnabled: true, offloadTarget: 'cpu-ram' }, model: L8, gpu: UNIFIED_M3, expected: { offloadingEnabled: false } },
  { rule: 'R6', path: 'dependency', name: 'offload + PCIe tier then unified GPU', config: { offloadingEnabled: true, offloadTarget: 'cpu-ram', kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' } }, model: L8, gpu: UNIFIED_M3, validUnder: { model: L8, gpu: H100 }, expected: { offloadingEnabled: false, kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } } },
  { rule: 'R6', path: 'link', name: 'training optimizer offload on unified memory', config: { mode: 'training', frameworkPreset: 'deepspeed-zero3', cpuOffloadOptimizer: true }, model: L8, gpu: UNIFIED_M3, expected: { cpuOffloadOptimizer: false } },
  // R7: inference-only presets cleared in training
  { rule: 'R7', path: 'field', name: 'vLLM in training', config: { mode: 'training', frameworkPreset: 'vllm' }, model: L70, gpu: H100, expected: { frameworkPreset: 'none' } },
  { rule: 'R7', path: 'dependency', name: 'vLLM then training mode', config: { mode: 'training', frameworkPreset: 'vllm' }, model: L70, gpu: H100, validUnder: { model: L70, gpu: H100, mode: 'inference' }, expected: { frameworkPreset: 'none' } },
  { rule: 'R7', path: 'link', name: 'TGI + optimizer offload cascades into R8', config: { mode: 'training', frameworkPreset: 'tgi', cpuOffloadOptimizer: true }, model: L70, gpu: H100, expected: { frameworkPreset: 'none', cpuOffloadOptimizer: false } },
  // R8: CPU optimizer offload needs a ZeRO preset
  { rule: 'R8', path: 'field', name: 'offload with Unsloth', config: { mode: 'training', frameworkPreset: 'unsloth', cpuOffloadOptimizer: true }, model: L70, gpu: H100, expected: { cpuOffloadOptimizer: false } },
  { rule: 'R8', path: 'dependency', name: 'ZeRO-2 offload then no preset', config: { mode: 'training', frameworkPreset: 'none', cpuOffloadOptimizer: true }, model: L70, gpu: H100, expected: { cpuOffloadOptimizer: false } },
  { rule: 'R8', path: 'link', name: 'offload flag with no preset', config: { mode: 'training', cpuOffloadOptimizer: true }, model: L70, gpu: H100, expected: { cpuOffloadOptimizer: false } },
  // R9: offload amounts bounded by the model
  { rule: 'R9', path: 'field', name: '500 layers on an 80-layer model', config: { offloadLayers: 500 }, model: L70, gpu: H100, expected: { offloadLayers: 80 } },
  { rule: 'R9', path: 'dependency', name: '60 layers then a 32-layer model', config: { offloadLayers: 60 }, model: L8, gpu: H100, validUnder: { model: L70, gpu: H100 }, expected: { offloadLayers: 32 } },
  { rule: 'R9', path: 'link', name: '150% and -2 layers', config: { offloadPercentage: 150, offloadLayers: -2 }, model: L70, gpu: H100, expected: { offloadPercentage: 100, offloadLayers: 0 } },
  // R10: numeric bounds (no model/GPU dependency: live in both modes)
  { rule: 'R10', path: 'field', name: 'batch 0', config: { batchSize: 0 }, model: L70, gpu: H100, expected: { batchSize: 1 } },
  { rule: 'R10', path: 'dependency', name: 'LoRA rank 0 in training', config: { mode: 'training', loraRank: 0 }, model: L70, gpu: H100, expected: { loraRank: 1 } },
  { rule: 'R10', path: 'link', name: 'every bound at once', config: { sequenceLength: 100, numNodes: 12, concurrentUsers: 0, gradientAccumulationSteps: 0 }, model: L70, gpu: H100, expected: { sequenceLength: 512, numNodes: 8, concurrentUsers: 1, gradientAccumulationSteps: 1 } },
  // R12: KV cache offload excludes a KV tier
  { rule: 'R12', path: 'field', name: 'tier on top of KV offload', config: { offloadingEnabled: true, kvCacheOffload: true, kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } } },
  { rule: 'R12', path: 'dependency', name: 'tier then offloading turned on', config: { offloadingEnabled: true, kvCacheOffload: true, kvTier: { ...DEFAULT_KV_TIER, tier: 'local-nvme' } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } } },
  { rule: 'R12', path: 'link', name: 'stale kvCacheOffload with offloading off is inert', config: { offloadingEnabled: false, kvCacheOffload: true, kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } }, model: L70, gpu: H100, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } } },
  // R14: vLLM tensor-parallel divisibility
  { rule: 'R14', path: 'field', name: 'Llama 70B on 6', config: { numGPUs: 6 }, model: L70, gpu: H100, expected: { numGPUs: 4 } },
  { rule: 'R14', path: 'dependency', name: 'PP at 6 then tensor parallel', config: { numGPUs: 6, shardingStrategy: 'tensor-parallel' }, model: L70, gpu: H100, expected: { numGPUs: 4 } },
  { rule: 'R14', path: 'dependency', name: '8 then a 4-head model', config: { numGPUs: 8 }, model: GEMMA_1B, gpu: H100, validUnder: { model: L70, gpu: H100 }, expected: { numGPUs: 4 } },
  { rule: 'R14', path: 'link', name: 'Kimi K3 keeps 6 (96 heads)', config: { numGPUs: 6 }, model: KIMI_K3, gpu: H100, expected: { numGPUs: 6 } },
]

describe('normalizeConfig: per-rule paths (field, dependency, link)', () => {
  it.each(CASES)('$rule $path: $name', (c) => {
    const result = normalizeConfig(cfg(c.config), c.model, c.gpu)
    expect(result.config).toMatchObject(c.expected)
    if (c.validUnder) {
      const before = normalizeConfig(
        cfg({ ...c.config, mode: c.validUnder.mode ?? c.config.mode ?? 'inference' }),
        c.validUnder.model,
        c.validUnder.gpu,
      )
      expect(ruleFixes(before, c.rule)).toEqual([])
    }
  })

  it('names the spec notice texts', () => {
    const r1 = normalizeConfig(cfg({ numGPUs: 9 }), L70, H100)
    expect(r1.corrections[0]?.message).toBe(
      `GPU count set to 8: ${H100.name} supports at most 8 per server.`,
    )
    const r14 = normalizeConfig(cfg({ numGPUs: 6 }), L70, H100)
    expect(r14.corrections[0]?.message).toBe(
      `GPU count set to 4: vLLM can't split ${L70.name}'s 64 attention heads across 6 GPUs. Use pipeline parallel for 6.`,
    )
    const r2 = normalizeConfig(cfg({ shardingStrategy: 'expert-parallel' }), L70, H100)
    expect(r2.corrections[0]?.message).toBe(
      `Strategy set to tensor parallel: ${L70.name} is not a MoE model.`,
    )
    const r7 = normalizeConfig(cfg({ mode: 'training', frameworkPreset: 'vllm' }), L70, H100)
    expect(r7.corrections[0]?.message).toBe('Framework preset cleared: vLLM is inference-only.')
  })
})

describe('normalizeConfig: mode gating', () => {
  it('training ZeRO-3 on 6 GPUs with Llama 3.1 70B keeps 6 and reports nothing (R14 is inference-only)', () => {
    const training = cfg({ mode: 'training', frameworkPreset: 'deepspeed-zero3', numGPUs: 6 })
    const result = normalizeConfig(training, L70, H100)
    expect(result.config.numGPUs).toBe(6)
    expect(result.corrections).toEqual([])
  })

  it('the switch to inference is the trigger: the same config snaps to 4 once', () => {
    const result = normalizeConfig(
      cfg({ mode: 'inference', frameworkPreset: 'deepspeed-zero3', numGPUs: 6 }),
      L70,
      H100,
    )
    expect(result.config.numGPUs).toBe(4)
    expect(result.corrections.map((c) => c.rule)).toEqual(['R14'])
  })

  it('never corrects fields that are inert in training (strategy, KV tier, interconnect)', () => {
    const inert = cfg({
      mode: 'training',
      shardingStrategy: 'expert-parallel',
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' },
      interconnectOverride: 'pcie-5',
    })
    const result = normalizeConfig(inert, L70, H100)
    expect(result.corrections).toEqual([])
    expect(result.config).toEqual(inert)
  })

  it('keeps inference-only corrections off in training, and the training ones off in inference', () => {
    const presetInInference = cfg({ frameworkPreset: 'vllm', cpuOffloadOptimizer: true })
    expect(normalizeConfig(presetInInference, L70, H100).corrections).toEqual([])
  })
})

describe('normalizeConfig: no model or GPU selected', () => {
  it('keeps EP and 6 GPUs while no model is selected (link whose model is unknown)', () => {
    const result = normalizeConfig(
      cfg({ shardingStrategy: 'expert-parallel', numGPUs: 6 }),
      null,
      H100,
    )
    expect(result.config.shardingStrategy).toBe('expert-parallel')
    expect(result.config.numGPUs).toBe(6)
  })

  it('bounds numGPUs by the flat sanity bound with no GPU', () => {
    expect(normalizeConfig(cfg({ numGPUs: 100 }), null, null).config.numGPUs).toBe(72)
  })
})

// Hostile configurations used by the property tests: every rule fires at least once.
const HOSTILE: { name: string; config: RuleConfig; model: Model | null; gpu: GPU | null }[] = [
  { name: 'dense EP at 100 GPUs', config: cfg({ shardingStrategy: 'expert-parallel', numGPUs: 100 }), model: L70, gpu: H100 },
  { name: 'GLM-4.7-Flash at 100 on 8', config: cfg({ numGPUs: 100 }), model: findModel('zai-org-glm-4.7-flash'), gpu: H100 },
  { name: 'every bound broken', config: cfg({ numGPUs: 2.5, batchSize: 0, sequenceLength: 100, numNodes: 12, concurrentUsers: 0, offloadLayers: 500, offloadPercentage: -5, offloadHostCapacityGB: -1, interconnectOverride: 'nvlink-9', kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 0 } }), model: L70, gpu: H100 },
  { name: 'unified memory, everything offloaded', config: cfg({ offloadingEnabled: true, offloadTarget: 'cpu-ram', kvCacheOffload: true, kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' } }), model: L8, gpu: UNIFIED_M3 },
  { name: 'training with inference preset', config: cfg({ mode: 'training', frameworkPreset: 'tgi', cpuOffloadOptimizer: true, loraRank: 0, gradientAccumulationSteps: -1 }), model: L70, gpu: UNIFIED_M3 },
  { name: 'NaN count', config: cfg({ numGPUs: Number.NaN }), model: L70, gpu: NVL72 },
  { name: 'a billion GPUs (R14 must stay bounded in any rule order)', config: cfg({ numGPUs: 1e9 }), model: L70, gpu: H100 },
]

function ruleOrders(): (readonly (typeof RULES)[number][])[] {
  const orders: (typeof RULES)[number][][] = []
  for (let i = 0; i < RULES.length; i++) {
    const rotated = [...RULES.slice(i), ...RULES.slice(0, i)]
    orders.push(rotated, [...rotated].reverse())
  }
  return orders
}

describe('normalizeConfig: properties', () => {
  it.each(HOSTILE)('is idempotent: $name', ({ config, model, gpu }) => {
    const once = normalizeConfig(config, model, gpu)
    const twice = normalizeConfig(once.config, model, gpu)
    expect(twice.corrections).toEqual([])
    expect(twice.passes).toBe(1)
    expect(twice.config).toEqual(once.config)
  })

  it.each(HOSTILE)('is independent of rule order and converges within 4 passes: $name', ({
    config,
    model,
    gpu,
  }) => {
    const reference = normalizeConfig(config, model, gpu).config
    for (const order of ruleOrders()) {
      const result = normalizeConfig(config, model, gpu, order)
      expect(result.config).toEqual(reference)
      expect(result.passes).toBeLessThanOrEqual(MAX_NORMALIZE_PASSES)
    }
  })

  it('reaches the GLM-4.7-Flash fixpoint (100 -> 5) in 3 passes at worst', () => {
    const glm = findModel('zai-org-glm-4.7-flash')
    const result = normalizeConfig(cfg({ numGPUs: 100 }), glm, H100)
    expect(result.config.numGPUs).toBe(5)
    expect(result.passes).toBeLessThanOrEqual(3)
  })
})

describe('R14: valid tensor-parallel degrees from models.json', () => {
  it('allows {1,2,4,8} up to 8 GPUs for every model except three', () => {
    const special: Record<string, number[]> = {
      'google-gemma-3-1b': [1, 2, 4],
      'zai-org-glm-4.7-flash': [1, 2, 4, 5],
      'moonshotai-kimi-k3': [1, 2, 3, 4, 6, 8],
    }
    for (const m of models) {
      expect(validTPDegrees(m, 8), m.id).toEqual(special[m.id] ?? [1, 2, 4, 8])
    }
  })

  it('matches the NVL72 degree sets by head count', () => {
    const table: Record<string, number[]> = {
      'meta-llama-llama-3.1-70b': [1, 2, 4, 8, 16, 32, 64],
      'meta-llama-llama-4-scout': [1, 2, 4, 8, 40],
      'minimax-m2.1': [1, 2, 4, 8, 16, 24, 48],
      'zai-org-glm-4.7': [1, 2, 4, 8, 16, 24, 32, 48],
      'moonshotai-kimi-k3': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48],
      'qwen-qwen3.6-27b': [1, 2, 4, 8, 12, 24],
      'zai-org-glm-4.7-flash': [1, 2, 4, 5, 10, 20],
      'google-gemma-3-1b': [1, 2, 4],
    }
    for (const [id, degrees] of Object.entries(table)) {
      expect(validTPDegrees(findModel(id), 72), id).toEqual(degrees)
    }
  })

  it('no model is TP-1-only, and TP-1 is always valid', () => {
    for (const m of models) {
      expect(isValidTPDegree(m, 1), m.id).toBe(true)
      expect(validTPDegrees(m, 8).length, m.id).toBeGreaterThan(1)
    }
  })

  it('rejects non-integer and non-positive degrees', () => {
    expect(isValidTPDegree(L70, 2.5)).toBe(false)
    expect(isValidTPDegree(L70, 0)).toBe(false)
    expect(isValidTPDegree(L70, -2)).toBe(false)
  })
})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `npx vitest run src/engines/config-rules.test.ts`
Expected: FAIL with `Failed to resolve import "./config-rules"`.

- [ ] **Step 4: Write the implementation**

Create `src/engines/config-rules.ts`:

```ts
import { maxGPUsFor } from '@utils/gpuLimits'
import { type GPU, MAX_SEQUENCE_LENGTH, type Model } from '@utils/schemas'
import { INTERCONNECT_LABELS, MAX_CONCURRENT_USERS } from './constants'
import { FRAMEWORK_PRESETS, type FrameworkPreset } from './frameworks'
import { splitMoEParams } from './inference'
import { clampKVTier, DEFAULT_KV_TIER, graceLinkGBps, type KVTierSettings } from './kv-tier'
import type { OffloadTarget, ShardingStrategy } from './types'

/**
 * One configuration rule set (ADR 0004, spec Section 1).
 *
 * Every input dependency lives in RULES. normalizeConfig applies the rules live in
 * the current mode until nothing changes, so a field that is hidden as inert is
 * never corrected. Every correction is a constant or a clamp toward model/GPU data,
 * which no rule writes, so the loop converges in at most 3 passes in any rule order.
 * Engine throws stay as a backstop only.
 */

export type ConfigMode = 'inference' | 'training'

/** The store fields the rules read or write */
export interface RuleConfig {
  mode: ConfigMode
  /** Parallel degree of one replica inside one server */
  numGPUs: number
  numNodes: number
  shardingStrategy: ShardingStrategy
  batchSize: number
  concurrentUsers: number
  sequenceLength: number
  loraRank: number
  gradientAccumulationSteps: number
  offloadingEnabled: boolean
  offloadTarget: OffloadTarget
  offloadPercentage: number
  offloadLayers: number
  kvCacheOffload: boolean
  offloadHostCapacityGB: number | null
  kvTier: KVTierSettings
  interconnectOverride: string | null
  frameworkPreset: FrameworkPreset
  cpuOffloadOptimizer: boolean
}

/** The calculator's defaults for the rule fields; the store spreads these into its own */
export const DEFAULT_RULE_CONFIG: RuleConfig = {
  mode: 'inference',
  numGPUs: 1,
  numNodes: 1,
  shardingStrategy: 'tensor-parallel',
  batchSize: 1,
  concurrentUsers: 1,
  sequenceLength: 4096,
  loraRank: 16,
  gradientAccumulationSteps: 1,
  offloadingEnabled: false,
  offloadTarget: 'cpu-ram',
  offloadPercentage: 0,
  offloadLayers: 0,
  kvCacheOffload: false,
  offloadHostCapacityGB: null,
  kvTier: DEFAULT_KV_TIER,
  interconnectOverride: null,
  frameworkPreset: 'none',
  cpuOffloadOptimizer: false,
}

/** Copy only the rule fields out of a larger object (e.g. the store state) */
export function pickRuleConfig(source: RuleConfig): RuleConfig {
  return {
    mode: source.mode,
    numGPUs: source.numGPUs,
    numNodes: source.numNodes,
    shardingStrategy: source.shardingStrategy,
    batchSize: source.batchSize,
    concurrentUsers: source.concurrentUsers,
    sequenceLength: source.sequenceLength,
    loraRank: source.loraRank,
    gradientAccumulationSteps: source.gradientAccumulationSteps,
    offloadingEnabled: source.offloadingEnabled,
    offloadTarget: source.offloadTarget,
    offloadPercentage: source.offloadPercentage,
    offloadLayers: source.offloadLayers,
    kvCacheOffload: source.kvCacheOffload,
    offloadHostCapacityGB: source.offloadHostCapacityGB,
    kvTier: source.kvTier,
    interconnectOverride: source.interconnectOverride,
    frameworkPreset: source.frameworkPreset,
    cpuOffloadOptimizer: source.cpuOffloadOptimizer,
  }
}

export type RuleId = 'R1' | 'R2' | 'R3' | 'R4' | 'R5' | 'R6' | 'R7' | 'R8' | 'R9' | 'R10' | 'R12' | 'R14'

/** What the correction was adjusted for; picks the notice title for a plain setting change */
export type CorrectionSubject = 'gpu' | 'model' | 'mode' | 'preset' | 'offload' | 'range'

export interface Correction {
  rule: RuleId
  field: keyof RuleConfig
  from: unknown
  to: unknown
  subject: CorrectionSubject
  /** English UI text, spec Section 1 */
  message: string
}

export interface RuleContext {
  model: Model | null
  gpu: GPU | null
}

export interface Rule {
  id: RuleId
  /** Modes in which the rule's fields are live; inert fields are never corrected */
  modes: readonly ConfigMode[]
  check: (config: RuleConfig, ctx: RuleContext) => Correction[]
}

const BOTH: readonly ConfigMode[] = ['inference', 'training']
const INFERENCE: readonly ConfigMode[] = ['inference']
const TRAINING: readonly ConfigMode[] = ['training']

function fix<K extends keyof RuleConfig>(
  rule: RuleId,
  field: K,
  from: RuleConfig[K],
  to: RuleConfig[K],
  subject: CorrectionSubject,
  message: string,
): Correction {
  return { rule, field, from, to, subject, message }
}

/** Integer clamp; a non-finite value falls to the minimum */
function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, Math.trunc(value)))
}

function gpuName(gpu: GPU | null): string {
  return gpu?.name ?? 'this server'
}

function sameTier(a: KVTierSettings, b: KVTierSettings): boolean {
  return (
    a.tier === b.tier &&
    a.customGBps === b.customGBps &&
    a.activeShare === b.activeShare &&
    a.burstSeconds === b.burstSeconds &&
    a.capacityTB === b.capacityTB
  )
}

/** DeepSpeed ZeRO presets are the only ones that can offload optimizer state (R8) */
export function supportsCpuOffload(preset: FrameworkPreset): boolean {
  return FRAMEWORK_PRESETS[preset].zeroStage !== null
}

/**
 * vLLM's tensor-parallel checks: attention heads divide by t (config/model.py), and
 * t divides by the KV heads when t >= kv, else the KV heads divide by t
 * (model_executor/layers/linear.py). MLA rows store kv heads = heads, so the rule
 * reduces to the heads check, matching vLLM's MLA path.
 */
export function isValidTPDegree(model: Model, t: number): boolean {
  if (!Number.isInteger(t) || t < 1) return false
  const heads = model.num_attention_heads
  const kv = model.num_kv_heads ?? heads
  if (heads % t !== 0) return false
  return t >= kv ? t % kv === 0 : kv % t === 0
}

/** Valid tensor-parallel degrees 1..max, ascending (always starts with 1) */
export function validTPDegrees(model: Model, max: number): number[] {
  const degrees: number[] = []
  for (let t = 1; t <= max; t++) if (isValidTPDegree(model, t)) degrees.push(t)
  return degrees
}

const RANGE_LABELS = {
  numNodes: 'Servers',
  batchSize: 'Batch size',
  concurrentUsers: 'Concurrent users',
  sequenceLength: 'Context length',
  loraRank: 'LoRA rank',
  gradientAccumulationSteps: 'Gradient accumulation steps',
} as const

const RANGE_BOUNDS: { field: keyof typeof RANGE_LABELS; min: number; max: number }[] = [
  { field: 'numNodes', min: 1, max: 8 },
  { field: 'batchSize', min: 1, max: Number.MAX_SAFE_INTEGER },
  { field: 'concurrentUsers', min: 1, max: MAX_CONCURRENT_USERS },
  { field: 'sequenceLength', min: 512, max: MAX_SEQUENCE_LENGTH },
  { field: 'loraRank', min: 1, max: Number.MAX_SAFE_INTEGER },
  { field: 'gradientAccumulationSteps', min: 1, max: Number.MAX_SAFE_INTEGER },
]

/**
 * The hard rules. Order is the dependency order (R1 and R2 before R14, R7 before R8),
 * but normalizeConfig loops to a fixpoint, so any order gives the same result.
 */
export const RULES: readonly Rule[] = [
  {
    id: 'R1',
    modes: BOTH,
    check: (c, { gpu }) => {
      const max = maxGPUsFor(gpu)
      const n = clampInt(c.numGPUs, 1, max)
      if (n === c.numGPUs) return []
      const tooMany = c.numGPUs > max
      return [
        fix(
          'R1',
          'numGPUs',
          c.numGPUs,
          n,
          tooMany ? 'gpu' : 'range',
          tooMany
            ? `GPU count set to ${n}: ${gpuName(gpu)} supports at most ${max} per server.`
            : `GPU count set to ${n}: outside the allowed range.`,
        ),
      ]
    },
  },
  {
    id: 'R2',
    modes: INFERENCE,
    check: (c, { model }) =>
      c.shardingStrategy === 'expert-parallel' && model && !splitMoEParams(model)
        ? [
            fix(
              'R2',
              'shardingStrategy',
              c.shardingStrategy,
              'tensor-parallel',
              'model',
              `Strategy set to tensor parallel: ${model.name} is not a MoE model.`,
            ),
          ]
        : [],
  },
  {
    id: 'R3',
    modes: INFERENCE,
    check: (c, { gpu }) =>
      c.kvTier.tier === 'host-grace' && graceLinkGBps(gpu?.id ?? '') === null
        ? [
            fix(
              'R3',
              'kvTier',
              c.kvTier,
              { ...c.kvTier, tier: 'none' },
              'gpu',
              `KV tier turned off: ${gpuName(gpu)} has no Grace host memory.`,
            ),
          ]
        : [],
  },
  {
    id: 'R4',
    modes: BOTH,
    check: (c) => {
      const out: Correction[] = []
      const tier = clampKVTier(c.kvTier)
      if (!sameTier(tier, c.kvTier)) {
        out.push(
          fix('R4', 'kvTier', c.kvTier, tier, 'range', 'KV tier setting adjusted to its allowed range.'),
        )
      }
      const hc = c.offloadHostCapacityGB
      if (hc !== null && !(Number.isFinite(hc) && hc > 0)) {
        out.push(
          fix(
            'R4',
            'offloadHostCapacityGB',
            hc,
            null,
            'range',
            'Host capacity reset to the default: it must be a positive number.',
          ),
        )
      }
      return out
    },
  },
  {
    id: 'R5',
    modes: INFERENCE,
    check: (c, { gpu }) => {
      const override = c.interconnectOverride
      if (override === null) return []
      // R13: variants are only offered on GPUs that form multi-GPU servers
      const offered: readonly string[] =
        gpu && gpu.max_gpus_per_node > 1 ? (gpu.interconnect_options ?? []) : []
      if (offered.includes(override)) return []
      const fallback = INTERCONNECT_LABELS[gpu?.interconnect ?? 'none'] ?? 'the default'
      return [
        fix(
          'R5',
          'interconnectOverride',
          override,
          null,
          'gpu',
          `Interconnect reset to ${fallback}: not available on ${gpuName(gpu)}.`,
        ),
      ]
    },
  },
  {
    id: 'R6',
    modes: INFERENCE,
    check: (c, { gpu }) => {
      if (!gpu || gpu.unified_memory !== true) return []
      const out: Correction[] = []
      if (c.offloadingEnabled && c.offloadTarget === 'cpu-ram') {
        out.push(
          fix(
            'R6',
            'offloadingEnabled',
            true,
            false,
            'gpu',
            `Offloading turned off: ${gpu.name} has unified memory, RAM is the same pool.`,
          ),
        )
      }
      if (c.kvTier.tier === 'host-pcie' || c.kvTier.tier === 'host-grace') {
        out.push(
          fix(
            'R6',
            'kvTier',
            c.kvTier,
            { ...c.kvTier, tier: 'none' },
            'gpu',
            `KV tier turned off: ${gpu.name} has no separate host memory.`,
          ),
        )
      }
      return out
    },
  },
  {
    id: 'R6',
    modes: TRAINING,
    check: (c, { gpu }) =>
      gpu?.unified_memory === true && c.cpuOffloadOptimizer
        ? [
            fix(
              'R6',
              'cpuOffloadOptimizer',
              true,
              false,
              'gpu',
              'CPU optimizer offload turned off: unified memory.',
            ),
          ]
        : [],
  },
  {
    id: 'R7',
    modes: TRAINING,
    // Explicit list, not FRAMEWORK_PRESETS[p].mode: 'none' is valid in both modes.
    check: (c) =>
      c.frameworkPreset === 'vllm' || c.frameworkPreset === 'tgi'
        ? [
            fix(
              'R7',
              'frameworkPreset',
              c.frameworkPreset,
              'none',
              'mode',
              `Framework preset cleared: ${FRAMEWORK_PRESETS[c.frameworkPreset].name} is inference-only.`,
            ),
          ]
        : [],
  },
  {
    id: 'R8',
    modes: TRAINING,
    check: (c) =>
      c.cpuOffloadOptimizer && !supportsCpuOffload(c.frameworkPreset)
        ? [
            fix(
              'R8',
              'cpuOffloadOptimizer',
              true,
              false,
              'preset',
              'CPU optimizer offload turned off: needs a DeepSpeed ZeRO preset.',
            ),
          ]
        : [],
  },
  {
    id: 'R9',
    modes: BOTH,
    check: (c, { model }) => {
      const out: Correction[] = []
      const layers = model?.num_hidden_layers ?? Number.MAX_SAFE_INTEGER
      const l = clampInt(c.offloadLayers, 0, layers)
      if (l !== c.offloadLayers) {
        const beyondModel = model !== null && c.offloadLayers > layers
        out.push(
          fix(
            'R9',
            'offloadLayers',
            c.offloadLayers,
            l,
            beyondModel ? 'model' : 'range',
            beyondModel && model
              ? `Offloaded layers set to ${l}: ${model.name} has ${l} layers.`
              : `Offloaded layers set to ${l}: outside the allowed range.`,
          ),
        )
      }
      const p = Number.isFinite(c.offloadPercentage)
        ? Math.min(100, Math.max(0, c.offloadPercentage))
        : 0
      if (p !== c.offloadPercentage) {
        out.push(
          fix(
            'R9',
            'offloadPercentage',
            c.offloadPercentage,
            p,
            'range',
            `Offload percentage set to ${p}%: outside the allowed range.`,
          ),
        )
      }
      return out
    },
  },
  {
    id: 'R10',
    modes: BOTH,
    check: (c) =>
      RANGE_BOUNDS.flatMap(({ field, min, max }) => {
        const v = clampInt(c[field], min, max)
        return v === c[field]
          ? []
          : [
              fix(
                'R10',
                field,
                c[field],
                v,
                'range',
                `${RANGE_LABELS[field]} set to ${v.toLocaleString('en-US')}: outside the allowed range.`,
              ),
            ]
      }),
  },
  {
    id: 'R12',
    modes: INFERENCE,
    // kvCacheOffload only matters while offloading is on; a stale flag is inert.
    check: (c) =>
      c.offloadingEnabled && c.kvCacheOffload && c.kvTier.tier !== 'none'
        ? [
            fix(
              'R12',
              'kvTier',
              c.kvTier,
              { ...c.kvTier, tier: 'none' },
              'offload',
              'KV tier turned off: KV cache offload already keeps all KV off the GPU.',
            ),
          ]
        : [],
  },
  {
    id: 'R14',
    modes: INFERENCE,
    check: (c, { model, gpu }) => {
      if (c.shardingStrategy !== 'tensor-parallel' || !model) return []
      if (isValidTPDegree(model, c.numGPUs)) return []
      // Bounded by the GPU so a hostile count (1e9) never loops, whatever the rule order
      const degrees = validTPDegrees(model, Math.min(Math.floor(c.numGPUs), maxGPUsFor(gpu)))
      const t = degrees[degrees.length - 1] ?? 1
      return [
        fix(
          'R14',
          'numGPUs',
          c.numGPUs,
          t,
          'model',
          `GPU count set to ${t}: vLLM can't split ${model.name}'s ${model.num_attention_heads} attention heads across ${c.numGPUs} GPUs. Use pipeline parallel for ${c.numGPUs}.`,
        ),
      ]
    },
  },
]

export const MAX_NORMALIZE_PASSES = 4

export interface NormalizeResult<C extends RuleConfig> {
  config: C
  /** Every correction applied, in order (a field corrected twice appears twice) */
  corrections: Correction[]
  /** Passes run, including the final pass that changed nothing */
  passes: number
}

/**
 * Apply every rule live in config.mode until a pass changes nothing. Generic so the
 * store can pass its whole state and get the untouched fields back.
 *
 * @param rules - Override the table order; tests use it to prove order independence
 * @throws Error if the table ever fails to converge (a rule-table bug, not user input)
 */
export function normalizeConfig<C extends RuleConfig>(
  config: C,
  model: Model | null,
  gpu: GPU | null,
  rules: readonly Rule[] = RULES,
): NormalizeResult<C> {
  const ctx: RuleContext = { model, gpu }
  let current = config
  const corrections: Correction[] = []
  for (let pass = 1; pass <= MAX_NORMALIZE_PASSES; pass++) {
    let changed = false
    for (const rule of rules) {
      if (!rule.modes.includes(current.mode)) continue
      for (const correction of rule.check(current, ctx)) {
        current = { ...current, [correction.field]: correction.to } as C
        corrections.push(correction)
        changed = true
      }
    }
    if (!changed) return { config: current, corrections, passes: pass }
  }
  throw new Error(`normalizeConfig did not reach a fixpoint in ${MAX_NORMALIZE_PASSES} passes`)
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run src/engines/config-rules.test.ts`
Expected: PASS (all rows of the per-rule table, properties, R14 table).

- [ ] **Step 6: Typecheck and lint**

Run: `npm run typecheck && npm run lint:fix && npm run lint`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
rtk git add src/utils/schemas.ts src/engines/config-rules.ts src/engines/config-rules.test.ts
rtk git commit -m "feat(rules): one configuration rule table with normalizeConfig

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 1b: `allowedOptions`, soft warnings, notice builder

**Files:**
- Modify: `src/engines/config-rules.ts` (append)
- Test: `src/engines/config-rules.test.ts` (append)

**Interfaces:**
- Consumes: everything Task 1a produced; `KV_TIER_TYPES`, `KVTierType` (`src/engines/kv-tier.ts:13-15`).
- Produces (exported from `src/engines/config-rules.ts`):
  - `type AllowedOptionsInput = Pick<RuleConfig, 'mode' | 'shardingStrategy' | 'offloadingEnabled' | 'kvCacheOffload' | 'frameworkPreset'>`
  - `interface AllowedOptions { gpuCounts: number[]; strategies: ShardingStrategy[]; kvTiers: KVTierType[]; offloadTargets: OffloadTarget[]; interconnectOptions: string[]; cpuOffloadOptimizer: boolean }`
  - `function allowedOptions(input: AllowedOptionsInput, model: Model | null, gpu: GPU | null): AllowedOptions`
  - `type SoftWarningId = 'W3' | 'W6' | 'W8'`; `interface SoftWarning { id: SoftWarningId; message: string }`
  - `function softWarnings(config: Pick<RuleConfig, 'mode' | 'numGPUs' | 'numNodes' | 'shardingStrategy'>, model: Model | null, gpu: GPU | null): SoftWarning[]`
  - `interface Notice { title: string; lines: string[] }`
  - `type NoticeSource = 'gpu' | 'model' | 'mode' | 'link' | 'setting'`
  - `interface NoticeContext { model: Model | null; gpu: GPU | null; config: RuleConfig }`
  - `function buildNotice(corrections: Correction[], source: NoticeSource, ctx: NoticeContext): Notice | null`

- [ ] **Step 1: Write the failing tests**

Append to `src/engines/config-rules.test.ts`, and extend the import from `./config-rules` in the same edit with `allowedOptions, buildNotice, softWarnings`:

```ts
describe('allowedOptions', () => {
  const input = {
    mode: 'inference' as const,
    shardingStrategy: 'tensor-parallel' as const,
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'none' as const,
  }

  it('offers only valid TP degrees in inference tensor parallel (R1 + R14)', () => {
    expect(allowedOptions(input, L70, H100).gpuCounts).toEqual([1, 2, 4, 8])
    expect(allowedOptions(input, L70, NVL72).gpuCounts).toEqual([1, 2, 4, 8, 16, 32, 64])
    expect(allowedOptions(input, L70, M3).gpuCounts).toEqual([1])
  })

  it('offers every count up to the bound for pipeline parallel, training, or no model', () => {
    const all8 = [1, 2, 3, 4, 5, 6, 7, 8]
    expect(allowedOptions({ ...input, shardingStrategy: 'pipeline-parallel' }, L70, H100).gpuCounts).toEqual(all8)
    expect(allowedOptions({ ...input, mode: 'training' }, L70, H100).gpuCounts).toEqual(all8)
    expect(allowedOptions(input, null, H100).gpuCounts).toEqual(all8)
  })

  it('offers expert parallel only for a splittable MoE (R2)', () => {
    expect(allowedOptions(input, L70, H100).strategies).toEqual(['tensor-parallel', 'pipeline-parallel'])
    expect(allowedOptions(input, DSR1, H100).strategies).toContain('expert-parallel')
    expect(allowedOptions(input, null, H100).strategies).not.toContain('expert-parallel')
  })

  it('filters KV tiers by Grace host, unified memory and KV offload (R3, R6, R12)', () => {
    expect(allowedOptions(input, L70, H100).kvTiers).toEqual(['none', 'host-pcie', 'local-nvme', 'network'])
    expect(allowedOptions(input, L70, NVL72).kvTiers).toContain('host-grace')
    expect(allowedOptions(input, L8, UNIFIED_M3).kvTiers).toEqual(['none', 'local-nvme', 'network'])
    expect(
      allowedOptions({ ...input, offloadingEnabled: true, kvCacheOffload: true }, L70, H100).kvTiers,
    ).toEqual(['none'])
  })

  it('offers NVMe only on unified memory (R6)', () => {
    expect(allowedOptions(input, L8, UNIFIED_M3).offloadTargets).toEqual(['nvme'])
    expect(allowedOptions(input, L70, H100).offloadTargets).toEqual(['cpu-ram', 'nvme'])
  })

  it('offers interconnect variants only with two or more options on a multi-GPU part (R5, R13)', () => {
    expect(allowedOptions(input, L70, WITH_OPTIONS).interconnectOptions).toEqual(['nvlink-4', 'pcie-5'])
    expect(allowedOptions(input, L70, H100).interconnectOptions).toEqual([])
    const singleWithOptions = validateGPU({ ...M3, interconnect_options: ['pcie-5', 'nvlink-5'] })
    expect(allowedOptions(input, L70, singleWithOptions).interconnectOptions).toEqual([])
  })

  it('offers CPU optimizer offload only with a ZeRO preset and separate host memory (R6, R8)', () => {
    expect(allowedOptions({ ...input, frameworkPreset: 'deepspeed-zero3' }, L70, H100).cpuOffloadOptimizer).toBe(true)
    expect(allowedOptions({ ...input, frameworkPreset: 'unsloth' }, L70, H100).cpuOffloadOptimizer).toBe(false)
    expect(allowedOptions({ ...input, frameworkPreset: 'deepspeed-zero3' }, L8, UNIFIED_M3).cpuOffloadOptimizer).toBe(false)
  })

  it('never offers a value the store would correct', () => {
    const pairs: [Model, GPU][] = [[L70, H100], [KIMI_K3, H100], [GEMMA_1B, NVL72], [DSR1, NVL72], [L8, UNIFIED_M3]]
    for (const [model, gpu] of pairs) {
      const options = allowedOptions(input, model, gpu)
      for (const n of options.gpuCounts) {
        expect(normalizeConfig(cfg({ numGPUs: n }), model, gpu).corrections, `${model.id} ${n}`).toEqual([])
      }
      for (const tier of options.kvTiers) {
        const tiered = cfg({ kvTier: { ...DEFAULT_KV_TIER, tier } })
        expect(normalizeConfig(tiered, model, gpu).corrections, `${gpu.id} ${tier}`).toEqual([])
      }
    }
  })
})

describe('softWarnings', () => {
  const base = { mode: 'inference' as const, numGPUs: 1, numNodes: 1, shardingStrategy: 'tensor-parallel' as const }

  it('W3: multi-node clusters of single-GPU or unified-memory parts', () => {
    expect(softWarnings({ ...base, numNodes: 2 }, L70, M3).map((w) => w.id)).toEqual(['W3'])
    expect(softWarnings({ ...base, numNodes: 2 }, L70, H100)).toEqual([])
  })

  it('W6: more pipeline stages than layers', () => {
    const w = softWarnings({ ...base, numGPUs: 8, numNodes: 8, shardingStrategy: 'pipeline-parallel' }, L8, H100)
    expect(w).toEqual([
      { id: 'W6', message: `64 pipeline stages exceed ${L8.name}'s ${L8.num_hidden_layers} layers; some stages would be empty.` },
    ])
  })

  it('W8: experts that do not divide by the EP degree', () => {
    const uneven = softWarnings({ ...base, numGPUs: 6, shardingStrategy: 'expert-parallel' }, DSR1, H100)
    expect(uneven).toEqual([{ id: 'W8', message: `${DSR1.num_experts} experts don't split evenly across 6 GPUs.` }])
    expect(softWarnings({ ...base, numGPUs: 8, shardingStrategy: 'expert-parallel' }, DSR1, H100)).toEqual([])
  })

  it('is silent in training, where these inputs are inert', () => {
    expect(softWarnings({ ...base, mode: 'training', numNodes: 2 }, L70, M3)).toEqual([])
  })
})

describe('buildNotice', () => {
  it('returns null when nothing was corrected', () => {
    expect(buildNotice([], 'gpu', { model: L70, gpu: H100, config: cfg() })).toBeNull()
  })

  it('titles a GPU change after the GPU and lists one line per correction', () => {
    const r = normalizeConfig(cfg({ numGPUs: 64, kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' } }), L70, H100)
    const notice = buildNotice(r.corrections, 'gpu', { model: L70, gpu: H100, config: r.config })
    expect(notice?.title).toBe(`Adjusted for ${H100.name}`)
    expect(notice?.lines).toEqual([
      `GPU count set to 8: ${H100.name} supports at most 8 per server.`,
      `KV tier turned off: ${H100.name} has no Grace host memory.`,
    ])
  })

  it('titles a restored link "Shared link adjusted"', () => {
    const r = normalizeConfig(cfg({ batchSize: 0 }), L70, H100)
    expect(buildNotice(r.corrections, 'link', { model: L70, gpu: H100, config: r.config })?.title).toBe(
      'Shared link adjusted',
    )
  })

  it('titles a mode switch after the mode', () => {
    const r = normalizeConfig(cfg({ mode: 'training', frameworkPreset: 'vllm' }), L70, H100)
    expect(buildNotice(r.corrections, 'mode', { model: L70, gpu: H100, config: r.config })?.title).toBe(
      'Adjusted for fine-tuning mode',
    )
  })

  it('titles a plain setting change after the first correction subject', () => {
    const r = normalizeConfig(cfg({ numGPUs: 6 }), L70, H100)
    expect(buildNotice(r.corrections, 'setting', { model: L70, gpu: H100, config: r.config })?.title).toBe(
      `Adjusted for ${L70.name}`,
    )
  })

  it('keeps one line per rule and field, stating the final value, in first-appearance order', () => {
    const fix = (rule: RuleId, message: string): Correction => ({
      rule,
      field: 'numGPUs',
      from: 0,
      to: 0,
      subject: 'model',
      message,
    })
    const notice = buildNotice(
      [fix('R14', 'first snap'), fix('R1', 'clamp'), fix('R14', 'second snap')],
      'link',
      { model: L70, gpu: H100, config: cfg() },
    )
    expect(notice?.lines).toEqual(['second snap', 'clamp'])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/config-rules.test.ts`
Expected: FAIL with `allowedOptions is not a function` (and the same for `softWarnings`, `buildNotice`).

- [ ] **Step 3: Write the implementation**

In `src/engines/config-rules.ts`, change the kv-tier import (same edit as the first use below) to:

```ts
import {
  clampKVTier,
  DEFAULT_KV_TIER,
  graceLinkGBps,
  KV_TIER_TYPES,
  type KVTierSettings,
  type KVTierType,
} from './kv-tier'
```

and append:

```ts
export type AllowedOptionsInput = Pick<
  RuleConfig,
  'mode' | 'shardingStrategy' | 'offloadingEnabled' | 'kvCacheOffload' | 'frameworkPreset'
>

/** What the UI may offer, so an impossible value is never selectable (ADR 0004) */
export interface AllowedOptions {
  /** R1 + R14: ascending, always starts with 1 */
  gpuCounts: number[]
  /** R2 */
  strategies: ShardingStrategy[]
  /** R3, R6, R12 */
  kvTiers: KVTierType[]
  /** R6 */
  offloadTargets: OffloadTarget[]
  /** R5, R13: empty unless the part forms multi-GPU servers and has two or more variants */
  interconnectOptions: string[]
  /** R6 + R8 */
  cpuOffloadOptimizer: boolean
}

export function allowedOptions(
  input: AllowedOptionsInput,
  model: Model | null,
  gpu: GPU | null,
): AllowedOptions {
  const max = maxGPUsFor(gpu)
  const gpuCounts =
    input.mode === 'inference' && input.shardingStrategy === 'tensor-parallel' && model
      ? validTPDegrees(model, max)
      : Array.from({ length: max }, (_, i) => i + 1)

  const strategies: ShardingStrategy[] = ['tensor-parallel', 'pipeline-parallel']
  if (model && splitMoEParams(model)) strategies.push('expert-parallel')

  const unified = gpu?.unified_memory === true
  const kvTiers: KVTierType[] =
    input.offloadingEnabled && input.kvCacheOffload
      ? ['none']
      : KV_TIER_TYPES.filter((tier) => {
          if (tier === 'host-grace') return !unified && graceLinkGBps(gpu?.id ?? '') !== null
          if (tier === 'host-pcie') return !unified
          return true
        })

  const variants = gpu && gpu.max_gpus_per_node > 1 ? (gpu.interconnect_options ?? []) : []

  return {
    gpuCounts,
    strategies,
    kvTiers,
    offloadTargets: unified ? ['nvme'] : ['cpu-ram', 'nvme'],
    interconnectOptions: variants.length >= 2 ? [...variants] : [],
    cpuOffloadOptimizer: !unified && supportsCpuOffload(input.frameworkPreset),
  }
}

export type SoftWarningId = 'W3' | 'W6' | 'W8'

/** A soft rule: shown inline, the value is kept (spec Section 1). W1/W2/W4/W5 live where they already render. */
export interface SoftWarning {
  id: SoftWarningId
  message: string
}

export function softWarnings(
  config: Pick<RuleConfig, 'mode' | 'numGPUs' | 'numNodes' | 'shardingStrategy'>,
  model: Model | null,
  gpu: GPU | null,
): SoftWarning[] {
  if (config.mode !== 'inference') return []
  const out: SoftWarning[] = []
  if (config.numNodes > 1 && gpu && (gpu.unified_memory === true || gpu.max_gpus_per_node === 1)) {
    out.push({
      id: 'W3',
      message:
        'Small clusters: DGX Spark up to 4 units over 200 GbE, DGX Station up to 2; use the 200GbE fabric preset.',
    })
  }
  const stages =
    config.numNodes * (config.shardingStrategy === 'pipeline-parallel' ? config.numGPUs : 1)
  if (model && stages > model.num_hidden_layers) {
    out.push({
      id: 'W6',
      message: `${stages} pipeline stages exceed ${model.name}'s ${model.num_hidden_layers} layers; some stages would be empty.`,
    })
  }
  const experts = model?.num_experts
  if (
    config.shardingStrategy === 'expert-parallel' &&
    experts &&
    config.numGPUs > 1 &&
    experts % config.numGPUs !== 0
  ) {
    out.push({
      id: 'W8',
      message: `${experts} experts don't split evenly across ${config.numGPUs} GPUs.`,
    })
  }
  return out
}

export interface Notice {
  title: string
  lines: string[]
}

/** Which action produced the corrections: picks the notice title */
export type NoticeSource = 'gpu' | 'model' | 'mode' | 'link' | 'setting'

export interface NoticeContext {
  model: Model | null
  gpu: GPU | null
  /** The normalized config */
  config: RuleConfig
}

function subjectTitle(subject: CorrectionSubject, ctx: NoticeContext): string {
  switch (subject) {
    case 'gpu':
      return `Adjusted for ${ctx.gpu?.name ?? 'the selected GPU'}`
    case 'model':
      return `Adjusted for ${ctx.model?.name ?? 'the selected model'}`
    case 'mode':
      return `Adjusted for ${ctx.config.mode === 'training' ? 'fine-tuning' : 'inference'} mode`
    case 'preset':
      return `Adjusted for ${FRAMEWORK_PRESETS[ctx.config.frameworkPreset].name}`
    case 'offload':
      return 'Adjusted for KV cache offload'
    case 'range':
      return 'Adjusted to the allowed range'
  }
}

/**
 * One notice per user action (spec Section 1 "Notices"): "Adjusted for {GPU/model/mode}",
 * or "Shared link adjusted" on restore. One line per rule and field, in order of first
 * appearance, carrying that pair's last message (the final value).
 */
export function buildNotice(
  corrections: Correction[],
  source: NoticeSource,
  ctx: NoticeContext,
): Notice | null {
  const first = corrections[0]
  if (!first) return null
  const lines = new Map<string, string>()
  for (const c of corrections) lines.set(`${c.rule}:${c.field}`, c.message)
  let title: string
  if (source === 'link') title = 'Shared link adjusted'
  else if (source === 'setting') title = subjectTitle(first.subject, ctx)
  else title = subjectTitle(source, ctx)
  return { title, lines: [...lines.values()] }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/engines/config-rules.test.ts`
Expected: PASS.

- [ ] **Step 5: Typecheck, lint, commit**

Run: `npm run typecheck && npm run lint:fix && npm run lint`
Expected: no errors.

```bash
rtk git add src/engines/config-rules.ts src/engines/config-rules.test.ts
rtk git commit -m "feat(rules): allowedOptions, soft warnings W3/W6/W8, grouped notices

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 2a: Store integration and the notice channel

**Files:**
- Modify: `src/store/uiStore.ts` (full rewrite of the state creator, lines 1-285)
- Modify: `src/engines/kv-tier.ts:74-78` (comment) and `:118-132` (delete `resetTierForGPU`)
- Modify: `src/engines/kv-tier.test.ts:10` (import) and `:89-104` (delete the `resetTierForGPU` block)
- Modify: `src/utils/gpuLimits.ts:4-20` (comments)
- Test: `src/store/uiStore.test.ts`

**Interfaces:**
- Consumes: `normalizeConfig`, `pickRuleConfig`, `buildNotice`, `DEFAULT_RULE_CONFIG`, `RuleConfig`, `Notice`, `NoticeSource` (Tasks 1a/1b).
- Produces (exported from `src/store/uiStore.ts`):
  - `interface UIConfig extends RuleConfig { selectedModel: Model | null; selectedGPU: GPU | null; quantization: QuantizationFormat; kvQuantization: KVCachePrecision; interNodeFabric: FabricType; customFabric: CustomFabricInput | null; offloadMode: OffloadMode; trainingMethod: FineTuningMethod; optimizer: OptimizerType; trainingPrecision: TrainingPrecision; loraAlpha: number; targetModulesPercent: number; gradientCheckpointing: boolean; flashAttention: boolean }`
  - `const DEFAULT_UI_CONFIG: UIConfig`
  - `interface PendingNotice extends Notice { id: number }` (id increases on every notice, so an identical notice still re-fires React effects)
  - store state gains `pendingNotice: PendingNotice | null`, `restoreConfig(patch: Partial<UIConfig>): void`, `clearNotice(): void`; every existing setter keeps its name and signature.
  - `findModelById`, `findGPUById` unchanged.
- Removed: `resetTierForGPU` from `src/engines/kv-tier.ts` (only the store called it).

- [ ] **Step 1: Write the failing tests**

Append to `src/store/uiStore.test.ts`. In the same edit, change the imports at the top to:

```ts
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import type { GPU, Model } from '@utils/schemas'
import { validateGPU, validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'
```

and add below `findGPU`:

```ts
const models = validateModels(modelsData)

/** A real model row from the database, looked up by id (never hand-written). */
function findModel(id: string): Model {
  const model = models.find((m) => m.id === id)
  if (!model) throw new Error(`fixture model not found in models.json: ${id}`)
  return model
}

const L70 = findModel('meta-llama-llama-3.1-70b')
const DSR1 = findModel('deepseek-r1')
const H100 = findGPU('nvidia-h100-80gb-sxm')
const NVL72 = findGPU('nvidia-gb300-nvl72')
// Task 3a sets unified_memory in the data and replaces this with findGPU('apple-m3-ultra').
const UNIFIED_M3 = validateGPU({ ...findGPU('apple-m3-ultra'), unified_memory: true })

async function freshStore() {
  const { useUIStore, DEFAULT_UI_CONFIG } = await import('@store/uiStore')
  useUIStore.setState({ ...DEFAULT_UI_CONFIG, pendingNotice: null })
  return useUIStore
}

describe('uiStore: every action normalizes, one notice per action', () => {
  it('setNumGPUs(6) on Llama 3.1 70B snaps to 4 with one notice titled for the model', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(6)
    const state = store.getState()
    expect(state.numGPUs).toBe(4)
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${L70.name}`)
    expect(state.pendingNotice?.lines).toHaveLength(1)
  })

  it('a GPU switch reports the count clamp and the Grace tier reset in ONE notice', async () => {
    const store = await freshStore()
    store.setState({
      selectedModel: L70,
      selectedGPU: NVL72,
      numGPUs: 64,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' },
    })
    store.getState().setSelectedGPU(H100)
    const state = store.getState()
    expect(state.numGPUs).toBe(8)
    expect(state.kvTier.tier).toBe('none')
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${H100.name}`)
    expect(state.pendingNotice?.lines).toHaveLength(2)
  })

  it('leaves pendingNotice untouched when nothing is corrected', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(4)
    expect(store.getState().pendingNotice).toBeNull()
  })

  it('gives an identical repeated notice a new id, so it toasts again', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(6)
    const first = store.getState().pendingNotice
    store.getState().setNumGPUs(6)
    const second = store.getState().pendingNotice
    expect(second?.lines).toEqual(first?.lines)
    expect(second?.id).not.toBe(first?.id)
  })

  it('model change to a dense model resets EP and snaps 6 -> 4 in one notice (R2 then R14)', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: DSR1, selectedGPU: H100, shardingStrategy: 'expert-parallel', numGPUs: 6 })
    store.getState().setSelectedModel(L70)
    const state = store.getState()
    expect(state.shardingStrategy).toBe('tensor-parallel')
    expect(state.numGPUs).toBe(4)
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${L70.name}`)
    expect(state.pendingNotice?.lines).toHaveLength(2)
  })

  it('training ZeRO-3 on 6 GPUs keeps 6 silently; switching to inference snaps to 4 with a mode notice', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, mode: 'training', frameworkPreset: 'deepspeed-zero3' })
    store.getState().setNumGPUs(6)
    expect(store.getState().numGPUs).toBe(6)
    expect(store.getState().pendingNotice).toBeNull()
    store.getState().setMode('inference')
    expect(store.getState().numGPUs).toBe(4)
    expect(store.getState().pendingNotice?.title).toBe('Adjusted for inference mode')
  })

  it('picking vLLM in training switches to inference (action intent) and keeps the preset', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, mode: 'training' })
    store.getState().setFrameworkPreset('vllm')
    expect(store.getState().mode).toBe('inference')
    expect(store.getState().frameworkPreset).toBe('vllm')
  })

  it('switching to training with vLLM clears the preset (R7), titled for the mode', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, frameworkPreset: 'vllm' })
    store.getState().setMode('training')
    expect(store.getState().frameworkPreset).toBe('none')
    expect(store.getState().pendingNotice?.title).toBe('Adjusted for fine-tuning mode')
  })

  it('switching ZeRO-3 -> Unsloth turns optimizer offload off (R8) and keeps auto-optimizations', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, mode: 'training' })
    store.getState().setFrameworkPreset('deepspeed-zero3')
    store.getState().setCpuOffloadOptimizer(true)
    expect(store.getState().cpuOffloadOptimizer).toBe(true)
    store.getState().setFrameworkPreset('unsloth')
    const state = store.getState()
    expect(state.cpuOffloadOptimizer).toBe(false)
    expect(state.optimizer).toBe('adamw-8bit')
    expect(state.pendingNotice?.lines).toEqual([
      'CPU optimizer offload turned off: needs a DeepSpeed ZeRO preset.',
    ])
  })

  it('enabling offloading on unified memory with target cpu-ram switches to NVMe, with no notice', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: UNIFIED_M3, offloadTarget: 'cpu-ram' })
    store.getState().setOffloadingEnabled(true)
    const state = store.getState()
    expect(state.offloadingEnabled).toBe(true)
    expect(state.offloadTarget).toBe('nvme')
    expect(state.pendingNotice).toBeNull()
  })

  it('clearNotice empties the channel', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(6)
    store.getState().clearNotice()
    expect(store.getState().pendingNotice).toBeNull()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/store/uiStore.test.ts`
Expected: FAIL: `DEFAULT_UI_CONFIG` is undefined (so `setState({...undefined})`), `numGPUs` stays 6, `pendingNotice` undefined.

- [ ] **Step 3: Rewrite the store**

Replace `src/store/uiStore.ts` with:

```ts
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import {
  buildNotice,
  DEFAULT_RULE_CONFIG,
  type Notice,
  type NoticeSource,
  normalizeConfig,
  pickRuleConfig,
  type RuleConfig,
} from '@engines/config-rules'
import { FRAMEWORK_PRESETS, type FrameworkPreset } from '@engines/frameworks'
import type { KVTierSettings } from '@engines/kv-tier'
import type {
  FabricType,
  FineTuningMethod,
  KVCachePrecision,
  OffloadMode,
  OffloadTarget,
  OptimizerType,
  QuantizationFormat,
  ShardingStrategy,
  TrainingPrecision,
} from '@engines/types'
import type { CustomFabricInput, GPU, Model } from '@utils/schemas'
import { validateGPUs, validateModels } from '@utils/schemas'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

// Load and validate data once at module level
const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)

/**
 * Find a model by ID in the curated database
 */
export function findModelById(id: string): Model | null {
  return models.find((m) => m.id === id) ?? null
}

/**
 * Find a GPU by ID in the curated database
 */
export function findGPUById(id: string): GPU | null {
  return gpus.find((g) => g.id === id) ?? null
}

/**
 * Every calculator input. The rule fields (RuleConfig) are normalized by
 * config-rules on every action; the rest have no dependencies.
 *
 * numGPUs is the parallel degree of one replica inside one server (ADR 0006).
 */
export interface UIConfig extends RuleConfig {
  // Model and GPU selections (not persisted - large objects, may become stale)
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  kvQuantization: KVCachePrecision
  interNodeFabric: FabricType
  customFabric: CustomFabricInput | null
  offloadMode: OffloadMode
  trainingMethod: FineTuningMethod
  optimizer: OptimizerType
  trainingPrecision: TrainingPrecision
  loraAlpha: number
  targetModulesPercent: number
  gradientCheckpointing: boolean
  flashAttention: boolean
}

export const DEFAULT_UI_CONFIG: UIConfig = {
  ...DEFAULT_RULE_CONFIG,
  selectedModel: null,
  selectedGPU: null,
  quantization: 'fp16',
  kvQuantization: 'fp16',
  interNodeFabric: 'ethernet-800g',
  customFabric: null,
  offloadMode: 'percentage',
  trainingMethod: 'lora',
  optimizer: 'adamw',
  trainingPrecision: 'bf16',
  loraAlpha: 32,
  targetModulesPercent: 30,
  gradientCheckpointing: false,
  flashAttention: false,
}

/** A notice waiting to be shown; id increases on every notice so a repeat re-fires effects */
export interface PendingNotice extends Notice {
  id: number
}

interface UIState extends UIConfig {
  // UI preferences (persisted)
  isDarkMode: boolean

  /** Corrections from the last action, for useConfigNotices to toast */
  pendingNotice: PendingNotice | null

  // Actions
  setSelectedModel: (model: Model | null) => void
  setSelectedGPU: (gpu: GPU | null) => void
  setInterconnectOverride: (v: string | null) => void
  setQuantization: (quantization: QuantizationFormat) => void
  setSequenceLength: (sequenceLength: number) => void
  setBatchSize: (batchSize: number) => void
  setKVQuantization: (kvQuantization: KVCachePrecision) => void
  setNumGPUs: (numGPUs: number) => void
  setNumNodes: (numNodes: number) => void
  setInterNodeFabric: (fabric: FabricType) => void
  setCustomFabric: (fabric: CustomFabricInput | null) => void
  setShardingStrategy: (strategy: ShardingStrategy) => void
  setOffloadingEnabled: (enabled: boolean) => void
  setOffloadTarget: (target: OffloadTarget) => void
  setOffloadMode: (mode: OffloadMode) => void
  setOffloadPercentage: (percentage: number) => void
  setOffloadLayers: (layers: number) => void
  setKVCacheOffload: (enabled: boolean) => void
  setOffloadHostCapacityGB: (gb: number | null) => void
  setMode: (mode: 'inference' | 'training') => void
  setTrainingMethod: (method: FineTuningMethod) => void
  setOptimizer: (optimizer: OptimizerType) => void
  setTrainingPrecision: (precision: TrainingPrecision) => void
  setLoraRank: (rank: number) => void
  setLoraAlpha: (alpha: number) => void
  setTargetModulesPercent: (percent: number) => void
  setGradientAccumulationSteps: (steps: number) => void
  setGradientCheckpointing: (enabled: boolean) => void
  setFlashAttention: (enabled: boolean) => void
  setFrameworkPreset: (preset: FrameworkPreset) => void
  setCpuOffloadOptimizer: (enabled: boolean) => void
  setConcurrentUsers: (n: number) => void
  setKVTier: (patch: Partial<KVTierSettings>) => void
  /** Apply a whole shared-link configuration, normalized once ("Shared link adjusted") */
  restoreConfig: (patch: Partial<UIConfig>) => void
  clearNotice: () => void
  setIsDarkMode: (dark: boolean) => void
  toggleDarkMode: () => void
}

let noticeSeq = 0

/**
 * The one write path for configuration (ADR 0004): merge the patch, normalize the
 * whole config once, and publish every correction as one notice.
 */
function commit(
  set: (partial: Partial<UIState>) => void,
  get: () => UIState,
  patch: Partial<UIConfig>,
  source: NoticeSource,
): void {
  const next = { ...get(), ...patch }
  const { config, corrections } = normalizeConfig(
    pickRuleConfig(next),
    next.selectedModel,
    next.selectedGPU,
  )
  const notice = buildNotice(corrections, source, {
    model: next.selectedModel,
    gpu: next.selectedGPU,
    config,
  })
  set({
    ...patch,
    ...config,
    ...(notice ? { pendingNotice: { ...notice, id: ++noticeSeq } } : {}),
  })
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      ...DEFAULT_UI_CONFIG,
      pendingNotice: null,
      isDarkMode:
        typeof window !== 'undefined'
          ? window.matchMedia('(prefers-color-scheme: dark)').matches
          : false,

      // Actions: every config write goes through commit()
      setSelectedModel: (model) => commit(set, get, { selectedModel: model }, 'model'),
      setSelectedGPU: (gpu) => commit(set, get, { selectedGPU: gpu }, 'gpu'),
      setInterconnectOverride: (v) => commit(set, get, { interconnectOverride: v }, 'setting'),
      setQuantization: (quantization) => commit(set, get, { quantization }, 'setting'),
      setSequenceLength: (sequenceLength) => commit(set, get, { sequenceLength }, 'setting'),
      setBatchSize: (batchSize) => commit(set, get, { batchSize }, 'setting'),
      setKVQuantization: (kvQuantization) => commit(set, get, { kvQuantization }, 'setting'),
      setNumGPUs: (numGPUs) => commit(set, get, { numGPUs }, 'setting'),
      setNumNodes: (numNodes) => commit(set, get, { numNodes }, 'setting'),
      setInterNodeFabric: (interNodeFabric) => commit(set, get, { interNodeFabric }, 'setting'),
      setCustomFabric: (customFabric) => commit(set, get, { customFabric }, 'setting'),
      setShardingStrategy: (shardingStrategy) =>
        commit(set, get, { shardingStrategy }, 'setting'),
      setOffloadingEnabled: (enabled) => {
        const { selectedGPU, offloadTarget } = get()
        // Action intent, not a rule: on unified memory the only offload target is NVMe,
        // so turning offloading on picks it instead of letting R6 switch it off again.
        const toNVMe = enabled && selectedGPU?.unified_memory === true && offloadTarget === 'cpu-ram'
        commit(
          set,
          get,
          toNVMe ? { offloadingEnabled: true, offloadTarget: 'nvme' } : { offloadingEnabled: enabled },
          'setting',
        )
      },
      setOffloadTarget: (offloadTarget) => commit(set, get, { offloadTarget }, 'setting'),
      setOffloadMode: (offloadMode) => commit(set, get, { offloadMode }, 'setting'),
      setOffloadPercentage: (offloadPercentage) =>
        commit(set, get, { offloadPercentage }, 'setting'),
      setOffloadLayers: (offloadLayers) => commit(set, get, { offloadLayers }, 'setting'),
      setKVCacheOffload: (kvCacheOffload) => commit(set, get, { kvCacheOffload }, 'setting'),
      setOffloadHostCapacityGB: (offloadHostCapacityGB) =>
        commit(set, get, { offloadHostCapacityGB }, 'setting'),
      setMode: (mode) => commit(set, get, { mode }, 'mode'),
      setTrainingMethod: (trainingMethod) => commit(set, get, { trainingMethod }, 'setting'),
      setOptimizer: (optimizer) => commit(set, get, { optimizer }, 'setting'),
      setTrainingPrecision: (trainingPrecision) =>
        commit(set, get, { trainingPrecision }, 'setting'),
      setLoraRank: (loraRank) => commit(set, get, { loraRank }, 'setting'),
      setLoraAlpha: (loraAlpha) => commit(set, get, { loraAlpha }, 'setting'),
      setTargetModulesPercent: (targetModulesPercent) =>
        commit(set, get, { targetModulesPercent }, 'setting'),
      setGradientAccumulationSteps: (gradientAccumulationSteps) =>
        commit(set, get, { gradientAccumulationSteps }, 'setting'),
      setGradientCheckpointing: (gradientCheckpointing) =>
        commit(set, get, { gradientCheckpointing }, 'setting'),
      setFlashAttention: (flashAttention) => commit(set, get, { flashAttention }, 'setting'),
      setFrameworkPreset: (preset) => {
        const { mode, autoOptimizations } = FRAMEWORK_PRESETS[preset]
        commit(
          set,
          get,
          {
            frameworkPreset: preset,
            // Action intent, not a rule: picking vLLM/TGI means inference. Training
            // presets are kept when switching to inference (inert, preserved for the
            // round trip), so this never runs the other way.
            ...(mode === 'inference' ? { mode: 'inference' as const } : {}),
            ...(autoOptimizations.gradientCheckpointing !== undefined
              ? { gradientCheckpointing: autoOptimizations.gradientCheckpointing }
              : {}),
            ...(autoOptimizations.flashAttention !== undefined
              ? { flashAttention: autoOptimizations.flashAttention }
              : {}),
            ...(autoOptimizations.optimizer !== undefined
              ? { optimizer: autoOptimizations.optimizer }
              : {}),
          },
          'setting',
        )
      },
      setCpuOffloadOptimizer: (cpuOffloadOptimizer) =>
        commit(set, get, { cpuOffloadOptimizer }, 'setting'),
      setConcurrentUsers: (concurrentUsers) => commit(set, get, { concurrentUsers }, 'setting'),
      setKVTier: (patch) => commit(set, get, { kvTier: { ...get().kvTier, ...patch } }, 'setting'),
      restoreConfig: (patch) => commit(set, get, patch, 'link'),
      clearNotice: () => set({ pendingNotice: null }),
      setIsDarkMode: (dark) => set({ isDarkMode: dark }),
      toggleDarkMode: () => set({ isDarkMode: !get().isDarkMode }),
    }),
    {
      name: 'llmvram-ui-preferences',
      // Only persist dark mode preference - all other state managed via URL hash
      partialize: (state) => ({
        isDarkMode: state.isDarkMode,
      }),
    },
  ),
)
```

- [ ] **Step 4: Remove `resetTierForGPU` and update the comments that pointed at the old checks**

In `src/engines/kv-tier.ts` delete lines 118-132 (the `resetTierForGPU` doc comment and function), change the `tierBandwidthGBps` doc line 138 to `(shouldn't happen in practice — config-rules R3 keeps the two in sync).`, and replace the comment above `clampKVTier` (lines 74-78) with:

```ts
/**
 * Bounds for tier settings: active share 1-100%, burst >= 1 s, and a custom
 * bandwidth or capacity that is positive and finite, else null. Applied by
 * config-rules R4 (with a notice), so the engine below can trust its input.
 */
```

In `src/engines/kv-tier.test.ts` remove `resetTierForGPU,` from the import list (line 10) and delete the `describe('resetTierForGPU', ...)` block (lines 89-104).

In `src/utils/gpuLimits.ts` replace lines 4-20 (both doc comments) with:

```ts
/**
 * The per-node GPU bound for a selection, or the engine's sanity bound with none.
 *
 * Single definition on purpose: config-rules R1 and allowedOptions read it, so the
 * UI never offers a count the store corrects.
 */
```

and put above `clampGPUCount`:

```ts
/**
 * Clamp a GPU count to what the selected GPU can form (GPU.max_gpus_per_node).
 *
 * The store no longer calls this: config-rules R1 corrects numGPUs and always shows
 * the correction. Recommendations uses it to bound a suggested count.
 */
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run src/store/uiStore.test.ts src/engines/kv-tier.test.ts src/engines/config-rules.test.ts`
Expected: PASS, including the four pre-existing host-grace and host-capacity tests (R3 and R4 now enforce them).

- [ ] **Step 6: Full suite, typecheck, lint**

Run: `npx vitest run && npm run typecheck && npm run lint:fix && npm run lint`
Expected: all green. Component tests mock `@store/uiStore` with their own stores and are unaffected.

- [ ] **Step 7: Commit**

```bash
rtk git add src/store/uiStore.ts src/store/uiStore.test.ts src/engines/kv-tier.ts src/engines/kv-tier.test.ts src/utils/gpuLimits.ts
rtk git commit -m "feat(store): every action normalizes through the rule table; one notice per action

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---

### Task 2b: URL restore rebuilt (whole config, normalized once)

**Files:**
- Modify: `src/store/urlSerializer.ts:48-55` (loosen `sl`, `bs`, `cu`), `:67` (`nn`), after `:124` (new `io`), `:139-172` (param type), `:205-272` (serialize `io`), append `urlStateToConfig`
- Modify: `src/hooks/useURLSync.ts:1-157` (restore effect)
- Test: `src/store/urlSerializer.test.ts`, `src/store/uiStore.test.ts`

**Interfaces:**
- Consumes: `UIConfig`, `restoreConfig`, `DEFAULT_UI_CONFIG` (Task 2a); `DEFAULT_KV_TIER`.
- Produces (exported from `src/store/urlSerializer.ts`):
  - `URLStateSchema` gains `io: z.string().optional()` (interconnectOverride); `sl`, `bs`, `ng` are plain `z.number()`; `cu`, `nn` are `z.number().optional()` (R10 corrects them after restore).
  - `serializeToURL(state)` param gains `interconnectOverride: string | null`.
  - `interface URLLookups { findModel: (id: string) => Model | null; findGPU: (id: string) => GPU | null }`
  - `interface RestoredConfig { patch: Partial<UIConfig>; missing: string[] }`
  - `function urlStateToConfig(state: URLState, lookups: URLLookups): RestoredConfig` (pure; `fp` restored raw, no auto-optimizations; restores `ga`, `gc`, `fa`, `co`, `io`)

- [ ] **Step 1: Write the failing tests**

Append to `src/store/urlSerializer.test.ts`, and in the same edit replace its imports with:

```ts
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { type GPU, type Model, validateGPUs, validateModels } from '@utils/schemas'
import { compressToEncodedURIComponent } from 'lz-string'
import { describe, expect, it } from 'vitest'
import { deserializeFromURL, isCustomId, serializeToURL, urlStateToConfig } from './urlSerializer'
```

```ts
const realModels = validateModels(modelsData)
const realGPUs = validateGPUs(gpusData)
const lookups = {
  findModel: (id: string): Model | null => realModels.find((m) => m.id === id) ?? null,
  findGPU: (id: string): GPU | null => realGPUs.find((g) => g.id === id) ?? null,
}
function realModel(id: string): Model {
  const m = lookups.findModel(id)
  if (!m) throw new Error(`fixture model not found in models.json: ${id}`)
  return m
}
function realGPU(id: string): GPU {
  const g = lookups.findGPU(id)
  if (!g) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return g
}

describe('urlStateToConfig', () => {
  const everyKey = {
    ...baseState,
    selectedModel: realModel('meta-llama-llama-3.1-70b'),
    selectedGPU: realGPU('nvidia-h100-80gb-sxm'),
    quantization: 'fp8' as const,
    sequenceLength: 32768,
    batchSize: 8,
    kvQuantization: 'fp8' as const,
    numGPUs: 4,
    shardingStrategy: 'pipeline-parallel' as const,
    concurrentUsers: 64,
    kvTier: { tier: 'network' as const, customGBps: 20, activeShare: 0.5, burstSeconds: 12, capacityTB: 3 },
    numNodes: 2,
    interNodeFabric: 'custom' as const,
    customFabric: { name: 'Lab', port_gbps: 25 },
    offloadingEnabled: true,
    offloadTarget: 'nvme' as const,
    offloadMode: 'layers' as const,
    offloadPercentage: 10,
    offloadLayers: 12,
    kvCacheOffload: true,
    offloadHostCapacityGB: 4096,
    mode: 'training' as const,
    trainingMethod: 'qlora' as const,
    optimizer: 'adafactor' as const,
    trainingPrecision: 'fp16' as const,
    loraRank: 64,
    loraAlpha: 128,
    targetModulesPercent: 50,
    gradientAccumulationSteps: 8,
    gradientCheckpointing: true,
    flashAttention: true,
    frameworkPreset: 'deepspeed-zero3' as const,
    cpuOffloadOptimizer: true,
    interconnectOverride: 'pcie-5',
  }

  it('round-trips every serialized key, including ga, gc, fa, fp, co and the new io', () => {
    const decoded = deserializeFromURL(serializeToURL(everyKey))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch, missing } = urlStateToConfig(decoded, lookups)
    expect(missing).toEqual([])
    const { selectedModel, selectedGPU, ...rest } = everyKey
    expect(patch.selectedModel?.id).toBe(selectedModel.id)
    expect(patch.selectedGPU?.id).toBe(selectedGPU.id)
    expect(patch).toMatchObject(rest)
  })

  it('restores fp raw: no auto-optimizations overwrite the link optimizer and flags', () => {
    const state = { ...everyKey, frameworkPreset: 'unsloth' as const, optimizer: 'adamw' as const, gradientCheckpointing: false }
    const decoded = deserializeFromURL(serializeToURL(state))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, lookups)
    expect(patch.frameworkPreset).toBe('unsloth')
    expect(patch.optimizer).toBe('adamw')
    expect(patch.gradientCheckpointing).toBe(false)
  })

  it('defaults absent keys like links made before they existed', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(JSON.stringify({ q: 'fp16', sl: 4096, bs: 1, kvq: 'fp16', ng: 1, ss: 'tensor-parallel' })),
    )
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, lookups)
    expect(patch).toMatchObject({ mode: 'inference', concurrentUsers: 1, numNodes: 1, kvTier: DEFAULT_KV_TIER, interconnectOverride: null })
  })

  it('reports an unknown model or GPU id without custom params', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(JSON.stringify({ modelId: 'gone', gpuId: 'gone', q: 'fp16', sl: 4096, bs: 1, kvq: 'fp16', ng: 1, ss: 'tensor-parallel' })),
    )
    if (!decoded) throw new Error('expected the link to parse')
    expect(urlStateToConfig(decoded, lookups).missing).toEqual([
      'Model from shared link not found in database',
      'GPU from shared link not found in database',
    ])
  })

  it('parses out-of-range numbers instead of rejecting the whole link (R10 corrects them)', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(JSON.stringify({ q: 'fp16', sl: 100, bs: 0, kvq: 'fp16', ng: -3, ss: 'tensor-parallel', nn: 12, cu: 0 })),
    )
    expect(decoded).toMatchObject({ sl: 100, bs: 0, ng: -3, nn: 12, cu: 0 })
  })
})
```

Append to `src/store/uiStore.test.ts`, adding `import { compressToEncodedURIComponent } from 'lz-string'` to its imports in the same edit:

```ts
describe('uiStore: shared-link restore is one normalized action', () => {
  async function restore(json: Record<string, unknown>) {
    const store = await freshStore()
    const { deserializeFromURL, urlStateToConfig } = await import('@store/urlSerializer')
    const { findGPUById, findModelById } = await import('@store/uiStore')
    const decoded = deserializeFromURL(compressToEncodedURIComponent(JSON.stringify(json)))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, { findModel: findModelById, findGPU: findGPUById })
    store.getState().restoreConfig(patch)
    return store.getState()
  }
  const base = { q: 'fp16', sl: 4096, bs: 1, kvq: 'fp16', ng: 1, ss: 'tensor-parallel' }

  it('opens a hostile link corrected, with one "Shared link adjusted" notice', async () => {
    const state = await restore({
      ...base,
      modelId: L70.id,
      gpuId: H100.id,
      ss: 'expert-parallel',
      ng: 6,
      sl: 100,
      bs: 0,
    })
    expect(state.shardingStrategy).toBe('tensor-parallel')
    expect(state.numGPUs).toBe(4)
    expect(state.sequenceLength).toBe(512)
    expect(state.batchSize).toBe(1)
    expect(state.pendingNotice?.title).toBe('Shared link adjusted')
    expect(state.pendingNotice?.lines).toHaveLength(4)
  })

  it('clamps non-integer and negative counts', async () => {
    expect((await restore({ ...base, modelId: L70.id, gpuId: H100.id, ng: 2.5 })).numGPUs).toBe(2)
    expect((await restore({ ...base, modelId: L70.id, gpuId: H100.id, ng: -3 })).numGPUs).toBe(1)
    expect((await restore({ ...base, modelId: L70.id, gpuId: H100.id, nn: 12, cu: 0 })).numNodes).toBe(8)
  })

  it('keeps EP and 6 GPUs when the link model is unknown; the first model pick then corrects both', async () => {
    const state = await restore({ ...base, modelId: 'no-such-model', gpuId: H100.id, ss: 'expert-parallel', ng: 6 })
    expect(state.selectedModel).toBeNull()
    expect(state.shardingStrategy).toBe('expert-parallel')
    expect(state.numGPUs).toBe(6)
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedModel(L70)
    expect(useUIStore.getState().shardingStrategy).toBe('tensor-parallel')
    expect(useUIStore.getState().numGPUs).toBe(4)
    expect(useUIStore.getState().pendingNotice?.lines).toHaveLength(2)
  })

  it('restores training settings that links carried but never restored (ga, gc, fa, co)', async () => {
    const state = await restore({
      ...base,
      modelId: L70.id,
      gpuId: H100.id,
      m: 'training',
      ga: 8,
      gc: true,
      fa: true,
      fp: 'deepspeed-zero3',
      co: true,
    })
    expect(state).toMatchObject({
      mode: 'training',
      gradientAccumulationSteps: 8,
      gradientCheckpointing: true,
      flashAttention: true,
      frameworkPreset: 'deepspeed-zero3',
      cpuOffloadOptimizer: true,
    })
    expect(state.pendingNotice).toBeNull()
  })

  it('gives the same result whatever order the link keys come in', async () => {
    const link = { ...base, modelId: L70.id, gpuId: H100.id, ss: 'expert-parallel', ng: 6, bs: 0, sl: 100, nn: 12 }
    const entries = Object.entries(link)
    const orders = [entries, [...entries].reverse(), [...entries].sort(([a], [b]) => a.localeCompare(b))]
    const results = []
    for (const order of orders) {
      const state = await restore(Object.fromEntries(order))
      results.push({
        numGPUs: state.numGPUs,
        shardingStrategy: state.shardingStrategy,
        batchSize: state.batchSize,
        sequenceLength: state.sequenceLength,
        numNodes: state.numNodes,
        lines: state.pendingNotice?.lines,
      })
    }
    expect(results[1]).toEqual(results[0])
    expect(results[2]).toEqual(results[0])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/store/urlSerializer.test.ts src/store/uiStore.test.ts`
Expected: FAIL: `urlStateToConfig is not a function`; the out-of-range link returns `null` (schema `min(512)`).

- [ ] **Step 3: Implement the serializer changes**

In `src/store/urlSerializer.ts`:

Replace lines 49-55 with:

```ts
  q: z.string(), // quantization
  // Numbers are NOT range-checked here: a hand-edited or old link with an
  // out-of-range value must still open. config-rules R1/R10 correct it after
  // restore, with a "Shared link adjusted" notice.
  sl: z.number(), // sequenceLength
  bs: z.number(), // batchSize
  kvq: z.string(), // kvQuantization
  ng: z.number(), // numGPUs — parallel degree of one replica, per server; see CHANGELOG.md "Multi-node inference" entry
  ss: z.string(), // shardingStrategy
  cu: z.number().optional(), // concurrentUsers (absent = 1)
```

Replace line 67 (`nn`) with:

```ts
  nn: z.number().optional(), // numNodes (R10 bounds it to 1-8)
```

After the `co` line (124) add:

```ts
  io: z.string().optional(), // interconnectOverride (absent = the GPU's default)
```

In the `serializeToURL` parameter type, after `cpuOffloadOptimizer: boolean` add `interconnectOverride: string | null`. In the returned object, after the `...(state.frameworkPreset !== 'none' ...)` spread add:

```ts
    // Interconnect variant (only when overridden)
    ...(state.interconnectOverride ? { io: state.interconnectOverride } : {}),
```

Remove the now-unused `MAX_CONCURRENT_USERS` and `MAX_SEQUENCE_LENGTH` imports in the same edit that removes their last uses (lines 1 and 16).

Append:

```ts
/** How restore resolves ids against the curated databases */
export interface URLLookups {
  findModel: (id: string) => Model | null
  findGPU: (id: string) => GPU | null
}

export interface RestoredConfig {
  /** The whole configuration the link describes; the store normalizes it once */
  patch: Partial<UIConfig>
  /** User-facing warnings for ids that resolved to nothing */
  missing: string[]
}

/**
 * Map a parsed link onto the store's configuration, without applying any rule:
 * the store normalizes the whole patch in one step (spec Section 2). fp is taken
 * raw: running setFrameworkPreset would re-apply its auto-optimizations and
 * overwrite the link's optimizer, gc and fa.
 */
export function urlStateToConfig(state: URLState, lookups: URLLookups): RestoredConfig {
  const patch: Partial<UIConfig> = {}
  const missing: string[] = []

  if (state.modelId) {
    const model = lookups.findModel(state.modelId)
    if (model) patch.selectedModel = model
    else if (!state.customModel) missing.push('Model from shared link not found in database')
  }
  if (!patch.selectedModel && state.customModel) {
    patch.selectedModel = { id: 'custom-restored', architecture: 'dense', ...state.customModel }
  }

  if (state.gpuId) {
    const gpu = lookups.findGPU(state.gpuId)
    if (gpu) patch.selectedGPU = gpu
    else if (!state.customGPU) missing.push('GPU from shared link not found in database')
  }
  if (!patch.selectedGPU && state.customGPU) {
    patch.selectedGPU = {
      id: 'custom-restored',
      manufacturer: 'nvidia',
      memory_type: 'Custom',
      bus_width: 0,
      tier: 'consumer',
      interconnect: 'none',
      max_gpus_per_node: 8,
      ...state.customGPU,
    }
  }

  patch.quantization = state.q as QuantizationFormat
  patch.sequenceLength = state.sl
  patch.batchSize = state.bs
  patch.kvQuantization = state.kvq as KVCachePrecision
  patch.numGPUs = state.ng
  patch.shardingStrategy = state.ss as ShardingStrategy
  // Absent in links made before these keys existed: 1 user, no tier, one node
  patch.concurrentUsers = state.cu ?? 1
  patch.kvTier = state.kt
    ? {
        tier: state.kt.t,
        customGBps: state.kt.g ?? null,
        activeShare: state.kt.a,
        burstSeconds: state.kt.b,
        capacityTB: state.kt.c ?? null,
      }
    : DEFAULT_KV_TIER
  patch.numNodes = state.nn ?? 1
  if (state.fab) patch.interNodeFabric = state.fab
  if (state.fabc) patch.customFabric = state.fabc

  // Offloading keys are only written while offloading is enabled
  if (state.oe) {
    patch.offloadingEnabled = true
    if (state.ot) patch.offloadTarget = state.ot as OffloadTarget
    if (state.om) patch.offloadMode = state.om as OffloadMode
    if (state.op !== undefined) patch.offloadPercentage = state.op
    if (state.ol !== undefined) patch.offloadLayers = state.ol
    if (state.ko !== undefined) patch.kvCacheOffload = state.ko
    if (state.hc !== undefined) patch.offloadHostCapacityGB = state.hc
  }

  patch.mode = state.m ?? 'inference'
  if (state.m === 'training') {
    if (state.tm) patch.trainingMethod = state.tm
    if (state.to) patch.optimizer = state.to
    if (state.tp) patch.trainingPrecision = state.tp
    if (state.lr !== undefined) patch.loraRank = state.lr
    if (state.la !== undefined) patch.loraAlpha = state.la
    if (state.tmp !== undefined) patch.targetModulesPercent = state.tmp
    if (state.ga !== undefined) patch.gradientAccumulationSteps = state.ga
    if (state.gc !== undefined) patch.gradientCheckpointing = state.gc
    if (state.fa !== undefined) patch.flashAttention = state.fa
    patch.cpuOffloadOptimizer = state.co ?? false
  }
  if (state.fp) patch.frameworkPreset = state.fp as FrameworkPreset
  patch.interconnectOverride = state.io ?? null

  return { patch, missing }
}
```

In the same edit add the imports it needs: `import { DEFAULT_KV_TIER, KV_TIER_TYPES, type KVTierSettings } from '@engines/kv-tier'` (replacing the existing kv-tier import) and `import type { UIConfig } from '@store/uiStore'` (type-only: no runtime cycle). `FrameworkPreset`, `QuantizationFormat`, `KVCachePrecision`, `OffloadMode`, `OffloadTarget`, `ShardingStrategy`, `GPU`, `Model` are already imported.

- [ ] **Step 4: Rebuild the restore effect in `useURLSync`**

Replace lines 1-157 of `src/hooks/useURLSync.ts` (imports through the end of the first `useEffect`) with:

```ts
import { findGPUById, findModelById, useUIStore } from '@store/uiStore'
import { deserializeFromURL, serializeToURL, urlStateToConfig } from '@store/urlSerializer'
import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * Hook that provides bidirectional sync between Zustand store and URL hash
 *
 * On mount:
 * - Reads URL hash and deserializes it
 * - Builds the whole configuration (urlStateToConfig) and applies it in ONE store
 *   action, normalized once; corrections surface as "Shared link adjusted"
 * - Warns when a referenced model/GPU is not in the database
 *
 * On store changes:
 * - Debounces changes by 300ms
 * - Serializes current state to compressed URL hash
 * - Updates URL without triggering navigation
 * - Warns if URL exceeds recommended length
 */
export function useURLSync() {
  // Hydrate store from URL hash on mount
  useEffect(() => {
    const hash = window.location.hash.slice(1)
    if (!hash) {
      return
    }

    const urlState = deserializeFromURL(hash)
    if (!urlState) {
      // Invalid/corrupted URL - show toast and continue with defaults
      toast.error('Could not restore configuration from URL')
      return
    }

    const { patch, missing } = urlStateToConfig(urlState, {
      findModel: findModelById,
      findGPU: findGPUById,
    })
    for (const message of missing) toast.warning(message)
    useUIStore.getState().restoreConfig(patch)
  }, []) // Empty deps - only run on mount
```

Leave the second `useEffect` (serialize on change) and the closing brace unchanged. The old `const store = useUIStore()` subscription (which re-rendered App on every store change) is gone.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `npx vitest run src/store/urlSerializer.test.ts src/store/uiStore.test.ts`
Expected: PASS. The pre-existing `'should return null for invalid schema'` test still passes (a string `sl` is still rejected).

- [ ] **Step 6: Full suite, typecheck, lint, commit**

Run: `npx vitest run && npm run typecheck && npm run lint:fix && npm run lint`
Expected: all green.

```bash
rtk git add src/store/urlSerializer.ts src/store/urlSerializer.test.ts src/store/uiStore.test.ts src/hooks/useURLSync.ts
rtk git commit -m "feat(url): restore the whole link in one normalized action; serialize interconnectOverride

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 3a: GPU topology data (schema, `nvlink-3`, per-card corrections, H200 NVL)

**Files:**
- Modify: `src/utils/schemas.ts:79-104` (interconnect enums, `nvlink_bridge`)
- Modify: `src/engines/types.ts:127-133` (`InterconnectType` gains `nvlink-3`)
- Modify: `src/engines/constants.ts:232` (`INTERCONNECT_SPECS['nvlink-3']`), `:458-466` (label)
- Modify: `src/engines/multi-gpu.ts:408` (map `nvlink-3`)
- Modify: `scripts/fetch-gpus.ts` (rows listed in Step 4)
- Regenerate: `src/data/gpus.json`
- Modify: `src/engines/frameworks.ts:148-154`, `src/components/inputs/FrameworkPresetPicker.tsx:41`
- Modify: `src/types/gpu.ts:7-31`, `src/store/urlSerializer.ts:40-47` and `:192-200` (custom GPU fields)
- Test: `src/utils/gpus.test.ts`, `src/engines/config-rules.test.ts`, `src/store/uiStore.test.ts`, `src/store/urlSerializer.test.ts`

**Interfaces:**
- Consumes: `GPUSchema.unified_memory` (Task 1a).
- Produces:
  - `GPUSchema.interconnect` / `interconnect_options` enums include `'nvlink-3'`.
  - `GPUSchema.nvlink_bridge?: { type: 'nvlink-3' | 'nvlink-4' | 'nvlink-5'; size: number }` (size is an integer >= 2).
  - `InterconnectType` includes `'nvlink-3'`; `INTERCONNECT_SPECS['nvlink-3'] = { bandwidthGBps: 600, recommendedMaxTPDegree: 8, tpScalingEfficiency: 0.89, allreduceLatencyUs: 11 }`; `INTERCONNECT_LABELS['nvlink-3'] = 'NVLink 3 — 600 GB/s'`.
  - New row `nvidia-h200-nvl-141gb`; 28 GPU rows in total.
  - `CustomGPUInput.unified_memory?: boolean`; URL `customGPU` carries optional `unified_memory` and `nvlink_bridge`.

- [ ] **Step 1: Write the failing data tests**

In `src/utils/gpus.test.ts`, replace the `validInterconnects` array in `'should have consistent interconnect types'` (lines 70-78) with:

```ts
    const validInterconnects = [
      'none',
      'nvlink',
      'nvlink-3',
      'nvlink-4',
      'nvlink-5',
      'pcie-4',
      'pcie-5',
      'infinity-fabric',
      'unified',
      undefined,
    ]
```

and append:

```ts
describe('GPU topology (v2 per-card audit, spec Section 3)', () => {
  const gpus = validateGPUs(gpusData)
  const byId = (id: string) => {
    const gpu = gpus.find((g) => g.id === id)
    if (!gpu) throw new Error(`GPU not found: ${id}`)
    return gpu
  }

  it('lists 28 GPUs (H200 NVL split from H200 SXM)', () => {
    expect(gpus).toHaveLength(28)
  })

  it('H100 PCIe: PCIe 5 with a 2-card NVLink bridge at 600 GB/s, dense FP16 756', () => {
    const g = byId('nvidia-h100-80gb-pcie')
    expect(g.interconnect).toBe('pcie-5')
    expect(g.nvlink_bridge).toEqual({ type: 'nvlink-3', size: 2 })
    expect(g.fp16_tflops).toBe(756)
  })

  it('A100 PCIe: PCIe 4 with a 2-card NVLink bridge; A100 SXM: NVLink 3', () => {
    expect(byId('nvidia-a100-80gb-pcie').interconnect).toBe('pcie-4')
    expect(byId('nvidia-a100-80gb-pcie').nvlink_bridge).toEqual({ type: 'nvlink-3', size: 2 })
    expect(byId('nvidia-a100-80gb-sxm').interconnect).toBe('nvlink-3')
  })

  it('H200 id is the SXM product with no interconnect variants', () => {
    const g = byId('nvidia-h200-141gb')
    expect(g.interconnect).toBe('nvlink-4')
    expect(g.interconnect_options).toBeUndefined()
    expect(g.nvlink_bridge).toBeUndefined()
  })

  it('H200 NVL: PCIe 5 card with a 4-way NVLink 4 bridge, dense FP16 835', () => {
    const g = byId('nvidia-h200-nvl-141gb')
    expect(g).toMatchObject({
      vram_gb: 141,
      memory_bandwidth_gbps: 4800,
      fp16_tflops: 835,
      fp32_tflops: 60,
      interconnect: 'pcie-5',
      nvlink_bridge: { type: 'nvlink-4', size: 4 },
      max_gpus_per_node: 8,
      tier: 'datacenter',
    })
  })

  it('GB10 (DGX Spark) is one GPU per node, no scale-up link, unified memory', () => {
    const g = byId('nvidia-gb10')
    expect(g.max_gpus_per_node).toBe(1)
    expect(g.interconnect).toBe('none')
    expect(g.interconnect_options).toBeUndefined()
    expect(g.unified_memory).toBe(true)
  })

  it('GB300 Desktop Superchip has no GPU-to-GPU link (single GPU)', () => {
    expect(byId('nvidia-gb300-desktop-252gb').interconnect).toBe('none')
  })

  it('L40S and RTX 6000 Ada are explicit PCIe 4', () => {
    expect(byId('nvidia-l40s').interconnect).toBe('pcie-4')
    expect(byId('nvidia-rtx-6000-ada').interconnect).toBe('pcie-4')
  })

  it('marks every Apple Silicon row and GB10 as unified memory, and nothing else', () => {
    const unified = gpus.filter((g) => g.unified_memory === true).map((g) => g.id)
    const expected = gpus.filter((g) => g.manufacturer === 'apple').map((g) => g.id)
    expect(unified.sort()).toEqual([...expected, 'nvidia-gb10'].sort())
  })

  it('no database GPU offers interconnect variants any more', () => {
    expect(gpus.filter((g) => g.interconnect_options !== undefined)).toEqual([])
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/utils/gpus.test.ts`
Expected: FAIL (27 rows; H100 PCIe interconnect `nvlink-4`; no `nvidia-h200-nvl-141gb`).

- [ ] **Step 3: Schema, engine type, spec, label, mapping**

In `src/utils/schemas.ts`, add above `export const GPUSchema`:

```ts
/** GPU.interconnect values (scale-up link as the data states it) */
const GPU_INTERCONNECTS = [
  'none',
  'nvlink',
  'nvlink-3',
  'nvlink-4',
  'nvlink-5',
  'pcie-4',
  'pcie-5',
  'infinity-fabric',
  'unified',
] as const
```

and replace lines 79-104 (`interconnect` and `interconnect_options`) with:

```ts
  interconnect: z.enum(GPU_INTERCONNECTS).optional(),
  interconnect_options: z.array(z.enum(GPU_INTERCONNECTS)).optional(),
  /**
   * NVLink bridge between PCIe cards (H100/A100 PCIe pair 2 cards; H200 NVL up to 4).
   * The bridge carries a tensor-parallel group only while it fits: a larger group
   * crosses the card's PCIe link (`interconnect`). See resolveInterconnect.
   */
  nvlink_bridge: z
    .object({
      type: z.enum(['nvlink-3', 'nvlink-4', 'nvlink-5']),
      size: z.number().int().min(2),
    })
    .optional(),
```

In `src/engines/types.ts`, add `| 'nvlink-3'` to `InterconnectType` (before `'nvlink-4'`) and add `- nvlink-3: 3rd gen NVLink (600 GB/s): A100 SXM, and the H100/A100 PCIe bridges` to its doc list.

In `src/engines/constants.ts`, add as the first entry of `INTERCONNECT_SPECS`:

```ts
  'nvlink-3': {
    type: 'nvlink-3',
    bandwidthGBps: 600,
    recommendedMaxTPDegree: 8,
    // Same +0.05-per-doubling slope as the other rows: 0.92 - 0.05 * log2(900/600) = 0.891.
    // Derived, not measured.
    tpScalingEfficiency: 0.89,
    allreduceLatencyUs: NVLINK_ALLREDUCE_LATENCY_US,
  },
```

and add `'nvlink-3': 'NVLink 3 — 600 GB/s',` as the first entry of `INTERCONNECT_LABELS`.

In `src/engines/multi-gpu.ts` `resolveInterconnect`, before `if (interconnect === 'nvlink-4') return 'nvlink-4'` add:

```ts
  if (interconnect === 'nvlink-3') return 'nvlink-3'
```

- [ ] **Step 4: Edit the GPU rows in `scripts/fetch-gpus.ts`**

Apply each change (ids never change):

`nvidia-h100-80gb-pcie` (lines 22-37): set `fp16_tflops: 756, // Dense: the datasheet's 1,513 TF is with sparsity`, `interconnect: 'pcie-5',` and add `nvlink_bridge: { type: 'nvlink-3', size: 2 }, // 2-way bridge, 600 GB/s (NVIDIA H100 datasheet; Lenovo LP1732)` after it.

`nvidia-h200-141gb` (lines 54-70): set `name: 'NVIDIA H200 141GB SXM'`, delete the `interconnect_options` line, and put this comment above the row:

```ts
  // H200 SXM: HGX 4- or 8-GPU NVSwitch baseboards only (NVIDIA HGX AI Factory
  // reference architecture). The PCIe card is a separate row below (H200 NVL);
  // this id stays SXM so old links resolve to the product they most likely meant.
```

Insert right after the H200 SXM row:

```ts
  // H200 NVL: the PCIe card. 2- or 4-way NVLink bridges at 900 GB/s (HPE
  // PSN1014857028PLEN / PSN1014856854VNEN), up to 8 cards per server. PNY H200 NVL
  // datasheet: 141 GB HBM3e, 4.8 TB/s, 60 TF FP32, 1,671 TF FP16 with sparsity
  // = 835 dense, up to 600 W.
  {
    id: 'nvidia-h200-nvl-141gb',
    name: 'NVIDIA H200 NVL 141GB (PCIe)',
    manufacturer: 'nvidia',
    vram_gb: 141,
    memory_bandwidth_gbps: 4800,
    memory_type: 'HBM3e',
    bus_width: 5120,
    fp16_tflops: 835,
    fp32_tflops: 60,
    tdp_watts: 600,
    interconnect: 'pcie-5',
    nvlink_bridge: { type: 'nvlink-4', size: 4 },
    max_gpus_per_node: 8,
    tier: 'datacenter',
    spec_url:
      'https://www.pny.com/file%20library/company/support/linecards/data-center-gpus/h200-nvl-datasheet.pdf',
  },
```

Before committing, open the PNY datasheet URL above and confirm 141 GB, 4.8 TB/s, 60 TF FP32, 1,671 TF FP16 sparse, 600 W. If a figure differs, use the datasheet's and update the Step 1 test to match.

`nvidia-a100-80gb-pcie` (lines 141-156): `interconnect: 'pcie-4',` plus `nvlink_bridge: { type: 'nvlink-3', size: 2 }, // 2-way bridge, 600 GB/s (NVIDIA A100 page)`.

`nvidia-a100-80gb-sxm` (lines 157-172): `interconnect: 'nvlink-3', // NVSwitch at 600 GB/s (was priced at NVLink 4's 900)`.

`nvidia-l40s` (line 184) and `nvidia-rtx-6000-ada` (line 218): `interconnect: 'pcie-4',`.

`nvidia-gb300-desktop-252gb` (line 282): `interconnect: 'none', // one GPU; NVLink-C2C links it to the CPU, not to another GPU`.

`nvidia-gb10` (lines 287-303): `interconnect: 'none',`, delete `interconnect_options`, `max_gpus_per_node: 1,`, add `unified_memory: true,`, and put above the row:

```ts
  // DGX Spark: one GB10 per unit, no GPU-to-GPU link. Two or four Sparks cluster
  // over ConnectX-7 200 GbE as separate servers (NVIDIA Sync cluster assistant), so
  // model them as numNodes with the 200GbE fabric preset, not as 2 GPUs per node.
```

Every Apple row (`apple-m1-ultra` ... `apple-m1-max`): add `unified_memory: true,` after `interconnect: 'unified',`.

- [ ] **Step 5: Regenerate `gpus.json`**

Run: `npm run refresh:gpus && npx biome format --write src/data/gpus.json && rtk git diff --stat src/data/gpus.json`
Expected: `✓ All GPUs valid`, `✓ Wrote 28 GPUs`; the diff touches only the rows edited in Step 4 (no whole-file reformat). If the diff reformats untouched rows, compare with `git show HEAD:src/data/gpus.json` and rerun `npx biome format --write src/data/gpus.json`.

- [ ] **Step 6: TGI label, custom GPU passthrough, URL custom GPU fields**

`src/engines/frameworks.ts` (tgi entry): `name: 'TGI (archived)'`, `description: 'HuggingFace Text Generation Inference (archived upstream; kept so existing links still open)'`.

`src/components/inputs/FrameworkPresetPicker.tsx:41`: `<option value="tgi">TGI (archived, inference only)</option>`.

`src/types/gpu.ts`: add `unified_memory?: boolean` to `CustomGPUInput` and, in `createCustomGPU`, after `max_gpus_per_node: 8,` add `...(input.unified_memory ? { unified_memory: true } : {}),`.

`src/store/urlSerializer.ts` `customGPU` object (lines 40-47): add

```ts
      unified_memory: z.boolean().optional(),
      nvlink_bridge: z
        .object({ type: z.enum(['nvlink-3', 'nvlink-4', 'nvlink-5']), size: z.number().int().min(2) })
        .optional(),
```

and in `serializeToURL`'s `customGPU` literal (lines 194-199) add `unified_memory: state.selectedGPU.unified_memory,` and `nvlink_bridge: state.selectedGPU.nvlink_bridge,` (JSON drops them when undefined). `urlStateToConfig` already spreads `...state.customGPU`.

Add to `src/store/urlSerializer.test.ts` (inside `describe('urlStateToConfig', ...)`):

```ts
  it('round-trips a custom GPU unified-memory flag', () => {
    const custom = { ...realGPU('apple-m3-ultra'), id: 'custom-1' }
    const decoded = deserializeFromURL(serializeToURL({ ...everyKey, mode: 'inference' as const, selectedGPU: custom }))
    if (!decoded) throw new Error('expected the link to parse')
    expect(urlStateToConfig(decoded, lookups).patch.selectedGPU?.unified_memory).toBe(true)
  })
```

- [ ] **Step 7: Replace the Task 1a/2a derived fixtures with the real rows**

In `src/engines/config-rules.test.ts` replace the `UNIFIED_M3` definition and its comment line with `const UNIFIED_M3 = M3 // unified_memory comes from gpus.json since Task 3a`, and add to the `CASES` array:

```ts
  { rule: 'R6', path: 'dependency', name: 'PCIe tier then DGX Spark', config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' } }, model: L8, gpu: findGPU('nvidia-gb10'), validUnder: { model: L8, gpu: H100 }, expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } } },
```

In `src/store/uiStore.test.ts` replace the `UNIFIED_M3` definition and its comment with `const UNIFIED_M3 = findGPU('apple-m3-ultra')`, and remove `validateGPU` from the `@utils/schemas` import in the same edit.

- [ ] **Step 8: Run the tests**

Run: `npx vitest run`
Expected: PASS. `WITH_OPTIONS` in config-rules.test.ts stays derived (no database GPU carries options now).

- [ ] **Step 9: Typecheck, lint, commit**

Run: `npm run typecheck && npm run lint:fix && npm run lint`

```bash
rtk git add src/utils/schemas.ts src/engines/types.ts src/engines/constants.ts src/engines/multi-gpu.ts scripts/fetch-gpus.ts src/data/gpus.json src/engines/frameworks.ts src/components/inputs/FrameworkPresetPicker.tsx src/types/gpu.ts src/store/urlSerializer.ts src/store/urlSerializer.test.ts src/utils/gpus.test.ts src/engines/config-rules.test.ts src/store/uiStore.test.ts
rtk git commit -m "fix(data): per-card GPU topology (bridges, NVLink 3, unified memory), H200 NVL split

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---

### Task 3b: `resolveInterconnect(gpu, groupSize)` and every consumer

**Files:**
- Modify: `src/engines/multi-gpu.ts:370` (TP/EP call), `:388-430` (signature + bridge), `:461` (validate), add `interconnectLabel`
- Modify: `src/engines/performance.ts:260`
- Modify: `src/components/inputs/ShardingStrategySelector.tsx:28-56`, `:149-168`
- Modify: `src/hooks/useInferenceCalculation.ts:256-259`
- Test: `src/engines/multi-gpu.test.ts` (existing calls at 608-668, new tests), `src/engines/performance.test.ts` (calls at 1036, 1212, 1316, 1441; new anchor), `src/components/inputs/ShardingStrategySelector.test.tsx`

**Interfaces:**
- Consumes: `GPU.nvlink_bridge`, `INTERCONNECT_SPECS['nvlink-3']` (Task 3a).
- Produces:
  - `resolveInterconnect(gpu: GPU, groupSize: number): InterconnectType` — the bridge type when `groupSize <= gpu.nvlink_bridge.size`, else the mapping of `gpu.interconnect`.
  - `interconnectLabel(gpu: GPU, groupSize: number): string` — `"NVLink bridge — {bw} GB/s"` while the bridge applies, else `INTERCONNECT_LABELS[type] ?? type`.
  - Group sizes used by consumers: TP all-reduce and EP all-to-all use the GPUs per stage (`layout.gpusPerStage` in performance, `numGPUs` in multi-gpu); `validateInterconnect` and the strategy badge use `numGPUs`. `MultiGPUBreakdownChart` and the PPTX read `breakdown.interconnectBandwidthGBps`, which the engine now resolves with the group size; the worker and the sync hook reach `resolveInterconnect` only through `calculateMultiNodeVRAM`/`validateInterconnect` (no direct call to change).

- [ ] **Step 1: Write the failing tests**

In `src/engines/multi-gpu.test.ts`, change every existing call `resolveInterconnect(X)` (lines 610-667 and 777) to `resolveInterconnect(X, 8)` (none of those fixtures has a bridge, so the group size does not matter). Then append:

```ts
describe('resolveInterconnect with NVLink bridges (spec Section 3)', () => {
  const realGPUs = validateGPUs(gpusData)
  const byId = (id: string) => {
    const gpu = realGPUs.find((g) => g.id === id)
    if (!gpu) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
    return gpu
  }
  const h100pcie = byId('nvidia-h100-80gb-pcie')
  const h200nvl = byId('nvidia-h200-nvl-141gb')
  const llama70 = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
  const dsr1 = validateModels(modelsData).find((m) => m.id === 'deepseek-r1')
  if (!llama70 || !dsr1) throw new Error('fixture models not found')

  it('uses the bridge while the group fits it, PCIe beyond', () => {
    expect(resolveInterconnect(h100pcie, 2)).toBe('nvlink-3')
    expect(resolveInterconnect(h100pcie, 4)).toBe('pcie-5')
    expect(resolveInterconnect(h200nvl, 4)).toBe('nvlink-4')
    expect(resolveInterconnect(h200nvl, 8)).toBe('pcie-5')
  })

  it('labels the bridge by bandwidth', () => {
    expect(interconnectLabel(h100pcie, 2)).toBe('NVLink bridge — 600 GB/s')
    expect(interconnectLabel(h200nvl, 4)).toBe('NVLink bridge — 900 GB/s')
    expect(interconnectLabel(h100pcie, 4)).toBe('PCIe 5 — 128 GB/s')
  })

  it('prices TP-2 over the bridge and TP-4 over PCIe (chart and PPTX read this figure)', () => {
    const single = calculateInferenceVRAM({ model: llama70, quantization: 'fp8', sequenceLength: 8192, batchSize: 1 })
    const tp2 = calculateMultiGPUVRAM(single, llama70, h100pcie.vram_gb, 2, 'tensor-parallel', h100pcie)
    const tp4 = calculateMultiGPUVRAM(single, llama70, h100pcie.vram_gb, 4, 'tensor-parallel', h100pcie)
    expect(tp2.interconnectBandwidthGBps).toBe(600)
    expect(tp4.interconnectBandwidthGBps).toBe(128)
  })

  it('falls back to PCIe for EP-4 on a 2-card bridge', () => {
    const single = calculateInferenceVRAM({ model: dsr1, quantization: 'fp8', sequenceLength: 8192, batchSize: 1 })
    const ep4 = calculateMultiGPUVRAM(single, dsr1, h100pcie.vram_gb, 4, 'expert-parallel', h100pcie, 'fp8')
    expect(ep4.interconnectBandwidthGBps).toBe(128)
  })

  it('warns (W2) at TP-8 on PCIe 5 and TP-4 on PCIe 4, not at TP-2 over the bridge', () => {
    expect(validateInterconnect(h100pcie, 2, 'tensor-parallel').warning).toBeNull()
    expect(validateInterconnect(h100pcie, 8, 'tensor-parallel').warning).not.toBeNull()
    expect(validateInterconnect(byId('nvidia-a100-80gb-pcie'), 4, 'tensor-parallel').warning).not.toBeNull()
  })
})
```

Add `interconnectLabel` to the `./multi-gpu` import in the same edit.

In `src/engines/performance.test.ts`, change `resolveInterconnect(gpu)` at lines 1036, 1212 and 1316 to `resolveInterconnect(gpu, gpusPerStage)`, and `resolveInterconnect(b300)` at line 1441 to `resolveInterconnect(b300, numGPUs)`. Then append:

```ts
describe('bridge-aware decode (spec Section 3 impact anchor)', () => {
  const gpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-pcie')
  const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
  if (!gpu || !model) throw new Error('fixture not found')
  const sequenceLength = 8192
  const batchSize = 1

  const tps = (numGPUs: number) => {
    const single = calculateInferenceVRAM({ model, quantization: 'fp8', sequenceLength, batchSize })
    const multi = calculateMultiGPUVRAM(single, model, gpu.vram_gb, numGPUs, 'tensor-parallel', gpu, 'fp8')
    return estimatePerformance({ model, gpu, quantization: 'fp8', sequenceLength, batchSize, multiGPUResult: multi })
      .tokensPerSecond.toNumber()
  }
  // Hand-priced from the corrected data: weights and KV split across the TP group,
  // read at 2000 GB/s, plus two all-reduces per layer at the resolved link's latency.
  const expected = (numGPUs: number, latencyUs: number) => {
    const weights = calculateModelWeightVRAM(model.num_parameters_billion, 'fp8', model).mul(BYTES_PER_GB)
    const kv = calculateKVCacheVRAM({ model, sequenceLength, batchSize, kvPrecision: 'fp16' }).mul(BYTES_PER_GB)
    const kvShards = Math.min(numGPUs, model.num_kv_heads ?? model.num_attention_heads)
    const memorySeconds = weights.div(numGPUs).add(kv.div(kvShards)).div(new Decimal(gpu.memory_bandwidth_gbps).mul(1e9))
    const stageSeconds = memorySeconds.add((2 * latencyUs * model.num_hidden_layers) / 1e6)
    return new Decimal(1).div(stageSeconds).toNumber()
  }

  it('TP-8 crosses PCIe 5; TP-2 stays on the NVLink 3 bridge', () => {
    expect(tps(8) / expected(8, INTERCONNECT_SPECS['pcie-5'].allreduceLatencyUs)).toBeCloseTo(1, 9)
    expect(tps(2) / expected(2, INTERCONNECT_SPECS['nvlink-3'].allreduceLatencyUs)).toBeCloseTo(1, 9)
  })

  it('TP-8 decode drops about a quarter versus the old NVLink pricing (spec: -25.7%)', () => {
    const change = tps(8) / expected(8, INTERCONNECT_SPECS['nvlink-4'].allreduceLatencyUs) - 1
    expect(change).toBeGreaterThan(-0.27)
    expect(change).toBeLessThan(-0.24)
  })
})
```

Replace `src/components/inputs/ShardingStrategySelector.test.tsx` with:

```tsx
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create(() => ({
    numGPUs: 8,
    numNodes: 1,
    mode: 'inference',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'none',
    setShardingStrategy: () => {},
    selectedGPU: null as unknown,
    selectedModel: null as unknown,
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ShardingStrategySelector } from './ShardingStrategySelector'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => models.find((m) => m.id === id) ?? null
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null

describe('ShardingStrategySelector', () => {
  beforeEach(() => useUIStore.setState({ selectedModel: null, selectedGPU: null, numGPUs: 8 }))

  it('offers expert parallelism for a MoE model', () => {
    useUIStore.setState({ selectedModel: model('deepseek-r1') })
    render(<ShardingStrategySelector />)
    expect(screen.getByText(/Expert Parallel \+ DP attention/)).toBeInTheDocument()
  })

  it('hides expert parallelism for a dense model', () => {
    useUIStore.setState({ selectedModel: model('meta-llama-llama-3.1-70b') })
    render(<ShardingStrategySelector />)
    expect(screen.queryByText(/Expert Parallel/)).not.toBeInTheDocument()
  })

  it('shows the NVLink bridge at TP-2 and PCIe 5 at TP-4 on H100 PCIe (badge never contradicts the maths)', () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-h100-80gb-pcie'), numGPUs: 2 })
    const { unmount } = render(<ShardingStrategySelector />)
    expect(screen.getByText(/NVLink bridge: 600 GB\/s/)).toBeInTheDocument()
    unmount()
    useUIStore.setState({ numGPUs: 4 })
    render(<ShardingStrategySelector />)
    expect(screen.getByText(/PCIe 5: 128 GB\/s/)).toBeInTheDocument()
    expect(screen.queryByText(/NVLink bridge/)).not.toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/multi-gpu.test.ts src/engines/performance.test.ts src/components/inputs/ShardingStrategySelector.test.tsx`
Expected: FAIL: `interconnectLabel` is not exported; H100 PCIe TP-2 resolves to `pcie-5`; the badge shows PCIe at TP-2.

- [ ] **Step 3: Implement the engine change**

In `src/engines/multi-gpu.ts`, replace the doc comment and signature of `resolveInterconnect` (lines 388-406) with:

```ts
/**
 * Resolve the link a group of `groupSize` GPUs actually talks over.
 *
 * An NVLink bridge (H100/A100 PCIe pairs, H200 NVL up to 4) carries the group only
 * while it fits the bridge; a larger group crosses the card's own link, which the
 * rest of this function maps from GPU.interconnect.
 *
 * @param gpu - GPU configuration
 * @param groupSize - GPUs in the tensor/expert-parallel group (1 = no traffic)
 */
export function resolveInterconnect(gpu: GPU, groupSize: number): InterconnectType {
  if (gpu.nvlink_bridge && groupSize <= gpu.nvlink_bridge.size) return gpu.nvlink_bridge.type
  const interconnect = gpu.interconnect
```

(the rest of the body is unchanged). Add after the function:

```ts
/**
 * Display name of the resolved link, labelling a bridge by its bandwidth
 * ("NVLink bridge — 600 GB/s") so the badge never shows NVLink while the maths uses PCIe.
 */
export function interconnectLabel(gpu: GPU, groupSize: number): string {
  const type = resolveInterconnect(gpu, groupSize)
  if (gpu.nvlink_bridge && groupSize <= gpu.nvlink_bridge.size) {
    return `NVLink bridge — ${INTERCONNECT_SPECS[type].bandwidthGBps} GB/s`
  }
  return INTERCONNECT_LABELS[type] ?? type
}
```

Change line 370 to `const interconnectType = resolveInterconnect(gpu, numGPUs)` and line 461 to `const interconnectType = resolveInterconnect(gpu, numGPUs)`.

In `src/engines/performance.ts` line 260: `const link = INTERCONNECT_SPECS[resolveInterconnect(gpu, layout.gpusPerStage)]`.

- [ ] **Step 4: Update the badge**

In `src/components/inputs/ShardingStrategySelector.tsx`, change the import on line 3 to `import { interconnectLabel as linkLabel, resolveInterconnect } from '@engines/multi-gpu'`, replace lines 28-30 with:

```tsx
  // Resolve the link for THIS group size: an NVLink bridge only carries a group that fits it
  const interconnectType = selectedGPU ? resolveInterconnect(selectedGPU, numGPUs) : 'none'
  const interconnectSpec = INTERCONNECT_SPECS[interconnectType]
  const bridgeSize =
    selectedGPU?.nvlink_bridge && numGPUs <= selectedGPU.nvlink_bridge.size
      ? selectedGPU.nvlink_bridge.size
      : null
```

replace lines 47-52 (the `interconnectLabel` const and its comment) with:

```tsx
  // The badge renders bandwidthGBps separately, so only the name part of the label is used
  const interconnectLabel = (
    selectedGPU ? linkLabel(selectedGPU, numGPUs) : (INTERCONNECT_LABELS.none ?? 'None')
  ).split(' — ')[0]
```

and in the NVLink/Infinity Fabric badge branch (lines 157-159) replace `TP up to {interconnectSpec.recommendedMaxTPDegree} GPUs` with `TP up to {bridgeSize ?? interconnectSpec.recommendedMaxTPDegree} GPUs`.

- [ ] **Step 5: The interconnect override drops the bridge**

In `src/hooks/useInferenceCalculation.ts` replace lines 256-259 with:

```ts
    // Apply the interconnect variant the user picked; it replaces the bridge too,
    // because the user chose the link explicitly.
    const effectiveGPU = interconnectOverride
      ? {
          ...gpu,
          interconnect: interconnectOverride as GPU['interconnect'],
          nvlink_bridge: undefined,
        }
      : gpu
```

- [ ] **Step 6: Run the tests**

Run: `npx vitest run`
Expected: PASS.

- [ ] **Step 7: Typecheck, lint, commit**

Run: `npm run typecheck && npm run lint:fix && npm run lint`

```bash
rtk git add src/engines/multi-gpu.ts src/engines/performance.ts src/components/inputs/ShardingStrategySelector.tsx src/hooks/useInferenceCalculation.ts src/engines/multi-gpu.test.ts src/engines/performance.test.ts src/components/inputs/ShardingStrategySelector.test.tsx
rtk git commit -m "feat(engine): resolveInterconnect by group size; bridges carry only the groups they fit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 4a: Fabric primitives (hop time, eta, vLLM chunk, microbatch fill, 200GbE preset)

**Files:**
- Modify: `src/engines/types.ts:158-165` (`FabricType` gains `ethernet-200g`)
- Modify: `src/engines/fabric.ts` (add constants, functions, preset; the old heuristic stays until Task 4b)
- Modify: `src/store/urlSerializer.ts:68-78` (`fab` enum)
- Test: `src/engines/fabric.test.ts`, `src/store/urlSerializer.test.ts`

**Interfaces:**
- Consumes: `GPU` (with Task 3a data: GB10 is 128 GB, one per node).
- Produces (exported from `src/engines/fabric.ts`):
  - `HGX_EFFECTIVE_FRACTION = 0.8` (ASSUMPTION), `GB10_EFFECTIVE_FRACTION = 0.37` (SECONDARY SOURCE), `FABRIC_HOP_LATENCY_S = 10e-6`
  - `effectiveFraction(gpu: GPU): number` — 0.37 for `nvidia-gb10`, else 0.8
  - `interNodeGBps(portGBps: number, gpusPerNode: number, fraction: number): number` — `port x gpusPerNode x fraction`
  - `fabricHopSeconds(tokens: number, hiddenSize: number, gbps: number): number` — `tokens x hidden x 2 x 2 / (gbps x 1e9) + FABRIC_HOP_LATENCY_S`
  - `maxNumBatchedTokens(gpu: GPU): number` — 16384 (>= 160 GB), 8192 (>= 70 GB, not A100), else 2048
  - `prefillMicrobatches(batchSize: number, sequenceLength: number, chunkTokens: number): number` — `max(1, ceil(B x T / C))`
  - `prefillPipelineFill(microbatches: number, stages: number): number` — `M / (M + N - 1)`, 1 at one stage
  - `FABRIC_SPECS['ethernet-200g'] = { type: 'ethernet-200g', label: '200GbE (ConnectX-7)', portGBps: 25, classFactor: 1.0 }` (`classFactor` is dropped from every preset in Task 4b)

- [ ] **Step 1: Write the failing tests**

Append to `src/engines/fabric.test.ts`, extending its imports in the same edit to:

```ts
import gpusData from '@data/gpus.json'
import { validateGPUs } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import {
  effectiveFraction,
  FABRIC_HOP_LATENCY_S,
  FABRIC_SPECS,
  fabricDecodeEfficiency,
  fabricHopSeconds,
  fabricPrefillEfficiency,
  GB10_EFFECTIVE_FRACTION,
  HGX_EFFECTIVE_FRACTION,
  interNodeGBps,
  maxNumBatchedTokens,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
  prefillMicrobatches,
  prefillPipelineFill,
  resolveFabricSpec,
} from './fabric'
```

```ts
const realGPUs = validateGPUs(gpusData)
function findGPU(id: string) {
  const gpu = realGPUs.find((g) => g.id === id)
  if (!gpu) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return gpu
}

describe('fabricHopSeconds (spec Section 3b)', () => {
  it('Llama 70B at 8k over one GB10 on 200 GbE: 268,435,456 B at 25 x 0.37 GB/s + 10 us = 29.03 ms', () => {
    const gbps = interNodeGBps(FABRIC_SPECS['ethernet-200g'].portGBps, 1, effectiveFraction(findGPU('nvidia-gb10')))
    expect(gbps).toBeCloseTo(9.25, 10)
    expect(8192 * 8192 * 2 * 2).toBe(268_435_456)
    expect(fabricHopSeconds(8192, 8192, gbps)).toBeCloseTo(0.02903, 6)
  })

  it('is the latency floor alone for zero tokens', () => {
    expect(fabricHopSeconds(0, 8192, 100)).toBe(FABRIC_HOP_LATENCY_S)
  })
})

describe('effectiveFraction', () => {
  it('uses the GB10 measurement for DGX Spark and the HGX assumption elsewhere', () => {
    expect(effectiveFraction(findGPU('nvidia-gb10'))).toBe(GB10_EFFECTIVE_FRACTION)
    expect(effectiveFraction(findGPU('nvidia-h100-80gb-sxm'))).toBe(HGX_EFFECTIVE_FRACTION)
  })
})

describe('interNodeGBps', () => {
  it('is port x GPUs per node x eta', () => {
    expect(interNodeGBps(100, 8, 0.8)).toBeCloseTo(640, 10)
  })
})

describe('maxNumBatchedTokens (vLLM engine/arg_utils.py defaults)', () => {
  it('scales with GPU memory and excludes A100', () => {
    expect(maxNumBatchedTokens(findGPU('nvidia-b200-192gb'))).toBe(16384) // 180 GB
    expect(maxNumBatchedTokens(findGPU('nvidia-h100-80gb-sxm'))).toBe(8192)
    expect(maxNumBatchedTokens(findGPU('nvidia-gb10'))).toBe(8192) // 128 GB
    expect(maxNumBatchedTokens(findGPU('nvidia-a100-80gb-sxm'))).toBe(2048)
    expect(maxNumBatchedTokens(findGPU('nvidia-rtx-4090'))).toBe(2048)
  })
})

describe('prefillMicrobatches and prefillPipelineFill', () => {
  it('M = ceil(B x T / C), at least 1', () => {
    expect(prefillMicrobatches(1, 8192, 8192)).toBe(1)
    expect(prefillMicrobatches(32, 8192, 8192)).toBe(32)
    expect(prefillMicrobatches(1, 32768, 8192)).toBe(4)
    expect(prefillMicrobatches(3, 1000, 8192)).toBe(1)
  })

  it('fill is M / (M + N - 1): one microbatch walks the stages serially', () => {
    expect(prefillPipelineFill(1, 1)).toBe(1)
    expect(prefillPipelineFill(1, 2)).toBe(0.5)
    expect(prefillPipelineFill(32, 2)).toBeCloseTo(32 / 33, 12)
  })
})

describe('ethernet-200g preset', () => {
  it('is 200 Gb/s = 25 GB/s per port (ConnectX-7)', () => {
    expect(FABRIC_SPECS['ethernet-200g'].portGBps).toBe(25)
  })
})
```

Append to `src/store/urlSerializer.test.ts`:

```ts
describe('200GbE fabric links', () => {
  it('parses fab: ethernet-200g', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(JSON.stringify({ q: 'fp16', sl: 4096, bs: 1, kvq: 'fp16', ng: 1, ss: 'tensor-parallel', nn: 2, fab: 'ethernet-200g' })),
    )
    expect(decoded?.fab).toBe('ethernet-200g')
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/fabric.test.ts src/store/urlSerializer.test.ts`
Expected: FAIL: `effectiveFraction` not exported; `FABRIC_SPECS['ethernet-200g']` undefined; the 200g link returns `null`.

- [ ] **Step 3: Implement**

`src/engines/types.ts`: add `| 'ethernet-200g'` after `'ethernet-400g'` in `FabricType`.

`src/store/urlSerializer.ts` `fab` enum: add `'ethernet-200g',` after `'ethernet-400g',`.

`src/engines/fabric.ts`: change line 1 to `import type { CustomFabricInput, GPU } from '@utils/schemas'` in the same edit as the first `GPU` use below. In `FABRIC_SPECS`, between `ethernet-400g` and `ethernet-100g`, add:

```ts
  // DGX Spark / DGX Station scale-out: ConnectX-7 at 200 Gb/s (NVIDIA Sync cluster assistant)
  'ethernet-200g': {
    type: 'ethernet-200g',
    label: '200GbE (ConnectX-7)',
    portGBps: 25,
    classFactor: 1.0,
  },
```

Append at the end of the file:

```ts
/**
 * Share of line rate a stage handoff reaches with GPUDirect RDMA (HGX-class nodes).
 * ASSUMPTION: the only hint is GH200 all_reduce at 45.4 of 50 GB/s. Results are
 * insensitive to it: doubling port speed changes decode by under 1% (vllm#6610:
 * 21.0 tok/s at 400G vs 21.1 at 800G on 2x GH200 PP=2).
 */
export const HGX_EFFECTIVE_FRACTION = 0.8

/**
 * Share of line rate on DGX Spark (GB10), which has no GPUDirect RDMA.
 * SECONDARY SOURCE, one measurement: NCCL send/recv ~9 GB/s vs 24.6 GB/s with RDMA on
 * a Spark 200G link (multimodalflow.net, DGX Spark dual-node NCCL RDMA).
 */
export const GB10_EFFECTIVE_FRACTION = 0.37

/**
 * Fixed cost of one stage handoff: the 10 us RDMA small-message floor (arXiv 2511.15076).
 * vLLM's metadata exchange is likely 50-200 us, still under 1% of a decode step.
 */
export const FABRIC_HOP_LATENCY_S = 10e-6

/** eta: the fraction of the fabric's line rate a stage handoff reaches on this GPU's node */
export function effectiveFraction(gpu: GPU): number {
  return gpu.id === 'nvidia-gb10' ? GB10_EFFECTIVE_FRACTION : HGX_EFFECTIVE_FRACTION
}

/**
 * Effective scale-out bandwidth of one node, GB/s: one NIC per GPU, and each TP rank
 * sends its 1/tp slice, so the node's aggregate applies, times eta.
 */
export function interNodeGBps(portGBps: number, gpusPerNode: number, fraction: number): number {
  return perNodeFabricGBps(portGBps, gpusPerNode) * fraction
}

/**
 * One pipeline stage-boundary transfer of `tokens` tokens, in seconds. vLLM sends
 * hidden_states and residual (2 tensors x 2 bytes, BF16) per token
 * (distributed/parallel_state.py send_tensor_dict).
 */
export function fabricHopSeconds(tokens: number, hiddenSize: number, gbps: number): number {
  return (tokens * hiddenSize * 2 * 2) / (gbps * 1e9) + FABRIC_HOP_LATENCY_S
}

/**
 * vLLM's default max_num_batched_tokens for the OpenAI server (engine/arg_utils.py):
 * 16384 on GPUs with >= 160 GB, 8192 on >= 70 GB other than A100, else 2048.
 */
export function maxNumBatchedTokens(gpu: GPU): number {
  if (gpu.vram_gb >= 160) return 16384
  if (gpu.vram_gb >= 70 && !/a100/i.test(gpu.name)) return 8192
  return 2048
}

/** Microbatches a burst of B prompts of T tokens splits into at chunk size C: ceil(B x T / C) */
export function prefillMicrobatches(
  batchSize: number,
  sequenceLength: number,
  chunkTokens: number,
): number {
  return Math.max(1, Math.ceil((batchSize * sequenceLength) / chunkTokens))
}

/**
 * GPipe fill of a prefill over `stages` pipeline stages, M / (M + N - 1). Times N it is
 * the speedup N x M / (M + N - 1); one microbatch (M = 1) walks the stages serially and
 * gains nothing (GPipe, arXiv 1811.06965).
 */
export function prefillPipelineFill(microbatches: number, stages: number): number {
  if (stages <= 1) return 1
  return microbatches / (microbatches + stages - 1)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `npx vitest run src/engines/fabric.test.ts src/store/urlSerializer.test.ts src/components/inputs/InterNodeFabricSelector.test.tsx`
Expected: PASS (the fabric selector now lists 200GbE between 400GbE and 100GbE).

- [ ] **Step 5: Typecheck, lint, commit**

Run: `npm run typecheck && npm run lint:fix && npm run lint`

```bash
rtk git add src/engines/types.ts src/engines/fabric.ts src/engines/fabric.test.ts src/store/urlSerializer.ts src/store/urlSerializer.test.ts
rtk git commit -m "feat(fabric): stage-hop time, eta, vLLM chunk size, microbatch fill, 200GbE preset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---

### Task 4b: Multi-node prefill and decode from bytes over the fabric (replace the heuristic)

**Files:**
- Modify: `src/engines/types.ts:167-178` (`FabricSpec`), `:213-238` (`MultiGPUVRAMBreakdown`)
- Modify: `src/engines/fabric.ts:4-21`, `:32-69` (`classFactor`), `:83-142` (delete heuristic), `:150-161`
- Modify: `src/engines/index.ts:20-28`
- Modify: `src/engines/multi-gpu.ts:131-134`, `:210-213`, `:284-287`, `:359-362`
- Modify: `src/engines/multi-node.ts` (params, composition)
- Modify: `src/engines/performance.ts:85-101`, `:253-296`, `:322-361`
- Modify: `src/workers/calculation.worker.ts:114-122`, `:220-231`, `:305-313`
- Modify: `src/hooks/useInferenceCalculation.ts:106-159`, `:387-398`
- Modify: `src/components/outputs/MultiGPUBreakdownChart.tsx:184-189`
- Test: `src/engines/fabric.test.ts`, `src/engines/multi-node.test.ts`, `src/engines/performance.test.ts`, `src/engines/multi-gpu.test.ts:826-828`, `src/engines/inference.integration.test.ts:399-422`, `src/utils/exportPptx.test.ts:144-156` and `:534-546`, `src/components/outputs/MultiGPUBreakdownChart.test.tsx:189-200`

**Interfaces:**
- Consumes: Task 4a functions.
- Produces:
  - `FabricSpec = { type: FabricType; label: string; portGBps: number }` (no `classFactor`).
  - `MultiGPUVRAMBreakdown`: removes `interNodeDecodeEfficiency`, `interNodePrefillEfficiency`; adds `interNodeGBps: number` (0 at one node); `bubbleEfficiency` is the prefill fill `M / (M + N - 1)`; `scalingEfficiency = intraNodeEfficiency`; `prefillScalingEfficiency = intraNodeEfficiency x bubbleEfficiency`.
  - `calculateMultiNodeVRAM(params)` requires `sequenceLength: number`.
  - `estimatePerformance`: decode step adds `(N - 1) / stages x fabricHopSeconds(B, hidden, interNodeGBps)` per stage; no inter-node efficiency multiplier; `prefillSeconds` adds `(N - 1) x fabricHopSeconds(T, hidden, interNodeGBps) / B`.
  - Removed from `fabric.ts` and the barrel: `FABRIC_REFERENCE_GBPS`, `PP_BASE_EFFICIENCY`, `fabricPrefillEfficiency`, `fabricDecodeEfficiency`, `pipelineBubbleEfficiency`.

- [ ] **Step 1: Write the failing tests**

Append to `src/engines/performance.test.ts`, extending its imports in the same edit with `import { DEFAULT_KV_TIER, kvTierSummary } from './kv-tier'`, `import { calculateMultiNodeVRAM } from './multi-node'`, `import { effectiveFraction, FABRIC_SPECS, fabricHopSeconds, interNodeGBps } from './fabric'`, `import type { FabricType, QuantizationFormat } from './types'` (merge with the existing `./types` import), and `PREFILL_MFU` in the existing `./constants` import (`calculateMoEActiveParams` is already imported from `./inference`):

```ts
describe('multi-node prefill and decode from bytes over the fabric (spec Section 3b)', () => {
  const allModels = validateModels(modelsData)
  const allGPUs = validateGPUs(gpusData)
  const model = allModels.find((m) => m.id === 'meta-llama-llama-3.1-70b')
  const h100 = allGPUs.find((g) => g.id === 'nvidia-h100-80gb-sxm')
  const gb10 = allGPUs.find((g) => g.id === 'nvidia-gb10')
  const b200 = allGPUs.find((g) => g.id === 'nvidia-b200-192gb')
  if (!model || !h100 || !gb10 || !b200) throw new Error('fixture not found')

  function run(
    gpu: GPU,
    o: { gpusPerNode: number; numNodes: number; batchSize: number; sequenceLength: number; fabric: Exclude<FabricType, 'custom'>; quantization: QuantizationFormat },
  ) {
    const singleGPU = calculateInferenceVRAM({ model, quantization: o.quantization, sequenceLength: o.sequenceLength, batchSize: o.batchSize })
    const multi = calculateMultiNodeVRAM({
      singleGPU,
      model,
      gpuVramGB: gpu.vram_gb,
      gpusPerNode: o.gpusPerNode,
      numNodes: o.numNodes,
      intraNodeStrategy: 'tensor-parallel',
      gpu,
      fabric: FABRIC_SPECS[o.fabric],
      batchSize: o.batchSize,
      sequenceLength: o.sequenceLength,
      quantization: o.quantization,
    })
    const perf = estimatePerformance({
      model,
      gpu,
      quantization: o.quantization,
      sequenceLength: o.sequenceLength,
      batchSize: o.batchSize,
      multiGPUResult: multi,
    })
    return { singleGPU, multi, perf }
  }
  const hgx = { gpusPerNode: 8, batchSize: 1, sequenceLength: 8192, fabric: 'ethernet-400g' as const, quantization: 'fp8' as const }

  it('2-node batch-1 prefill = single-node prefill + one hop of the whole prompt, exactly', () => {
    const one = run(h100, { ...hgx, numNodes: 1 })
    const two = run(h100, { ...hgx, numNodes: 2 })
    const gbps = interNodeGBps(FABRIC_SPECS['ethernet-400g'].portGBps, 8, effectiveFraction(h100))
    expect(two.multi.interNodeGBps).toBeCloseTo(320, 10)
    const hop = fabricHopSeconds(8192, model.hidden_size, gbps)
    expect(two.perf.prefillSeconds?.toNumber()).toBeCloseTo((one.perf.prefillSeconds?.toNumber() ?? Number.NaN) + hop, 10)
  })

  it('2x GB10 on 200 GbE: TTFT within 1% of one GB10 (was 15x slower under the heuristic)', () => {
    const single = estimatePerformance({ model, gpu: gb10, quantization: 'fp8', sequenceLength: 8192, batchSize: 1 })
    const two = run(gb10, { gpusPerNode: 1, numNodes: 2, batchSize: 1, sequenceLength: 8192, fabric: 'ethernet-200g', quantization: 'fp8' })
    const ratio = two.perf.timeToFirstToken.div(single.timeToFirstToken).toNumber()
    expect(Math.abs(ratio - 1)).toBeLessThan(0.01)
  })

  it('doubling port speed changes decode by under 1%', () => {
    const slow = run(h100, { ...hgx, numNodes: 2, batchSize: 32, fabric: 'ethernet-400g' })
    const fast = run(h100, { ...hgx, numNodes: 2, batchSize: 32, fabric: 'ethernet-800g' })
    const change = fast.perf.tokensPerSecond.div(slow.perf.tokensPerSecond).toNumber() - 1
    expect(change).toBeGreaterThanOrEqual(0)
    expect(change).toBeLessThan(0.01)
  })

  it('batch 32 pipelines M = ceil(B x T / C) = 32 microbatches over 2 nodes', () => {
    const { multi } = run(h100, { ...hgx, numNodes: 2, batchSize: 32 })
    expect(multi.bubbleEfficiency).toBeCloseTo(32 / 33, 12)
    expect(multi.prefillScalingEfficiency).toBeCloseTo(INTERCONNECT_SPECS['nvlink-4'].tpScalingEfficiency * (32 / 33), 12)
    const longPrompt = run(h100, { ...hgx, numNodes: 2, batchSize: 1, sequenceLength: 32768 })
    expect(longPrompt.multi.bubbleEfficiency).toBeCloseTo(4 / 5, 12) // M = 32768 / 8192
  })

  it('single node, batch > 1: prefill is exactly the per-prompt formula (burst / B cancels)', () => {
    const { perf } = run(h100, { ...hgx, numNodes: 1, batchSize: 8 })
    const promptTokens = 8192
    const flops = calculateMoEActiveParams(model) * 2e9 * promptTokens + 2 * model.num_hidden_layers * promptTokens ** 2 * model.hidden_size
    const effective = (h100.fp16_tflops ?? 0) * 1e12 * PREFILL_MFU.toNumber() * 8 * INTERCONNECT_SPECS['nvlink-4'].tpScalingEfficiency
    expect(perf.prefillSeconds?.toNumber()).toBeCloseTo(flops / effective, 9)
  })

  it('KV-tier verdict on the tightest config: B200x8, 4 nodes, 1.6T, batch 32 still resumes faster', () => {
    const { singleGPU, multi, perf } = run(b200, { gpusPerNode: 8, numNodes: 4, batchSize: 32, sequenceLength: 8192, fabric: 'ethernet-1600g', quantization: 'fp16' })
    const summary = kvTierSummary({
      settings: { ...DEFAULT_KV_TIER, tier: 'network' },
      maxHotSessions: 32,
      perGPUKVGB: multi.perGPU.kvCache.toNumber(),
      totalKVGB: singleGPU.kvCache.toNumber(),
      concurrentUsers: 32,
      multi,
      recomputeSeconds: perf.prefillSeconds?.toNumber() ?? null,
      gpuId: b200.id,
    })
    expect(summary?.resumeFaster).toBe(true)
    expect(summary?.resumeSeconds).toBeCloseTo(0.037, 3)
    expect(perf.prefillSeconds?.toNumber()).toBeCloseTo(0.048, 2)
  })
})
```

In the same file, in `describe('multi-node roofline separation', ...)`, add `interNodeGBps: 320,` to the `multiGPUResult` object literal (after `gpusPerNode: 8,`) so the 4-node spread carries a fabric; in the literal fixture near line 575 replace the lines `interNodeDecodeEfficiency: 1,` and `interNodePrefillEfficiency: 1,` with `interNodeGBps: 0,`.

In `src/engines/multi-node.test.ts`:
- Change line 1 to `import { FABRIC_SPECS, resolveFabricSpec } from '@engines/fabric'`.
- Add `sequenceLength: 4096,` to the `base` object (after `batchSize: 1,`).
- Replace the four tests `'separates decode from prefill efficiency once nodes > 1'`, `'charges a pipeline bubble at batch 1 across nodes'`, `'rewards a faster fabric'` and `'puts a 72-GPU NVL72 node at the pipeline efficiency ceiling, not beyond it'` with:

```ts
  it('keeps decode efficiency intra-node only; prefill adds the pipeline fill', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.scalingEfficiency).toBe(result.intraNodeEfficiency)
    expect(result.prefillScalingEfficiency).toBeCloseTo(result.intraNodeEfficiency * result.bubbleEfficiency, 12)
  })

  it('gives one request within one scheduler chunk no cross-node prefill speedup', () => {
    // B = 1, T = 4096 <= C = 16384 (288 GB MI355X): M = 1, fill = 1 / 4
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.bubbleEfficiency).toBeCloseTo(0.25, 12)
  })

  it('reports the effective bandwidth between servers: port x GPUs per node x eta', () => {
    const slow = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4, fabric: FABRIC_SPECS['ethernet-100g'] })
    const fast = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4, fabric: FABRIC_SPECS['ethernet-1600g'] })
    expect(slow.interNodeGBps).toBeCloseTo(12.5 * 8 * 0.8, 10)
    expect(fast.interNodeGBps).toBeCloseTo(200 * 8 * 0.8, 10)
  })

  it('handles a 72-GPU NVL72 node per stage', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 72, numNodes: 2, fabric: resolveFabricSpec('ethernet-800g', null) })
    expect(result.interNodeGBps).toBeCloseTo(100 * 72 * 0.8, 10)
    expect(result.totalPerGPU.isFinite()).toBe(true)
  })
```

In `src/engines/multi-gpu.test.ts` lines 826-828 replace the two efficiency assertions with `expect(result.interNodeGBps).toBe(0)` (keep the `bubbleEfficiency` one).

In `src/engines/fabric.test.ts`: delete the `describe` blocks `fabricPrefillEfficiency`, `fabricDecodeEfficiency`, `pipelineBubbleEfficiency` and the test `'marks only the InfiniBand entries with the class bonus'`; in `'builds a spec from custom input'` replace `expect(spec.classFactor).toBe(1.0)` with `expect('classFactor' in spec).toBe(false)`; remove `fabricDecodeEfficiency`, `fabricPrefillEfficiency`, `pipelineBubbleEfficiency` from the import in the same edit.

Add `sequenceLength: <the singleGPU's sequenceLength>` to every other `calculateMultiNodeVRAM({...})` call: `src/engines/inference.integration.test.ts:399` and `:411` (`sequenceLength: 131072`), `src/utils/exportPptx.test.ts:144` and `:534` (`sequenceLength: 4096`), `src/components/outputs/MultiGPUBreakdownChart.test.tsx:189` (`sequenceLength: 4096`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/engines/performance.test.ts src/engines/multi-node.test.ts`
Expected: FAIL: `interNodeGBps` undefined; 2x GB10 TTFT is ~15x one GB10; bubble at B32 is 32/35 (old `max(B, N)` heuristic).

- [ ] **Step 3: Types and fabric cleanup**

`src/engines/types.ts`: delete the `classFactor` member (and its doc comment) from `FabricSpec`. Replace the `MultiGPUVRAMBreakdown` members from `intraNodeEfficiency` through `prefillScalingEfficiency` (lines 217-238) with:

```ts
  /** Intra-node scaling efficiency, from INTERCONNECT_SPECS */
  intraNodeEfficiency: number
  /**
   * Effective scale-out bandwidth of one node, GB/s: port x GPUs per node x eta
   * (fabric.ts effectiveFraction). 0 when numNodes === 1.
   */
  interNodeGBps: number
  /** Prefill pipeline fill across nodes, M / (M + N - 1); 1.0 when numNodes === 1 */
  bubbleEfficiency: number
  /** Intra-node efficiency (the decode roofline never multiplies by it) */
  scalingEfficiency: number
  /** Prefill roofline efficiency: intraNodeEfficiency * bubbleEfficiency */
  prefillScalingEfficiency: number
```

`src/engines/fabric.ts`: delete lines 4-21 (`FABRIC_REFERENCE_GBPS`, `PP_BASE_EFFICIENCY`, `EFFICIENCY_FLOOR`), every `classFactor: ...` line in `FABRIC_SPECS` and in `resolveFabricSpec`'s custom branch, and lines 83-142 (`halvingsBelowReference`, `clamp`, `fabricPrefillEfficiency`, `fabricDecodeEfficiency`, `pipelineBubbleEfficiency`).

`src/engines/index.ts` lines 20-28 become:

```ts
// Scale-out fabric
export {
  effectiveFraction,
  FABRIC_SPECS,
  fabricHopSeconds,
  interNodeGBps,
  maxNumBatchedTokens,
  perNodeFabricGBps,
  prefillMicrobatches,
  prefillPipelineFill,
  resolveFabricSpec,
} from './fabric'
```

`src/engines/multi-gpu.ts`: in each of the four breakdown literals (lines 131-134, 210-213, 284-287, 359-362) replace `interNodeDecodeEfficiency: 1,` and `interNodePrefillEfficiency: 1,` with `interNodeGBps: 0,`.

- [ ] **Step 4: Multi-node composition**

In `src/engines/multi-node.ts`: replace the `./fabric` import with

```ts
import {
  effectiveFraction,
  interNodeGBps,
  maxNumBatchedTokens,
  prefillMicrobatches,
  prefillPipelineFill,
} from './fabric'
```

add `sequenceLength: number` to the params type (after `batchSize: number`) and to the destructuring, add `sequenceLength: 8192,` to the `@example`, and replace lines 141-159 with:

```ts
  // Section 3b (ADR 0007): stage handoffs cost bytes over the fabric (performance.ts
  // prices them from interNodeGBps), and prefill pipelines M = ceil(B x T / C)
  // microbatches across the node stages, C being vLLM's max_num_batched_tokens.
  const gbps = interNodeGBps(fabric.portGBps, gpusPerNode, effectiveFraction(gpu))
  const microbatches = prefillMicrobatches(batchSize, sequenceLength, maxNumBatchedTokens(gpu))
  const bubbleEfficiency = prefillPipelineFill(microbatches, numNodes)
  const intraNodeEfficiency = inner.intraNodeEfficiency

  return {
    ...inner,
    numGPUs: gpusPerNode * numNodes,
    numNodes,
    gpusPerNode,
    intraNodeEfficiency,
    interNodeGBps: gbps,
    bubbleEfficiency,
    scalingEfficiency: intraNodeEfficiency,
    prefillScalingEfficiency: intraNodeEfficiency * bubbleEfficiency,
    singleGPUBaseline: singleGPU.total,
  }
```

- [ ] **Step 5: Performance: decode hop, amortized prefill hop**

In `src/engines/performance.ts`, add `import { fabricHopSeconds } from './fabric'` in the same edit as its first use. Replace `decodeLayout` (lines 85-101) with:

```ts
function decodeLayout(model: Model, multi: MultiGPUVRAMBreakdown | null | undefined) {
  if (!multi || multi.numGPUs <= 1) {
    return { strategy: null, stages: 1, gpusPerStage: 1, kvShards: 1, numNodes: 1, interNodeGBps: 0 }
  }
  const intraPP = multi.strategy === 'pipeline-parallel'
  const gpusPerStage = intraPP ? 1 : multi.gpusPerNode
  let kvShards = 1
  if (multi.strategy === 'expert-parallel') kvShards = gpusPerStage
  else if (gpusPerStage > 1) kvShards = kvCacheTPShards(model, gpusPerStage)
  return {
    strategy: gpusPerStage > 1 ? multi.strategy : null,
    stages: multi.numNodes * (intraPP ? multi.gpusPerNode : 1),
    gpusPerStage,
    kvShards,
    numNodes: multi.numNodes,
    interNodeGBps: multi.interNodeGBps,
  }
}
```

Replace lines 253-281 (the step-4 comment through `tokensPerSecond`) with:

```ts
  // 4. Roofline per stage, plus communication per layer: two all-reduces for tensor
  //    parallelism (after attention and after the MLP), one dispatch + combine
  //    all-to-all for expert parallelism. Across servers each step also crosses N - 1
  //    stage boundaries with B tokens (Section 3b), spread over the stages
  //    (conservative: vLLM sends asynchronously). A step flows through the stages in
  //    turn; with batchSize sequences in flight the pipeline overlaps them, less the
  //    bubble B / (B + stages - 1). A decode token cannot be split into micro-batches
  //    (unlike a prompt, see fabric.ts prefillPipelineFill), so at batch 1 pipeline
  //    parallelism gives no decode speedup.
  const link = INTERCONNECT_SPECS[resolveInterconnect(gpu, layout.gpusPerStage)]
  const layersPerStage = model.num_hidden_layers / layout.stages
  let commSecondsPerLayer = 0
  if (layout.strategy === 'tensor-parallel') {
    commSecondsPerLayer = (2 * link.allreduceLatencyUs) / 1e6
  } else if (layout.strategy === 'expert-parallel' && split) {
    // bandwidthGBps is bidirectional per GPU; one direction carries each transfer
    commSecondsPerLayer = expertAllToAllSeconds(
      batchSize / layout.gpusPerStage,
      split.expertsPerToken,
      model.hidden_size,
      link.bandwidthGBps / 2,
      link.allreduceLatencyUs,
    )
  }
  const multiNode = layout.numNodes > 1 && layout.interNodeGBps > 0
  const hopSecondsPerStage = multiNode
    ? ((layout.numNodes - 1) / layout.stages) *
      fabricHopSeconds(batchSize, model.hidden_size, layout.interNodeGBps)
    : 0
  const stageSeconds = Decimal.max(memorySeconds, computeSeconds)
    .add(commSecondsPerLayer * layersPerStage)
    .add(hopSecondsPerStage)
  const tokensPerSecond = new Decimal(batchSize)
    .div(stageSeconds)
    .mul(new Decimal(batchSize).div(batchSize + layout.stages - 1))
```

In the offload baseline (lines 292-294) change the `baselineStageSeconds` expression to add the hop too:

```ts
    const baselineStageSeconds = Decimal.max(baselineMemorySeconds, computeSeconds)
      .add(commSecondsPerLayer * layersPerStage)
      .add(hopSecondsPerStage)
```

Replace the prefill comment line `//    Batch is NOT applied: TTFT is a per-request latency for one sequence of T tokens.` with:

```ts
  //    Per request: a burst of B prompts takes B times the FLOPs on the same machine, so
  //    dividing by B leaves one prompt's FLOPs (ADR 0007). Across servers the pipeline
  //    fill (prefillScalingEfficiency) and N - 1 hops of the prompt are amortized the same way.
```

and after `prefillSeconds = linearFLOPs.add(attentionFLOPs).div(effectiveFLOPS)` insert:

```ts
    if (multiNode) {
      prefillSeconds = prefillSeconds.add(
        new Decimal(fabricHopSeconds(sequenceLength, model.hidden_size, layout.interNodeGBps))
          .mul(layout.numNodes - 1)
          .div(batchSize),
      )
    }
```

- [ ] **Step 6: Worker and sync hook**

`src/workers/calculation.worker.ts`: in the response type (lines 117-119) replace `interNodeDecodeEfficiency: number` and `interNodePrefillEfficiency: number` with `interNodeGBps: number`; add `sequenceLength,` to the `calculateMultiNodeVRAM({...})` call (after `batchSize,`); in the serialized payload (lines 308-309) replace the two efficiency lines with `interNodeGBps: multiGPUResult.interNodeGBps,`.

`src/hooks/useInferenceCalculation.ts`: in `reconstructMultiGPUBreakdown`'s parameter type replace `interNodeDecodeEfficiency: number` and `interNodePrefillEfficiency: number` with `interNodeGBps: number`, and in its body replace the two efficiency lines with `interNodeGBps: serialized.interNodeGBps,`; add `sequenceLength,` to the sync-path `calculateMultiNodeVRAM({...})` call (after `batchSize,`).

- [ ] **Step 7: Multi-node line of the multi-GPU chart**

In `src/components/outputs/MultiGPUBreakdownChart.test.tsx`, in `'reports the per-node split (not just the total) across multiple servers'`, add after the strategy-line assertion:

```tsx
    // 100 GB/s per port x 8 GPUs x eta 0.8 (HGX assumption, fabric.ts)
    expect(screen.getByText(/640 GB\/s effective between servers/)).toBeInTheDocument()
```

In `src/components/outputs/MultiGPUBreakdownChart.tsx` replace the efficiency paragraph (lines 184-189) with:

```tsx
          <p>
            Modelled from bandwidth, not measured:{' '}
            {(breakdown.intraNodeEfficiency * 100).toFixed(0)}% intra-server ·{' '}
            {breakdown.interNodeGBps.toFixed(0)} GB/s effective between servers ·{' '}
            {(breakdown.bubbleEfficiency * 100).toFixed(0)}% pipeline fill on prefill
          </p>
```


- [ ] **Step 8: Run the tests**

Run: `npx vitest run`
Expected: PASS.

- [ ] **Step 9: Typecheck, lint, commit**

Run: `npm run typecheck && npm run lint:fix && npm run lint`
Expected: no errors.

```bash
rtk git add src/engines src/workers/calculation.worker.ts src/hooks/useInferenceCalculation.ts src/utils/exportPptx.test.ts src/components/outputs/MultiGPUBreakdownChart.tsx src/components/outputs/MultiGPUBreakdownChart.test.tsx
rtk git commit -m "feat(engine): multi-node prefill/decode from bytes over the fabric (ADR 0007)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 4c: Relabel the first-token figure (UI and PPTX)

**Files:**
- Create: `src/utils/perfLabels.ts`, `src/utils/perfLabels.test.ts`
- Modify: `src/components/outputs/PerformanceSection.tsx:41`
- Modify: `src/utils/exportPptx.ts:480`, `:569`
- Test: `src/components/outputs/PerformanceSection.test.tsx`, `src/utils/exportPptx.test.ts`

**Interfaces:**
- Produces: `firstTokenLabel(batchSize: number): string` — `'Time to first token'` at batch 1, `'Prefill per request (amortized over batch {B})'` for B > 1 (ADR 0007: for B > 1 the figure is a burst's prefill divided by B). Consumed again by `VerdictBlock` in Task 6b.

- [ ] **Step 1: Write the failing tests**

Create `src/utils/perfLabels.test.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { firstTokenLabel } from './perfLabels'

describe('firstTokenLabel', () => {
  it('is a plain time to first token for one request', () => {
    expect(firstTokenLabel(1)).toBe('Time to first token')
  })

  it('says the figure is amortized over the batch for B > 1', () => {
    expect(firstTokenLabel(32)).toBe('Prefill per request (amortized over batch 32)')
  })
})
```

Append to `src/components/outputs/PerformanceSection.test.tsx` (inside its `describe`):

```tsx
  it('labels the first-token figure as amortized over the batch when B > 1', () => {
    render(<PerformanceSection performance={performance} concurrentUsers={8} batchSize={8} />)
    expect(screen.getByText('Prefill per request (amortized over batch 8)')).toBeInTheDocument()
    expect(screen.queryByText(/Time to First Token/i)).not.toBeInTheDocument()
  })
```

Append to `src/utils/exportPptx.test.ts` (inside the top-level `describe`, reusing its `model`, `gpu`, `performance`, `tableRows`):

```ts
  it('labels the first-token row as amortized over the batch when B > 1 (ADR 0007)', async () => {
    const singleGPU = calculateInferenceVRAM({ model, quantization: 'fp16', sequenceLength: 4096, batchSize: 8 })
    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 8,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 8,
      offload: null,
    })
    const perfRows = tableRows(2)
    expect(perfRows.some(([label]) => label === 'Prefill per request (amortized over batch 8)')).toBe(true)
    expect(perfRows.some(([label]) => label === 'Time to First Token')).toBe(false)
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/utils/perfLabels.test.ts src/components/outputs/PerformanceSection.test.tsx src/utils/exportPptx.test.ts`
Expected: FAIL (`./perfLabels` missing; labels still "Time to First Token").

- [ ] **Step 3: Implement**

Create `src/utils/perfLabels.ts`:

```ts
/**
 * Label of the first-token figure. For B > 1 the engine's figure is the prefill of a
 * burst of B prompts divided by B (ADR 0007), not one request's wait: in a real burst
 * the last request waits about B times longer.
 */
export function firstTokenLabel(batchSize: number): string {
  return batchSize > 1
    ? `Prefill per request (amortized over batch ${batchSize})`
    : 'Time to first token'
}
```

`src/components/outputs/PerformanceSection.tsx:41`: replace the text `Time to First Token` with `{firstTokenLabel(batchSize)}`, adding `import { firstTokenLabel } from '@utils/perfLabels'` in the same edit.

`src/utils/exportPptx.ts`: line 480 `{ label: firstTokenLabel(batchSize), value: ttftLabel },` and line 569 `{ text: firstTokenLabel(batchSize), options: { fill: C.altRowFill } },`, adding the import in the same edit. `batchSize` is already destructured from the params (it feeds the Batch Size row).

- [ ] **Step 4: Run the tests, typecheck, lint, commit**

Run: `npx vitest run && npm run typecheck && npm run lint:fix && npm run lint`
Expected: all green.

```bash
rtk git add src/utils/perfLabels.ts src/utils/perfLabels.test.ts src/components/outputs/PerformanceSection.tsx src/components/outputs/PerformanceSection.test.tsx src/utils/exportPptx.ts src/utils/exportPptx.test.ts
rtk git commit -m "feat(ui): label the first-token figure as amortized over the batch

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---

### Task 5: `allowedOptions` in the inputs, grouped notices, soft warnings

**Files:**
- Create: `src/hooks/useAllowedOptions.ts`
- Create: `src/hooks/useConfigNotices.tsx`, `src/hooks/useConfigNotices.test.tsx`
- Create: `src/components/outputs/SoftWarnings.tsx`, `src/components/outputs/SoftWarnings.test.tsx`
- Create: `src/components/inputs/OffloadingPanel.test.tsx`
- Modify: `src/App.tsx:1-16`
- Modify: `src/components/inputs/GPUCountSelector.tsx:20-77`
- Modify: `src/components/inputs/ShardingStrategySelector.tsx:21`, `:120`
- Modify: `src/components/inputs/KVTierPanel.tsx:2`, `:65`
- Modify: `src/components/inputs/OffloadingPanel.tsx:63-135`
- Modify: `src/components/inputs/InterconnectSelector.tsx:4-16`
- Modify: `src/components/inputs/CPUOffloadToggle.tsx:11-17`
- Modify: `src/components/layout/ResultsPanel.tsx` (soft warnings after the interconnect warning, line 611)
- Test: `src/components/inputs/GPUCountSelector.test.tsx`, `src/components/inputs/KVTierPanel.test.tsx`

**Interfaces:**
- Consumes: `allowedOptions`, `AllowedOptions`, `softWarnings`, `SoftWarning` (Task 1b); `pendingNotice`, `clearNotice` (Task 2a).
- Produces:
  - `useAllowedOptions(): AllowedOptions` (reads `mode`, `shardingStrategy`, `offloadingEnabled`, `kvCacheOffload`, `frameworkPreset`, `selectedModel`, `selectedGPU` from the store). Fake stores in component tests must carry these seven fields.
  - `useConfigNotices(): void` — toasts `pendingNotice` once (`toast.warning(title, { description })`, one line per correction), then `clearNotice()`.
  - `SoftWarnings({ warnings }: { warnings: SoftWarning[] })` — one amber block per warning, `data-testid="soft-warning-{id}"`.

- [ ] **Step 1: Write the failing tests**

Create `src/hooks/useConfigNotices.test.tsx`:

```tsx
import modelsData from '@data/models.json'
import gpusData from '@data/gpus.json'
import { act, renderHook } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})
const { warning } = vi.hoisted(() => ({ warning: vi.fn() }))
vi.mock('sonner', () => ({ toast: { warning } }))

import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { useConfigNotices } from './useConfigNotices'

const L70 = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b') ?? null
const H100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm') ?? null

describe('useConfigNotices', () => {
  beforeEach(() => {
    warning.mockClear()
    useUIStore.setState({ ...DEFAULT_UI_CONFIG, selectedModel: L70, selectedGPU: H100, pendingNotice: null })
  })

  it('toasts one grouped notice per action and clears it', () => {
    renderHook(() => useConfigNotices())
    act(() => useUIStore.getState().setNumGPUs(6))
    expect(warning).toHaveBeenCalledTimes(1)
    expect(warning.mock.calls[0]?.[0]).toBe(`Adjusted for ${L70?.name}`)
    expect(useUIStore.getState().pendingNotice).toBeNull()
  })

  it('toasts an identical repeated correction again', () => {
    renderHook(() => useConfigNotices())
    act(() => useUIStore.getState().setNumGPUs(6))
    act(() => useUIStore.getState().setNumGPUs(6))
    expect(warning).toHaveBeenCalledTimes(2)
  })

  it('says nothing when nothing was corrected', () => {
    renderHook(() => useConfigNotices())
    act(() => useUIStore.getState().setNumGPUs(4))
    expect(warning).not.toHaveBeenCalled()
  })
})
```

Create `src/components/outputs/SoftWarnings.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SoftWarnings } from './SoftWarnings'

describe('SoftWarnings', () => {
  it('renders each warning with its id', () => {
    render(<SoftWarnings warnings={[{ id: 'W8', message: "256 experts don't split evenly across 6 GPUs." }]} />)
    expect(screen.getByTestId('soft-warning-W8')).toHaveTextContent("256 experts don't split evenly across 6 GPUs.")
  })

  it('renders nothing without warnings', () => {
    const { container } = render(<SoftWarnings warnings={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
```

Create `src/components/inputs/OffloadingPanel.test.tsx`:

```tsx
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})

import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { OffloadingPanel } from './OffloadingPanel'

const gpus = validateGPUs(gpusData)
const L8 = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-8b') ?? null

describe('OffloadingPanel', () => {
  beforeEach(() => useUIStore.setState({ ...DEFAULT_UI_CONFIG, selectedModel: L8, pendingNotice: null }))

  it('offers only NVMe on unified memory (R6)', () => {
    useUIStore.setState({ selectedGPU: gpus.find((g) => g.id === 'apple-m3-ultra') ?? null })
    useUIStore.getState().setOffloadingEnabled(true)
    render(<OffloadingPanel />)
    expect(screen.queryByText('CPU/RAM')).not.toBeInTheDocument()
    expect(screen.getByText('NVMe SSD')).toBeInTheDocument()
  })

  it('offers CPU/RAM and NVMe on a discrete GPU', () => {
    useUIStore.setState({ selectedGPU: gpus.find((g) => g.id === 'nvidia-h100-80gb-sxm') ?? null })
    useUIStore.getState().setOffloadingEnabled(true)
    render(<OffloadingPanel />)
    expect(screen.getByText('CPU/RAM')).toBeInTheDocument()
    expect(screen.getByText('NVMe SSD')).toBeInTheDocument()
  })
})
```

Replace `src/components/inputs/GPUCountSelector.test.tsx` with:

```tsx
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { MAX_GPUS_PER_NODE } from '@engines/constants'
import { fireEvent, render, screen } from '@testing-library/react'
import { maxGPUsFor } from '@utils/gpuLimits'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the fields the component
// and useAllowedOptions read.
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>((set) => ({
    numGPUs: 1,
    numNodes: 1,
    selectedGPU: null,
    selectedModel: null,
    mode: 'inference',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'none',
    setNumGPUs: (value: number) => set({ numGPUs: value }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { GPUCountSelector } from './GPUCountSelector'

const gpus = validateGPUs(gpusData)
const models = validateModels(modelsData)
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null
const model = (id: string) => models.find((m) => m.id === id) ?? null

describe('GPUCountSelector', () => {
  beforeEach(() => {
    useUIStore.setState({
      numGPUs: 1,
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      selectedModel: null,
      mode: 'inference',
      shardingStrategy: 'tensor-parallel',
    })
  })

  it('offers every count up to max_gpus_per_node with no model selected', () => {
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '7') // 8 stops: 1..8
  })

  it('raises the range to 72 stops for an NVL72 rack under pipeline parallel', () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-gb300-nvl72'), shardingStrategy: 'pipeline-parallel' })
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '71')
  })

  it('offers only valid tensor-parallel degrees (R14): Llama 3.1 70B on 8 GPUs = 1, 2, 4, 8', () => {
    useUIStore.setState({ selectedModel: model('meta-llama-llama-3.1-70b') })
    render(<GPUCountSelector />)
    const slider = screen.getByRole('slider')
    expect(slider).toHaveAttribute('max', '3')
    fireEvent.change(slider, { target: { value: '2' } })
    expect(useUIStore.getState().numGPUs).toBe(4)
  })

  it('renders no slider for a single-GPU part', () => {
    useUIStore.setState({ selectedGPU: gpu('apple-m3-ultra') })
    render(<GPUCountSelector />)
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    expect(screen.getByText(/Single GPU/)).toBeInTheDocument()
  })

  it('falls back to the shared bound when no GPU is selected', () => {
    useUIStore.setState({ selectedGPU: null })
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', String(maxGPUsFor(null) - 1))
    expect(maxGPUsFor(null)).toBe(MAX_GPUS_PER_NODE)
  })

  it('states the cap in its tooltip', () => {
    render(<GPUCountSelector />)
    // InfoTip only renders its `text` prop into the DOM once its trigger is opened.
    fireEvent.click(screen.getByRole('button', { name: /more info/i }))
    expect(screen.getByText(/Capped at 8/)).toBeInTheDocument()
  })
})
```

In `src/components/inputs/KVTierPanel.test.tsx`, extend the mocked store's type and initial state with `selectedModel: null`, `mode: 'inference'`, `shardingStrategy: 'tensor-parallel'`, `offloadingEnabled: false`, `kvCacheOffload: false`, `frameworkPreset: 'none'` (add them to the `create<{...}>` type as `unknown`/literal types and to the initial object), reset them in the `beforeEach` setState, and append:

```tsx
  it('offers only None when the KV cache is already offloaded (R12)', () => {
    useUIStore.setState({ selectedGPU: findGPU('nvidia-h100-80gb-sxm'), offloadingEnabled: true, kvCacheOffload: true })
    render(<KVTierPanel />)
    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['None'])
  })

  it('hides host-memory tiers on unified memory (R6)', () => {
    useUIStore.setState({ selectedGPU: findGPU('apple-m3-ultra') })
    render(<KVTierPanel />)
    expect(screen.queryByRole('option', { name: /Host memory/ })).not.toBeInTheDocument()
  })
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/hooks/useConfigNotices.test.tsx src/components/outputs/SoftWarnings.test.tsx src/components/inputs`
Expected: FAIL: missing modules; the GPU slider max is `8`; CPU/RAM is offered on M3; KV tiers not filtered by R12/R6.

- [ ] **Step 3: Hooks and the soft-warnings block**

Create `src/hooks/useAllowedOptions.ts`:

```ts
import { type AllowedOptions, allowedOptions } from '@engines/config-rules'
import { useUIStore } from '@store/uiStore'
import { useMemo } from 'react'

/** What each input may offer for the current selection (ADR 0004: never offer an impossible value) */
export function useAllowedOptions(): AllowedOptions {
  const mode = useUIStore((s) => s.mode)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)
  const offloadingEnabled = useUIStore((s) => s.offloadingEnabled)
  const kvCacheOffload = useUIStore((s) => s.kvCacheOffload)
  const frameworkPreset = useUIStore((s) => s.frameworkPreset)
  const model = useUIStore((s) => s.selectedModel)
  const gpu = useUIStore((s) => s.selectedGPU)
  return useMemo(
    () =>
      allowedOptions(
        { mode, shardingStrategy, offloadingEnabled, kvCacheOffload, frameworkPreset },
        model,
        gpu,
      ),
    [mode, shardingStrategy, offloadingEnabled, kvCacheOffload, frameworkPreset, model, gpu],
  )
}
```

Create `src/hooks/useConfigNotices.tsx`:

```tsx
import { useUIStore } from '@store/uiStore'
import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * Show the store's correction notice as one toast per action ("Adjusted for ..." or
 * "Shared link adjusted"), one line per correction, then clear it. Mounted once in App.
 */
export function useConfigNotices(): void {
  const pendingNotice = useUIStore((s) => s.pendingNotice)
  const clearNotice = useUIStore((s) => s.clearNotice)

  useEffect(() => {
    if (!pendingNotice) return
    toast.warning(pendingNotice.title, {
      description: (
        <ul className="list-disc pl-4">
          {pendingNotice.lines.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      ),
    })
    clearNotice()
  }, [pendingNotice, clearNotice])
}
```

In `src/App.tsx` add `import { useConfigNotices } from '@hooks/useConfigNotices'` and call `useConfigNotices()` right after `useURLSync()`, in the same edit.

Create `src/components/outputs/SoftWarnings.tsx`:

```tsx
import type { SoftWarning } from '@engines/config-rules'

/** Soft rules (W3, W6, W8): shown inline, never corrected (spec Section 1) */
export function SoftWarnings({ warnings }: { warnings: SoftWarning[] }) {
  if (warnings.length === 0) return null
  return (
    <>
      {warnings.map((w) => (
        <div
          key={w.id}
          data-testid={`soft-warning-${w.id}`}
          className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4"
        >
          <p className="text-sm text-amber-800 dark:text-amber-200">⚠ {w.message}</p>
        </div>
      ))}
    </>
  )
}
```

In `src/components/layout/ResultsPanel.tsx`, after the interconnect-warning block (line 611) add `<SoftWarnings warnings={softWarnings({ mode, numGPUs, numNodes, shardingStrategy }, selectedModel, selectedGPU)} />`, adding `import { SoftWarnings } from '@components/outputs/SoftWarnings'` and `import { softWarnings } from '@engines/config-rules'` in the same edit.

- [ ] **Step 4: Inputs read `allowedOptions`**

`src/components/inputs/GPUCountSelector.tsx`: add `import { useAllowedOptions } from '@hooks/useAllowedOptions'` and, below the store selectors, `const { gpuCounts } = useAllowedOptions()`. Replace the `<input ... />` slider (lines 64-73) with:

```tsx
        {/* The slider moves over the allowed counts only (R1 + R14): a tensor-parallel
            degree vLLM refuses is not selectable. */}
        <input
          id="gpu-count"
          type="range"
          min={0}
          max={gpuCounts.length - 1}
          step={1}
          value={Math.max(0, gpuCounts.indexOf(numGPUs))}
          aria-valuetext={`${numGPUs} GPUs`}
          onChange={(e) => setNumGPUs(gpuCounts[Number(e.target.value)] ?? 1)}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
```

`src/components/inputs/ShardingStrategySelector.tsx`: replace line 21 (`const isMoE = ...`) with `const { strategies } = useAllowedOptions()` (import in the same edit) and line 120 `{isMoE && (` with `{strategies.includes('expert-parallel') && (`.

`src/components/inputs/KVTierPanel.tsx`: replace line 65 (`const tierOptions = KV_TIER_TYPES.filter(...)`) with `const tierOptions = useAllowedOptions().kvTiers` (import in the same edit) and drop `KV_TIER_TYPES` from the `@engines/kv-tier` import on line 2 in the same edit.

`src/components/inputs/OffloadingPanel.tsx`: add `const { offloadTargets } = useAllowedOptions()` below the store selectors (import in the same edit) and wrap the CPU/RAM `<button>` (lines 113-135) in `{offloadTargets.includes('cpu-ram') && ( ... )}`.

`src/components/inputs/InterconnectSelector.tsx`: replace lines 9-16 with:

```tsx
  const { interconnectOptions: options } = useAllowedOptions()

  // R5 + R13: variants only for multi-GPU parts with two or more options
  if (!selectedGPU || options.length < 2) {
    return null
  }

  const current = interconnectOverride ?? selectedGPU.interconnect ?? options[0]
```

(import `useAllowedOptions` in the same edit; `firstOption` is gone).

`src/components/inputs/CPUOffloadToggle.tsx`: replace lines 14-17 with:

```tsx
  // R6 + R8: only a ZeRO preset on a GPU with separate host memory can offload optimizer state
  if (!useAllowedOptions().cpuOffloadOptimizer) {
    return null
  }
```

(import in the same edit; `frameworkPreset` is then unused in the destructuring: remove it).

- [ ] **Step 5: Run the tests**

Run: `npx vitest run`
Expected: PASS.

- [ ] **Step 6: Typecheck, lint, commit**

Run: `npm run typecheck && npm run lint:fix && npm run lint`

```bash
rtk git add src/hooks/useAllowedOptions.ts src/hooks/useConfigNotices.tsx src/hooks/useConfigNotices.test.tsx src/App.tsx src/components/outputs/SoftWarnings.tsx src/components/outputs/SoftWarnings.test.tsx src/components/layout/ResultsPanel.tsx src/components/inputs
rtk git commit -m "feat(ui): inputs offer only allowed values; one grouped notice per action; soft warnings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 6a: Composition tests for InputPanel and ResultsPanel (before the layout change)

These characterization tests pass on the current layout and must still pass after Task 6b. They pin what the simplification may not lose: every input stays reachable, the essential inputs and the verdict figures stay visible, warnings stay visible.

**Files:**
- Create: `src/components/layout/InputPanel.test.tsx`
- Create: `src/components/layout/ResultsPanel.test.tsx`

**Interfaces:**
- Consumes: the real store (`useUIStore`, `DEFAULT_UI_CONFIG`) with `persist` mocked as a pass-through and `window.matchMedia` stubbed before the store module loads (same pattern as `src/store/uiStore.test.ts`, but the stub runs in `vi.hoisted` so static imports work).
- Produces: nothing new; Task 6b extends both files.

- [ ] **Step 1: Write the InputPanel composition test**

Create `src/components/layout/InputPanel.test.tsx`:

```tsx
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real store reads matchMedia at module load: stub it before any import runs.
vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})

import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { InputPanel } from './InputPanel'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => models.find((m) => m.id === id) ?? null
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null

function field(container: HTMLElement, id: string): HTMLElement | null {
  return container.querySelector(`#${id}`)
}

describe('InputPanel composition', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      pendingNotice: null,
    })
  })

  it('shows the essential inputs in inference', () => {
    const { container } = render(<InputPanel />)
    expect(screen.getByText('Inference Mode')).toBeVisible()
    for (const id of ['model-selector', 'quantization-picker', 'gpu-selector', 'gpu-count', 'node-count', 'sequence-length', 'concurrent-users']) {
      expect(field(container, id), id).toBeVisible()
    }
  })

  it('keeps every advanced input reachable', () => {
    useUIStore.setState({ numGPUs: 2 })
    const { container } = render(<InputPanel />)
    for (const id of ['batch-size', 'kv-quantization-picker', 'kv-tier']) {
      expect(field(container, id), id).toBeInTheDocument()
    }
    expect(screen.getByText('Intra-server sharding strategy')).toBeInTheDocument()
    expect(screen.getByText('Offloading Configuration')).toBeInTheDocument()
  })

  it('shows the fabric selector once the cluster spans servers', () => {
    useUIStore.setState({ numNodes: 2 })
    const { container } = render(<InputPanel />)
    expect(field(container, 'inter-node-fabric')).toBeInTheDocument()
  })

  it('hides servers in training mode', () => {
    useUIStore.setState({ mode: 'training' })
    const { container } = render(<InputPanel />)
    expect(field(container, 'node-count')).toBeNull()
    expect(screen.getByText('Training Configuration')).toBeVisible()
  })
})
```

- [ ] **Step 2: Write the ResultsPanel composition test**

Create `src/components/layout/ResultsPanel.test.tsx` (jsdom has no `Worker`, so the panel takes the synchronous fallback; results appear asynchronously, hence `findBy*`):

```tsx
import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})

import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { ResultsPanel } from './ResultsPanel'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => models.find((m) => m.id === id) ?? null
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null

/** At least one element matching `text` is visible (a copy may sit in a collapsed section) */
function expectSomeVisible(text: RegExp) {
  const visible = screen.getAllByText(text).some((el) => {
    try {
      expect(el).toBeVisible()
      return true
    } catch {
      return false
    }
  })
  expect(visible, String(text)).toBe(true)
}

describe('ResultsPanel composition', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      quantization: 'fp8',
      numGPUs: 2,
      pendingNotice: null,
    })
  })

  it('shows fit, decode speed, the first-token figure and max sessions', async () => {
    render(<ResultsPanel />)
    await screen.findAllByText(/tokens\/sec/)
    expectSomeVisible(/Fits Comfortably|Tight Fit|Does Not Fit/)
    expectSomeVisible(/tokens\/sec/)
    expectSomeVisible(/time to first token/i)
    expectSomeVisible(/sessions at 4,096 tokens/)
  })

  it('keeps the interconnect warning visible (W2 at TP-8 over PCIe 5)', async () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-h100-80gb-pcie'), numGPUs: 8 })
    render(<ResultsPanel />)
    await screen.findAllByText(/may have significant communication overhead/)
    expectSomeVisible(/may have significant communication overhead/)
  })

  it('keeps soft warnings visible (W8: 256 experts over 6 GPUs)', async () => {
    useUIStore.setState({ selectedModel: model('deepseek-r1'), shardingStrategy: 'expert-parallel', numGPUs: 6 })
    render(<ResultsPanel />)
    expect(await screen.findByTestId('soft-warning-W8')).toBeVisible()
  })
})
```

- [ ] **Step 3: Run the tests (they must pass on the current layout)**

Run: `npx vitest run src/components/layout`
Expected: PASS. If an id is missing, check the component's `htmlFor` target and use that id; do not change components in this task. These are the first tests to render the Headless UI comboboxes (ModelSelector, GPUSelector) against the real store: if a run throws `ResizeObserver is not defined`, add to `src/test/setup.ts`:

```ts
// Headless UI comboboxes observe their size; jsdom has no ResizeObserver
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
}
```

and add `src/test/setup.ts` to this task's commit.

- [ ] **Step 4: Lint, commit**

Run: `npm run lint:fix && npm run lint`

```bash
rtk git add src/components/layout/InputPanel.test.tsx src/components/layout/ResultsPanel.test.tsx
rtk git commit -m "test(ui): composition tests pinning what the layout change must keep

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---

### Task 6b: Essential/Advanced inputs, verdict block, collapsed details, PDF expansion, labels

**Files:**
- Create: `src/components/layout/advancedChanges.ts`, `src/components/layout/advancedChanges.test.ts`
- Create: `src/components/outputs/VerdictBlock.tsx`
- Create: `src/hooks/useResultExports.test.tsx`
- Modify: `src/components/layout/InputPanel.tsx` (full body)
- Modify: `src/components/layout/ResultsPanel.tsx:22` (imports), `:32-60` (state), `:454-658` (inference layout)
- Modify: `src/components/inputs/GPUSelector.tsx:15`, `:411-412` (InterconnectSelector moves to Advanced)
- Modify: `src/components/inputs/ShardingStrategySelector.tsx:23-26`, `:149-179`
- Modify: `src/components/inputs/GPUCountSelector.tsx:24-33`, `:78-86`
- Modify: `src/hooks/useResultExports.ts:46-77`
- Modify: `src/utils/exportPptx.ts:182`
- Test: `src/components/layout/InputPanel.test.tsx`, `src/components/layout/ResultsPanel.test.tsx`, `src/components/inputs/GPUCountSelector.test.tsx`, `src/components/inputs/ShardingStrategySelector.test.tsx`, `src/utils/exportPptx.test.ts:179-180`, `:237`

**Interfaces:**
- Consumes: `DEFAULT_UI_CONFIG`, `UIConfig` (Task 2a); `firstTokenLabel` (Task 4c); `SoftWarnings`, `softWarnings` (Task 5); `FRAMEWORK_PRESETS`; `maxGPUsFor`.
- Produces:
  - `countAdvancedChanges(config: UIConfig): number` — advanced settings differing from `DEFAULT_UI_CONFIG` (inference: batch size, KV precision, strategy, fabric when numNodes > 1, interconnect override, offloading, KV tier; training: batch size only).
  - `VerdictBlock(props: { fit: ReactNode; performance: PerformanceEstimate; batchSize: number; maxSessions: number | null; sequenceLength: number })`, root `data-testid="verdict"`.
  - `<details data-testid="advanced-settings">` in InputPanel, `<details data-testid="result-details">` in ResultsPanel; both controlled by component state so a recalculation (loading skeleton) never closes them.
  - PPTX config row `['GPUs per replica', '8 per server × 4 servers (32 total)']` (nodes > 1) or `['GPUs per replica', '1 (in one server)']`; no `Number of GPUs` row.

- [ ] **Step 1: Write the failing tests**

Create `src/components/layout/advancedChanges.test.ts`:

```ts
import { DEFAULT_UI_CONFIG } from '@store/uiStore'
import { describe, expect, it, vi } from 'vitest'
import { countAdvancedChanges } from './advancedChanges'

vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({ matches: false, media: query, addEventListener: () => {}, removeEventListener: () => {}, dispatchEvent: () => false }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})

describe('countAdvancedChanges', () => {
  it('is 0 at the defaults', () => {
    expect(countAdvancedChanges(DEFAULT_UI_CONFIG)).toBe(0)
  })

  it('counts each non-default advanced setting', () => {
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, batchSize: 8 })).toBe(1)
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, batchSize: 8, kvQuantization: 'fp8', offloadingEnabled: true })).toBe(3)
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, kvTier: { ...DEFAULT_UI_CONFIG.kvTier, tier: 'network' } })).toBe(1)
  })

  it('ignores the fabric while there is one server, and inert inputs in training', () => {
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, interNodeFabric: 'ethernet-100g' })).toBe(0)
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, numNodes: 2, interNodeFabric: 'ethernet-100g' })).toBe(1)
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, mode: 'training', kvQuantization: 'fp8', shardingStrategy: 'pipeline-parallel' })).toBe(0)
  })
})
```

Append to `src/components/layout/InputPanel.test.tsx`, adding `act` and `within` to its Testing Library import in the same edit:

```tsx
describe('InputPanel layout (ADR 0005)', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      pendingNotice: null,
    })
  })

  it('collapses Advanced at the defaults', () => {
    const { container } = render(<InputPanel />)
    const advanced = screen.getByTestId('advanced-settings')
    expect(advanced).not.toHaveAttribute('open')
    expect(within(advanced).getByText('Advanced')).toBeVisible()
    expect(field(container, 'batch-size')).not.toBeVisible()
  })

  it('opens Advanced with "N settings changed" when a value differs from its default', () => {
    useUIStore.setState({ batchSize: 8, kvQuantization: 'fp8' })
    render(<InputPanel />)
    const advanced = screen.getByTestId('advanced-settings')
    expect(advanced).toHaveAttribute('open')
    expect(within(advanced).getByText(/2 settings changed/)).toBeVisible()
  })

  it('opens Advanced when a setting changes after render', () => {
    render(<InputPanel />)
    act(() => useUIStore.getState().setBatchSize(8))
    expect(screen.getByTestId('advanced-settings')).toHaveAttribute('open')
    expect(screen.getByText(/1 setting changed/)).toBeVisible()
  })

  it('keeps the strategy reachable at 1 GPU on a multi-GPU part, so pipeline parallel is choosable before R14 snaps', () => {
    render(<InputPanel />)
    expect(screen.getByText('Intra-server sharding strategy')).toBeInTheDocument()
  })

  it('offers no strategy on a single-GPU part', () => {
    useUIStore.setState({ selectedGPU: gpu('apple-m3-ultra') })
    render(<InputPanel />)
    expect(screen.queryByText('Intra-server sharding strategy')).not.toBeInTheDocument()
  })

  it('labels the GPU count as GPUs per replica', () => {
    render(<InputPanel />)
    expect(screen.getByText('GPUs per replica (in one server)')).toBeVisible()
  })

  it('hides inputs that are inert in training', () => {
    useUIStore.setState({ mode: 'training' })
    const { container } = render(<InputPanel />)
    for (const id of ['quantization-picker', 'kv-quantization-picker', 'concurrent-users', 'node-count', 'kv-tier', 'gpu-count']) {
      expect(field(container, id), id).toBeNull()
    }
    expect(screen.queryByText('Offloading Configuration')).not.toBeInTheDocument()
    expect(screen.queryByText('Intra-server sharding strategy')).not.toBeInTheDocument()
  })

  it('shows the GPU count in training once a ZeRO preset makes it matter', () => {
    useUIStore.setState({ mode: 'training', frameworkPreset: 'deepspeed-zero3' })
    const { container } = render(<InputPanel />)
    expect(field(container, 'gpu-count')).toBeVisible()
  })
})
```

Append to `src/components/layout/ResultsPanel.test.tsx`, adding `act` and `within` to its Testing Library import in the same edit:

```tsx
describe('ResultsPanel layout (ADR 0005)', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      quantization: 'fp8',
      numGPUs: 2,
      pendingNotice: null,
    })
  })

  it('leads with a visible verdict: fit, decode, first token, sessions', async () => {
    render(<ResultsPanel />)
    const verdict = await screen.findByTestId('verdict')
    expect(verdict).toBeVisible()
    expect(within(verdict).getByText(/Fits Comfortably|Tight Fit|Does Not Fit/)).toBeVisible()
    expect(within(verdict).getByText(/tokens\/sec/)).toBeVisible()
    expect(within(verdict).getByText('Time to first token')).toBeVisible()
    expect(within(verdict).getByText('Max sessions at 4,096 tokens')).toBeVisible()
  })

  it('collapses the details by default', async () => {
    render(<ResultsPanel />)
    const details = await screen.findByTestId('result-details')
    expect(details).not.toHaveAttribute('open')
    expect(within(details).getByText(/Weights (measured from|estimated)/)).not.toBeVisible()
  })

  it('keeps Details open across a recalculation', async () => {
    render(<ResultsPanel />)
    const details = (await screen.findByTestId('result-details')) as HTMLDetailsElement
    act(() => {
      details.open = true
      details.dispatchEvent(new Event('toggle'))
    })
    act(() => useUIStore.getState().setBatchSize(2))
    await screen.findAllByText(/tokens\/sec/)
    expect(await screen.findByTestId('result-details')).toHaveAttribute('open')
  })
})
```

In `src/components/inputs/GPUCountSelector.test.tsx` append:

```tsx
  it('summarizes one replica across servers', () => {
    useUIStore.setState({ numGPUs: 8, numNodes: 2 })
    render(<GPUCountSelector />)
    expect(screen.getByText('8 GPUs per server × 2 servers per replica, tensor parallel')).toBeInTheDocument()
  })

  it('names expert parallel correctly (was "pipeline parallel")', () => {
    useUIStore.setState({ selectedModel: model('deepseek-r1'), shardingStrategy: 'expert-parallel', numGPUs: 8, numNodes: 1 })
    render(<GPUCountSelector />)
    expect(screen.getByText('8 GPUs per replica (in one server), expert parallel')).toBeInTheDocument()
  })
```

and reset `numNodes: 1` in its `beforeEach`.

In `src/components/inputs/ShardingStrategySelector.test.tsx` append:

```tsx
  it('stays visible at 1 GPU on a multi-GPU part, without an interconnect badge', () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-h100-80gb-sxm'), numGPUs: 1 })
    render(<ShardingStrategySelector />)
    expect(screen.getByText('Intra-server sharding strategy')).toBeInTheDocument()
    expect(screen.queryByText(/GB\/s ·/)).not.toBeInTheDocument()
  })
```

In `src/utils/exportPptx.test.ts` replace lines 179-180 with:

```ts
    expect(configRows).toContainEqual(['GPUs per replica', '8 per server × 4 servers (32 total)'])
    expect(configRows.some(([label]) => label === 'Number of GPUs')).toBe(false)
```

and line 237 with `expect(configRows).toContainEqual(['GPUs per replica', '1 (in one server)'])`.

Create `src/hooks/useResultExports.test.tsx`:

```tsx
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { capture } = vi.hoisted(() => ({ capture: vi.fn() }))
vi.mock('html2canvas-pro', () => ({ default: capture }))
vi.mock('jspdf', () => ({
  jsPDF: class {
    internal = { pageSize: { getWidth: () => 595, getHeight: () => 842 } }
    addImage = vi.fn()
    addPage = vi.fn()
    save = vi.fn()
  },
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

import { useResultExports } from './useResultExports'

const params = {
  selectedModel: null,
  selectedGPU: null,
  quantization: 'fp16' as const,
  sequenceLength: 4096,
  batchSize: 1,
  numGPUs: 1,
  numNodes: 1,
  result: null,
}

function mountSection() {
  document.body.insertAdjacentHTML(
    'beforeend',
    '<div id="calculator-section"><details id="advanced"><summary>A</summary></details><details id="details" open><summary>D</summary></details></div>',
  )
}
const isOpen = (id: string) => (document.getElementById(id) as HTMLDetailsElement).open

describe('useResultExports: PDF', () => {
  afterEach(() => {
    document.getElementById('calculator-section')?.remove()
    capture.mockReset()
  })

  it('expands every <details> for the capture, then restores each one exactly', async () => {
    mountSection()
    let openAtCapture: boolean[] = []
    capture.mockImplementation(async (el: HTMLElement) => {
      openAtCapture = Array.from(el.querySelectorAll('details')).map((d) => d.open)
      return { width: 100, height: 100, toDataURL: () => 'data:image/jpeg;base64,' }
    })
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(openAtCapture).toEqual([true, true])
    expect(isOpen('advanced')).toBe(false)
    expect(isOpen('details')).toBe(true)
  })

  it('restores the details when the capture fails', async () => {
    mountSection()
    capture.mockRejectedValue(new Error('canvas failed'))
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(isOpen('advanced')).toBe(false)
    expect(isOpen('details')).toBe(true)
  })
})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run src/components/layout src/components/inputs src/hooks/useResultExports.test.tsx src/utils/exportPptx.test.ts`
Expected: FAIL: no `advanced-settings`/`verdict`/`result-details` test ids; `advancedChanges` missing; label still "GPUs per server"; PDF leaves `advanced` closed during capture; PPTX still has `Number of GPUs`. The Task 6a tests still pass.

- [ ] **Step 3: `countAdvancedChanges`**

Create `src/components/layout/advancedChanges.ts`:

```ts
import { DEFAULT_UI_CONFIG, type UIConfig } from '@store/uiStore'

/**
 * Advanced settings that differ from their default. The Advanced section opens
 * itself when this is above 0, so a collapsed section never hides an active
 * non-default setting (ADR 0005). Inputs hidden as inert in training don't count.
 */
export function countAdvancedChanges(config: UIConfig): number {
  const d = DEFAULT_UI_CONFIG
  if (config.mode === 'training') return config.batchSize !== d.batchSize ? 1 : 0
  return [
    config.batchSize !== d.batchSize,
    config.kvQuantization !== d.kvQuantization,
    config.shardingStrategy !== d.shardingStrategy,
    config.numNodes > 1 && config.interNodeFabric !== d.interNodeFabric,
    config.interconnectOverride !== d.interconnectOverride,
    config.offloadingEnabled !== d.offloadingEnabled,
    config.kvTier.tier !== d.kvTier.tier,
  ].filter(Boolean).length
}
```

- [ ] **Step 4: InputPanel layout**

Replace the body of `src/components/layout/InputPanel.tsx` (keep the existing component imports, add the new ones in the same edit) with:

```tsx
import { BatchSizeInput } from '@components/inputs/BatchSizeInput'
import { ConcurrentUsersInput } from '@components/inputs/ConcurrentUsersInput'
import { GPUCountSelector } from '@components/inputs/GPUCountSelector'
import { GPUSelector } from '@components/inputs/GPUSelector'
import { InterconnectSelector } from '@components/inputs/InterconnectSelector'
import { InterNodeFabricSelector } from '@components/inputs/InterNodeFabricSelector'
import { KVQuantizationPicker } from '@components/inputs/KVQuantizationPicker'
import { KVTierPanel } from '@components/inputs/KVTierPanel'
import { ModelSelector } from '@components/inputs/ModelSelector'
import { ModeToggle } from '@components/inputs/ModeToggle'
import { NodeCountSelector } from '@components/inputs/NodeCountSelector'
import { OffloadingPanel } from '@components/inputs/OffloadingPanel'
import { QuantizationPicker } from '@components/inputs/QuantizationPicker'
import { SequenceLengthInput } from '@components/inputs/SequenceLengthInput'
import { ShardingStrategySelector } from '@components/inputs/ShardingStrategySelector'
import { TrainingPanel } from '@components/inputs/TrainingPanel'
import { FRAMEWORK_PRESETS } from '@engines/frameworks'
import { useUIStore } from '@store/uiStore'
import { useEffect, useState } from 'react'
import { countAdvancedChanges } from './advancedChanges'

/**
 * Input panel: essential inputs always visible, advanced ones in a native <details>
 * that opens itself whenever one of them differs from its default (ADR 0005).
 * Inputs that are inert in training are hidden, never reset (spec Section 1).
 */
export function InputPanel() {
  const state = useUIStore()
  const { selectedGPU, mode, frameworkPreset } = state
  const isInference = mode === 'inference'
  // In training, more than one GPU only matters with a DeepSpeed ZeRO preset
  const showGPUCount =
    selectedGPU !== null && (isInference || FRAMEWORK_PRESETS[frameworkPreset].zeroStage !== null)

  const changed = countAdvancedChanges(state)
  const [advancedOpen, setAdvancedOpen] = useState(changed > 0)
  useEffect(() => {
    if (changed > 0) setAdvancedOpen(true)
  }, [changed])

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="space-y-6">
        <ModeToggle />

        <hr className="border-gray-200 dark:border-gray-700" />

        <section aria-label="Essential settings" className="space-y-4">
          <ModelSelector />
          {isInference && <QuantizationPicker />}
          <GPUSelector />
          {showGPUCount && <GPUCountSelector />}
          {/* Multi-node is inference-only (multi-node training is a spec Non-Goal) */}
          {selectedGPU && isInference && <NodeCountSelector />}
          <SequenceLengthInput />
          {isInference && <ConcurrentUsersInput />}
        </section>

        {!isInference && (
          <>
            <hr className="border-gray-200 dark:border-gray-700" />
            <TrainingPanel />
          </>
        )}

        <hr className="border-gray-200 dark:border-gray-700" />

        <details
          data-testid="advanced-settings"
          open={advancedOpen}
          onToggle={(e) => setAdvancedOpen(e.currentTarget.open)}
        >
          <summary className="cursor-pointer text-lg font-semibold text-gray-900 dark:text-white">
            Advanced
            {changed > 0 && (
              <span className="ml-2 text-sm font-normal text-blue-600 dark:text-blue-400">
                {changed} setting{changed === 1 ? '' : 's'} changed
              </span>
            )}
          </summary>
          <div className="mt-4 space-y-4">
            <BatchSizeInput />
            {isInference && (
              <>
                <KVQuantizationPicker />
                {selectedGPU && <ShardingStrategySelector />}
                <InterNodeFabricSelector />
                <InterconnectSelector />
                {selectedGPU && <OffloadingPanel />}
                {selectedGPU && <KVTierPanel />}
              </>
            )}
          </div>
        </details>
      </div>
    </div>
  )
}
```

In `src/components/inputs/GPUSelector.tsx` delete line 15 (`import { InterconnectSelector } ...`) and lines 411-412 (the comment and `<InterconnectSelector />`) in one edit.

- [ ] **Step 5: Strategy selector visibility, GPU count label and summary**

`src/components/inputs/ShardingStrategySelector.tsx`: replace lines 23-26 with:

```tsx
  // Visible whenever the part forms multi-GPU servers (not only at numGPUs > 1), so
  // pipeline parallel is reachable before R14 snaps a tensor-parallel degree.
  if (maxGPUsFor(selectedGPU) <= 1) {
    return null
  }
```

(import `maxGPUsFor` from `@utils/gpuLimits` in the same edit), and wrap the interconnect badge `<div>` (lines 150-168) and the `tpExceedsMax` warning (lines 171-179) in `{numGPUs > 1 && ( <> ... </> )}`.

`src/components/inputs/GPUCountSelector.tsx`: add at module level

```tsx
const STRATEGY_LABELS: Record<ShardingStrategy, string> = {
  'tensor-parallel': 'tensor parallel',
  'pipeline-parallel': 'pipeline parallel',
  'expert-parallel': 'expert parallel',
}
```

(import `type ShardingStrategy` from `@engines/types` in the same edit); add `const numNodes = useUIStore((s) => s.numNodes)`; replace lines 29-33 with:

```tsx
  const label = isTraining ? 'Number of GPUs' : 'GPUs per replica (in one server)'

  const tooltip = isTraining
    ? 'GPUs used for data-parallel training, e.g. DeepSpeed ZeRO sharding. Multi-node training is not modelled, so this is the total GPU count.'
    : `The parallel degree of one model replica inside one server. Tensor, pipeline or expert parallelism runs at this level, over NVLink, Infinity Fabric or PCIe. Capped at ${maxGPUs} — the largest GPU count this hardware forms in one node.`
```

and replace the summary paragraph (lines 78-86) with:

```tsx
      {(numGPUs > 1 || (!isTraining && numNodes > 1)) && (
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {isTraining
            ? `${numGPUs} GPUs`
            : numNodes > 1
              ? `${numGPUs} GPUs per server × ${numNodes} servers per replica, ${STRATEGY_LABELS[shardingStrategy]}`
              : `${numGPUs} GPUs per replica (in one server), ${STRATEGY_LABELS[shardingStrategy]}`}
        </p>
      )}
```

- [ ] **Step 6: Verdict block and collapsed details**

Create `src/components/outputs/VerdictBlock.tsx`:

```tsx
import type { PerformanceEstimate } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import { firstTokenLabel } from '@utils/perfLabels'
import type { ReactNode } from 'react'

interface VerdictBlockProps {
  /** The fit gauge (FitIndicator) for the displayed breakdown */
  fit: ReactNode
  performance: PerformanceEstimate
  batchSize: number
  maxSessions: number | null
  sequenceLength: number
}

/** The answer first: does it fit, how fast, first-token delay, sessions (ADR 0005) */
export function VerdictBlock({
  fit,
  performance,
  batchSize,
  maxSessions,
  sequenceLength,
}: VerdictBlockProps) {
  return (
    <section data-testid="verdict" aria-label="Verdict" className="space-y-4">
      {fit}
      <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">Decode speed</dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {performance.tokensPerSecond.toFixed(1)} tokens/sec
          </dd>
        </div>
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">{firstTokenLabel(batchSize)}</dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {formatDuration(performance.timeToFirstToken)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            Max sessions at {sequenceLength.toLocaleString('en-US')} tokens
          </dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {maxSessions === null ? 'n/a' : maxSessions.toLocaleString('en-US')}
          </dd>
        </div>
      </dl>
    </section>
  )
}
```

In `src/components/layout/ResultsPanel.tsx`:
- change line 22 to `import { useEffect, useMemo, useRef, useState } from 'react'` and add `import { VerdictBlock } from '@components/outputs/VerdictBlock'` in the same edit as its use below;
- add `const [detailsOpen, setDetailsOpen] = useState(false)` right after `const resultsDivRef = useRef<HTMLDivElement>(null)` (before any early return: it survives the loading skeleton);
- replace the inference `return (...)` (lines 454-658) with:

```tsx
  const fit = result.multiGPU ? (
    <FitIndicator
      totalVRAM={result.multiGPU.totalPerGPU}
      availableVRAM={selectedGPU.vram_gb}
      numGPUs={result.multiGPU.numGPUs}
    />
  ) : (
    <FitIndicator totalVRAM={displayBreakdown.total} availableVRAM={selectedGPU.vram_gb} />
  )

  return (
    <div
      ref={resultsDivRef}
      className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">VRAM Requirements</h2>
          {/* KEEP: the existing <div className="flex items-center gap-2"> holding the
              Export PDF, Export PPTX and Save to Compare buttons, unchanged. */}
        </div>

        <VerdictBlock
          fit={fit}
          performance={result.performance}
          batchSize={batchSize}
          maxSessions={maxSessions}
          sequenceLength={sequenceLength}
        />

        {/* Warnings stay visible: they change the answer */}
        {/* KEEP: the {/* Host capacity exceeded */} block and the {/* Interconnect Warning */}
            block, unchanged, then: */}
        <SoftWarnings
          warnings={softWarnings({ mode, numGPUs, numNodes, shardingStrategy }, selectedModel, selectedGPU)}
        />
        {/* KEEP: the {/* Recommendations: ... */} block ({deviceDoesNotFit && <Recommendations ... />}), unchanged */}

        <details
          data-testid="result-details"
          open={detailsOpen}
          onToggle={(e) => setDetailsOpen(e.currentTarget.open)}
        >
          <summary className="cursor-pointer text-lg font-semibold text-gray-900 dark:text-white">
            Details
          </summary>
          <div className="mt-4 space-y-6">
            {/* MOVE here unchanged, in this order: the {/* Offloading Summary */} block;
                the {/* Single-GPU Breakdown Chart and Table */} block with the weight-source
                <p> after it; the {/* Multi-GPU Breakdown Chart */} block; the
                {/* Performance Section */} <div> (heading, PerformanceSection, CapacitySection);
                the "GB here means GiB" footnote <p>. */}
          </div>
        </details>
      </div>
    </div>
  )
```

The `KEEP`/`MOVE` comments above are instructions for this edit, not code to leave in the file: each names an existing block by the JSX comment that already labels it in `ResultsPanel.tsx`; paste that block in its place verbatim, then delete the instruction comment. The old standalone FitIndicator block (lines 519-531) is replaced by `fit`; the earlier `<SoftWarnings ... />` added in Task 5 moves into the warnings group as shown (keep exactly one).

- [ ] **Step 7: PDF expands every `<details>` before capture**

In `src/hooks/useResultExports.ts`, replace lines 58-77 (from `// Force light mode` through the inner `finally`) with:

```ts
      // Force light mode so the capture uses correct contrast
      const root = document.documentElement
      const wasDark = root.classList.contains('dark')
      if (wasDark) root.classList.remove('dark')

      // Collapsed sections (Advanced, Details) would be missing from the PDF: open every
      // <details> for the capture, then put each back exactly as the user left it.
      const sections = Array.from(captureEl.querySelectorAll('details'))
      const wasOpen = sections.map((d) => d.open)
      for (const d of sections) d.open = true

      let canvas: HTMLCanvasElement
      try {
        canvas = await html2canvasPro(captureEl, {
          scale: 2,
          useCORS: true,
          backgroundColor: '#f9fafb', // gray-50 — matches page background
          logging: false,
          // windowWidth: 800 tells the renderer to use the mobile breakpoint:
          // lg:grid-cols-12 collapses to grid-cols-1, panels stack vertically.
          // This is a render-time option — the live app and mobile users are unaffected.
          windowWidth: 800,
        })
      } finally {
        if (wasDark) root.classList.add('dark')
        sections.forEach((d, i) => {
          d.open = wasOpen[i] ?? false
        })
      }
```

- [ ] **Step 8: PPTX "GPUs per replica" row**

In `src/utils/exportPptx.ts` replace line 182 (`['Number of GPUs', String(numGPUs)],`) with:

```ts
    [
      'GPUs per replica',
      numNodes > 1
        ? `${numGPUs / numNodes} per server × ${numNodes} servers (${numGPUs} total)`
        : `${numGPUs} (in one server)`,
    ],
```

- [ ] **Step 9: Run the tests**

Run: `npx vitest run`
Expected: PASS, including every Task 6a composition test.

- [ ] **Step 10: Build, lint, commit**

Run: `npm run build && npm run lint:fix && npm run lint`
Expected: no errors.

```bash
rtk git add src/components src/hooks/useResultExports.ts src/hooks/useResultExports.test.tsx src/utils/exportPptx.ts src/utils/exportPptx.test.ts
rtk git commit -m "feat(ui): essential/advanced inputs, verdict block, collapsed details, GPUs per replica

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---
### Task 7: Docs (guide, CHANGELOG, CLAUDE.md, ARCHITECTURE.md, README)

No version bump: `package.json` stays at 1.13.0 (release PR later). ADR 0006 is already `Accepted` (no edit needed).

**Files:**
- Modify: `src/components/guide/GuidePage.tsx:3-16` (SECTIONS), `:166-219` (GPU Selection, Hardware), `:221-261` (Offloading), `:373-460` (Results), `:471-478` (URL Sharing), new sections
- Modify: `CHANGELOG.md:7` (new `## [2.0.0]` entry above `## [1.13.0]`)
- Modify: `CLAUDE.md:47-57` (Key Patterns bullets)
- Modify: `ARCHITECTURE.md:31-40`, `:227-250`, `:263-283`, `:333-357`
- Modify: `README.md:25-29`, `:45`
- Outside the repo (not committed): `~/.claude/projects/-Users-fjacquet-Projects-llmvram/memory/MEMORY.md` "Current counts" line

**Interfaces:** none (documentation only).

- [ ] **Step 1: Guide sections**

In `src/components/guide/GuidePage.tsx`, change `SECTIONS` to insert `{ id: 'whats-new', label: "What's new in 2.0" },` after `quick-start` and `{ id: 'why-changed', label: 'Why a setting changed' },` after `results`.

Insert after the Quick Start section (before `{/* Mode Toggle */}` / `<SectionHeading id="mode-toggle">`):

```tsx
        {/* What's new in 2.0 */}
        <SectionHeading id="whats-new">What&apos;s new in 2.0</SectionHeading>
        <P>
          <strong>Layout.</strong> The essential inputs stay visible: mode, model, format, GPU,
          GPUs per replica, servers, context and concurrent users. Everything else sits under{' '}
          <strong>Advanced</strong>, which opens by itself and shows &quot;N settings changed&quot;
          whenever one of its values differs from the default. Results lead with a verdict (does it
          fit, decode speed, first-token delay, sessions); charts and tables are under{' '}
          <strong>Details</strong>. The PDF export opens every section before it captures.
        </P>
        <P>
          <strong>Rules.</strong> A combination that cannot run is no longer offered, and a change
          that makes another setting impossible corrects that setting and says why in one notice
          (&quot;Adjusted for …&quot;). A shared link that encodes an impossible combination opens
          corrected, with a &quot;Shared link adjusted&quot; notice.
        </P>
        <P>
          <strong>GPU data, per card.</strong> H100 PCIe and A100 PCIe pair cards over a 2-card
          NVLink bridge (600 GB/s); larger groups cross PCIe 5 / PCIe 4, and H100 PCIe FP16 is the
          dense 756 TFLOPS. A100 SXM uses NVLink 3 (600 GB/s). The H200 entry is the SXM product
          (HGX 4 or 8); the new H200 NVL entry is the PCIe card with a 4-way bridge at 900 GB/s. DGX
          Spark (GB10) is one GPU per unit: cluster Sparks as servers over 200 GbE. DGX Station is a
          single GPU. L40S and RTX 6000 Ada are PCIe 4. Apple Silicon and GB10 are marked unified
          memory.
        </P>
        <P>
          <strong>Multi-server.</strong> Prefill and decode across servers are now computed from the
          bytes crossing the network, not from an efficiency curve. Small clusters get much faster
          first tokens; a single request over many HGX servers gets slower (it walks the servers one
          after another).
        </P>
        <P>
          <strong>Existing links.</strong> They open with the corrected values and a notice. Numbers
          change for the cards above and for most multi-server setups. An old H200 link opens as
          H200 SXM.
        </P>
```

Insert after the Results Panel section (before `{/* Comparison View */}`):

```tsx
        {/* Why a setting changed */}
        <SectionHeading id="why-changed">Why a setting changed</SectionHeading>
        <P>
          When a change makes another setting impossible, the calculator corrects it and shows one
          notice listing every correction. The rules:
        </P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>GPU count above what the part forms in one server</strong> is capped (8 on an HGX
            baseboard, 72 on an NVL72 rack, 1 on a DGX Spark or a Mac).
          </li>
          <li>
            <strong>Tensor-parallel degree vLLM cannot run</strong>: the GPU count must divide the
            model&apos;s attention heads (and match its KV heads). Llama 3.1 70B runs on 1, 2, 4 or 8
            GPUs; asking for 6 sets 4. Choose pipeline parallel to use 6.
          </li>
          <li>
            <strong>Expert parallel on a dense model</strong> switches to tensor parallel.
          </li>
          <li>
            <strong>Unified memory</strong> (Apple Silicon, DGX Spark): RAM is the GPU&apos;s memory,
            so CPU-RAM offload, host-memory KV tiers and CPU optimizer offload are turned off. NVMe
            offload stays available.
          </li>
          <li>
            <strong>Grace host KV tier</strong> needs a Grace host (GB300 NVL72, DGX Station).
          </li>
          <li>
            <strong>KV cache offload</strong> already keeps all KV off the GPU, so a KV tier is turned
            off.
          </li>
          <li>
            <strong>Interconnect variant</strong> not offered on the selected GPU resets to its
            default.
          </li>
          <li>
            <strong>vLLM and TGI</strong> are inference engines: switching to fine-tuning clears them.
            CPU optimizer offload needs a DeepSpeed ZeRO preset.
          </li>
          <li>
            <strong>Out-of-range numbers</strong> (offloaded layers beyond the model, batch 0, context
            below 512, more than 8 servers) are brought into range.
          </li>
        </ul>
        <P>
          Warnings that keep your value: context beyond the model&apos;s native length, a
          tensor-parallel degree above what the interconnect scales to, small clusters of single-GPU
          units, experts that do not split evenly across the GPUs, more pipeline stages than layers.
        </P>
```

Replace the GPU Selection paragraph about interconnects (line 173-175, the "Key specs" `<P>`) with:

```tsx
        <P>
          <strong>Key specs that affect calculations:</strong> VRAM determines fit/no-fit. Memory
          bandwidth determines decode speed. FP16 TFLOPS (dense) determines prefill. The scale-up
          link sets multi-GPU cost: NVSwitch baseboards connect 4 or 8 GPUs (HGX), an NVL72 rack up
          to 72; NVLink bridges pair 2 PCIe cards (H100/A100 PCIe) or up to 4 (H200 NVL), and a
          larger group crosses PCIe. Unified-memory parts (Apple Silicon, DGX Spark) have no
          separate host memory.
        </P>
```

Replace the Hardware Configuration `Number of GPUs` subsection (lines 185-191) with:

```tsx
        <SubHeading>GPUs per replica (in one server)</SubHeading>
        <P>
          The parallel degree of one model replica inside one server, capped by what the part forms
          in one node. Under tensor parallelism only degrees vLLM accepts are offered (they divide
          the attention heads; most models allow 1, 2, 4 and 8). With several servers the summary
          reads &quot;8 GPUs per server × 2 servers per replica&quot;. An 8-GPU server running four
          2-GPU replicas is sized as one 2-GPU replica.
        </P>
```

and change `<P>Visible when using 2+ GPUs. Two options:</P>` (line 194) to `<P>Visible whenever the GPU forms multi-GPU servers, so pipeline parallel can be chosen before picking a count tensor parallel cannot run. Options:</P>`. In the interconnect-badge paragraph (lines 213-218) add after the first sentence: `A bridged card shows &quot;NVLink bridge&quot; only while the group fits the bridge; beyond it the badge and the maths both use PCIe.`

In the Offloading section, after the Offload Target list add:

```tsx
        <P>
          On unified-memory parts (Apple Silicon, DGX Spark) only NVMe is offered: RAM is already the
          GPU&apos;s memory, so there is nothing to offload into.
        </P>
```

In the Results Panel section (after its first paragraph at line 373-376) add:

```tsx
        <SubHeading>Verdict and details</SubHeading>
        <P>
          The verdict block answers first: whether the configuration fits, decode speed, the
          first-token figure and how many sessions fit. Warnings stay visible under it. Charts, the
          memory table, the multi-GPU split, the weight source, the KV tier summary and per-user
          metrics are under Details. Export PDF opens every collapsed section for the capture.
        </P>
        <SubHeading>First token with a batch (burst)</SubHeading>
        <P>
          At batch size 1 the figure is the time to the first token. At batch size B above 1 it is
          labelled &quot;Prefill per request (amortized over batch B)&quot;: the prefill of a burst of
          B prompts divided by B. In a real burst the first request answers sooner and the last one
          waits about B times longer. Across servers the prompt also crosses the network once per
          server boundary.
        </P>
```

Replace the URL Sharing paragraph (lines 472-477) with:

```tsx
        <P>
          Every configuration change is encoded into the URL hash using LZ-String compression. Click
          the link icon in the header to copy the current URL. A link restores the whole
          configuration in one step, including the fine-tuning settings (gradient accumulation,
          gradient checkpointing, Flash Attention, framework preset, CPU optimizer offload) and the
          interconnect variant. If the link encodes a combination that is no longer allowed (or a GPU
          whose data was corrected), it opens corrected with a &quot;Shared link adjusted&quot; notice
          explaining what changed.
        </P>
```

- [ ] **Step 2: CHANGELOG**

Insert above `## [1.13.0] - 2026-09-27` in `CHANGELOG.md`:

```markdown
## [2.0.0] - Unreleased

### Breaking changes

- Layout: essential inputs stay visible; batch, KV precision, strategy, fabric, interconnect variant, offloading and KV tier move under "Advanced" (opens itself when any differs from its default). Results lead with a verdict; charts and tables are under "Details". The PDF export expands everything before capture.
- Labels: "GPUs per server" is now "GPUs per replica (in one server)" (UI and the PPTX "GPUs per replica" row, which replaces "Number of GPUs"); "Time to first token" reads "Prefill per request (amortized over batch B)" when the batch is above 1.
- Shared links open corrected, with a "Shared link adjusted" notice, when they encode a combination that is no longer allowed (expert parallel on a dense model, tensor-parallel degrees vLLM refuses, Grace or host tiers on GPUs without them, CPU-RAM offload on unified memory, inference presets in fine-tuning, out-of-range numbers). Links now also restore gradient accumulation, gradient checkpointing, Flash Attention, the framework preset, CPU optimizer offload and the interconnect variant.
- GPU data corrected per card (numbers change): H100 PCIe (PCIe 5 plus a 2-card NVLink bridge at 600 GB/s; FP16 756 dense, was the 1,513 sparse figure; TP-8 decode up to -26%), A100 PCIe (PCIe 4 plus a 2-card bridge; up to -25%), A100 SXM (NVLink 3 at 600 GB/s, was priced at 900), GB10 DGX Spark (1 GPU per node, no scale-up link: 2 Sparks are 2 servers), DGX Station (single GPU), L40S and RTX 6000 Ada (PCIe 4; TTFT about +20%).
- `nvidia-h200-141gb` is now H200 SXM only. H200 NVL is a new entry, `nvidia-h200-nvl-141gb`. A link that meant the NVL card opens as SXM; this cannot be detected.
- Multi-node TTFT and decode are computed from bytes over the fabric (ADR 0007): 2x GB10 Llama 3.1 70B 8k 195 s -> 13.3 s; HGX H100x8 70B batch 32 over two 100G servers 0.47 s -> 0.20 s; a single request over 2+ HGX servers is 16-115% slower (the old cross-node prefill speedup does not exist in vLLM). Decode moves under 10%.

### Added

- One configuration rule table (`src/engines/config-rules.ts`): impossible combinations are not offered, and every correction is shown in one notice per action.
- H200 NVL (PCIe, 4-way NVLink bridge at 900 GB/s). GPU data fields `nvlink_bridge` and `unified_memory`; NVLink 3 (600 GB/s) interconnect.
- 200GbE fabric preset (ConnectX-7) for DGX Spark / DGX Station clusters.
- Soft warnings for small single-GPU clusters, uneven expert splits and more pipeline stages than layers.

### Changed

- TGI is labelled "TGI (archived)"; existing links keep it.
```

- [ ] **Step 3: CLAUDE.md**

In `CLAUDE.md` Key Patterns:

Replace the bullet `- **`numGPUs` means GPUs PER NODE**: ...` with:

```markdown
- **`numGPUs` is the parallel degree of one replica inside one server** (UI label "GPUs per replica (in one server)"): total GPUs is `numGPUs × numNodes`, available as `MultiGPUVRAMBreakdown.numGPUs`. Any consumer displaying a GPU count must use the total, not the store field.
```

Replace the bullet `- **GPU count clamping is silent**: ...` with:

```markdown
- **Every correction is shown**: config-rules R1 clamps `numGPUs` to `max_gpus_per_node`, R14 snaps tensor-parallel degrees vLLM refuses (heads % t, KV-head rule), and every other hard rule corrects likewise; each action yields one toast ("Adjusted for …"), a restored link "Shared link adjusted". `clampGPUCount` remains only for Recommendations.
```

Add after the `max_gpus_per_node` bullet:

```markdown
- **Configuration rules live in one table**: `src/engines/config-rules.ts` (`RULES`, `normalizeConfig`, `allowedOptions`, `softWarnings`). Every store action goes through `commit()` in `uiStore.ts`, which normalizes the whole config once and writes `pendingNotice` (toasted by `useConfigNotices`); URL restore builds the whole config (`urlStateToConfig`) and calls `restoreConfig` once, with `fp` taken raw. Rules are mode-gated: a field hidden as inert is never corrected. A new input must declare its rules in the table, and inputs offer only `allowedOptions` values.
- **`nvlink_bridge` and `unified_memory` are GPU data**: a bridge carries a TP/EP group only while `groupSize <= nvlink_bridge.size`, so `resolveInterconnect(gpu, groupSize)` takes the group size and every consumer passes its own (TP/EP: GPUs per stage; badge and W2: `numGPUs`). `unified_memory` is the only source for "no separate host memory" (R6): never infer it from `interconnect === 'unified'` or the tier.
```

Append to the `**Decode = bytes per step / bandwidth + communication**` bullet: ` Across servers each step adds (N − 1)/stages stage hops of B tokens (`fabricHopSeconds`, eta 0.8 HGX assumption / 0.37 GB10); prefill pipelines M = ceil(B·T/C) microbatches (C = vLLM max_num_batched_tokens) and adds (N − 1) prompt hops divided by B.`

- [ ] **Step 4: ARCHITECTURE.md**

In the Directory Structure `engines/` block add, after `offloading.ts`:

```
│   ├── config-rules.ts         # Rule table: normalizeConfig, allowedOptions, soft warnings, notices
│   ├── fabric.ts               # Scale-out fabric presets, stage-hop time, prefill microbatch fill
│   ├── multi-node.ts           # Pipeline stages across servers over multi-gpu.ts
```

(skip any line already present). In `### Multi-GPU Engine`, after the first paragraph add: `The interconnect is resolved per group size: an NVLink bridge (\`nvlink_bridge\`) carries a group only while it fits, larger groups use the card's PCIe link (\`resolveInterconnect(gpu, groupSize)\`).` After the Pipeline Parallelism list add:

```markdown
**Multi-node (`multi-node.ts`, `fabric.ts`):** nodes are pipeline stages. A stage hop costs `tokens × hidden × 2 × 2 B / (eta × port × GPUs per node) + 10 µs`; decode adds `(N − 1)/stages` hops of B tokens per step, prefill pipelines `M = ceil(B × T / C)` microbatches (speedup `N × M / (M + N − 1)`) plus `(N − 1)` prompt hops, both amortized per request over B.
```

Replace the `### Zustand Store (uiStore.ts)` section body with:

```markdown
Single store with all calculator state (`UIConfig`, defaults in `DEFAULT_UI_CONFIG`). Every configuration action goes through one `commit()`: merge the patch, run `normalizeConfig` from `config-rules.ts` on the whole config (mode-gated rules, fixpoint in at most 4 passes), and write `pendingNotice` when something was corrected. `useConfigNotices` (mounted in `App.tsx`) toasts it once per action and clears it. Action intents that are not rules: picking vLLM/TGI switches to inference; enabling offloading on unified memory picks NVMe. Only the dark-mode preference is persisted to localStorage; everything else lives in the URL hash.
```

In `### URL Persistence` add: `- Restore: \`urlStateToConfig\` maps the parsed link to a whole config (framework preset taken raw, training keys ga/gc/fa/co and interconnect override \`io\` included) and the store applies it with one \`restoreConfig\` call, normalized once ("Shared link adjusted"). Numeric keys are not range-checked by the schema: R1/R10 correct them.`

In the Component Architecture mermaid: replace the `Left` subgraph body with `Essential["Essential: ModelSelector, QuantizationPicker, GPUSelector, GPUCountSelector, NodeCountSelector, SequenceLengthInput, ConcurrentUsersInput"]` and `Advanced["Advanced (details): BatchSizeInput, KVQuantizationPicker, ShardingStrategySelector, InterNodeFabricSelector, InterconnectSelector, OffloadingPanel, KVTierPanel"]`, and the `Right` subgraph body with `Verdict["VerdictBlock (FitIndicator, decode, first token, sessions)"]`, `Warnings["Warnings + SoftWarnings + Recommendations"]`, `Details["Details (details): charts, tables, MultiGPUBreakdownChart, PerformanceSection"]`.

- [ ] **Step 5: README**

`README.md:29` (Interconnect Selector bullet) becomes `- **Configuration rules**: impossible combinations are not offered; any correction (including on a shared link) is explained in one notice`. Line 45: add `H200 NVL` after `H200` (`H100 PCIe/SXM, H200 SXM/NVL, B200, ...`). In the Multi-GPU bullet (line 25) change `(NVLink-5/4, Infinity Fabric, PCIe-5/4)` to `(NVLink 5/4/3 and NVLink bridges, Infinity Fabric, PCIe 5/4)`.

- [ ] **Step 6: Auto-memory count (outside the repo, not committed)**

If the executing agent can write `~/.claude/projects/-Users-fjacquet-Projects-llmvram/memory/MEMORY.md`, change the "Current counts" line to start `- Current counts: 54 models, 28 GPUs (2026-09-27: H200 NVL split from H200 SXM.` keeping the rest of the line. Otherwise mention it in the PR description for the user.

- [ ] **Step 7: Verify and commit**

Run: `npx vitest run && npm run build && npm run lint:fix && npm run lint`
Expected: all green (docs-only changes plus GuidePage JSX).

```bash
rtk git add src/components/guide/GuidePage.tsx CHANGELOG.md CLAUDE.md ARCHITECTURE.md README.md
rtk git commit -m "docs: v2.0 guide, CHANGELOG breaking changes, CLAUDE.md and ARCHITECTURE.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z"
```

---

## Spec ambiguities resolved in this plan

- **eta per GPU, not per fabric.** The spec says `classFactor` becomes `effectiveFraction`, but eta depends on the node (GB10 lacks GPUDirect RDMA). `classFactor` is dropped from `FabricSpec`; `effectiveFraction(gpu)` returns 0.37 for `nvidia-gb10` (the only measured case), 0.8 elsewhere.
- **Hostile links must parse.** `sl` (min 512), `nn` (1-8) and `cu` (int, max) made the whole schema fail, so "sl < 512 opens corrected" was impossible. The schema now accepts any number and R1/R10 correct after restore. `cu` keeps its `MAX_CONCURRENT_USERS` upper bound inside R10.
- **R12 only when offloading is on.** A stale `kvCacheOffload` with offloading disabled is inert (the engine ignores it), so it does not clear a KV tier.
- **Enabling offloading on unified memory** is action intent (target switches to NVMe), otherwise R6 would switch the checkbox straight back off.
- **R4 host capacity notice** gets its own text ("Host capacity reset to the default: it must be a positive number.") since the spec's R4 text names only the KV tier.
- **Notice lines**: one per rule and field, carrying that pair's last message (a fixpoint can snap R14 twice). Titles for plain setting changes follow the first correction's subject.
- **First-token label** stays "Time to first token" at batch 1; the amortized label applies for B > 1 (the figure is exactly one request's TTFT at B = 1).
- **PPTX row**: "Number of GPUs" held the cluster total; it becomes "GPUs per replica" with `8 per server × 4 servers (32 total)` so the total is not lost.
- **Interconnect override** replaces the bridge too (the user picked the link). After Task 3a no database GPU has `interconnect_options`, so R5/R13 and the variant picker are reachable only through custom rows; kept per spec.
- **Derived fixtures before the data lands**: Tasks 1a/2a use `validateGPU({ ...findGPU('apple-m3-ultra'), unified_memory: true })`; Task 3a swaps it for the real row. R5 keeps one schema-validated derived row with options.
- **Order independence** is tested on the pure function (every rotation of the rule table, forward and reversed) and on restore (link keys in three orders). Interactive action sequences are path-dependent by design (R14 snaps immediately), which is why a link restores in one step.
- **H200 SXM display name** becomes "NVIDIA H200 141GB SXM" (id unchanged) to tell it apart from the NVL row.
- **GPU count summary** keeps the spec's `"{n} GPUs per server × {m} servers per replica"` and appends `, {strategy}` (the EP label fix needs the strategy in the summary); Task 6b pins the suffixed form.
- **R14 is bounded by the GPU** (`min(numGPUs, max_gpus_per_node)`) so a hostile count cannot loop in any rule order; with that bound R14 fires at most once, so the notice de-duplication is tested with hand-built corrections.
- **Every setter goes through `commit()`**, including fields with no rule (quantization, fabric...): harmless, and it makes "every action normalizes" literally true. Batch size and gradient accumulation are range sliders (no NaN on edit); number fields commit on blur, so routine typing does not produce correction toasts.
