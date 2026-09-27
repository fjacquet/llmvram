import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { MAX_CONCURRENT_USERS } from '@engines/constants'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { type GPU, type Model, validateGPU, validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import {
  allowedOptions,
  buildNotice,
  type Correction,
  DEFAULT_RULE_CONFIG,
  isValidTPDegree,
  MAX_NORMALIZE_PASSES,
  normalizeConfig,
  RULES,
  type RuleConfig,
  type RuleId,
  softWarnings,
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
const UNIFIED_M3 = M3 // unified_memory comes from gpus.json since Task 3a
// Derived row, validated by the real schema. After Task 3a no database GPU
// carries interconnect_options, so R5 is only reachable through a row like this one.
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
  {
    rule: 'R1',
    path: 'field',
    name: '9 on an 8-GPU board',
    config: { numGPUs: 9 },
    model: L70,
    gpu: H100,
    expected: { numGPUs: 8 },
  },
  {
    rule: 'R1',
    path: 'dependency',
    name: '8 then a 1-GPU part',
    config: { numGPUs: 8 },
    model: L70,
    gpu: M3,
    validUnder: { model: L70, gpu: H100 },
    expected: { numGPUs: 1 },
  },
  {
    rule: 'R1',
    path: 'link',
    name: 'negative count',
    config: { numGPUs: -3 },
    model: L70,
    gpu: H100,
    expected: { numGPUs: 1 },
  },
  // R2: expert-parallel only for a splittable MoE
  {
    rule: 'R2',
    path: 'field',
    name: 'EP on a dense model',
    config: { shardingStrategy: 'expert-parallel', numGPUs: 8 },
    model: L70,
    gpu: H100,
    expected: { shardingStrategy: 'tensor-parallel' },
  },
  {
    rule: 'R2',
    path: 'dependency',
    name: 'EP then a dense model',
    config: { shardingStrategy: 'expert-parallel', numGPUs: 8 },
    model: L70,
    gpu: H100,
    validUnder: { model: DSR1, gpu: H100 },
    expected: { shardingStrategy: 'tensor-parallel' },
  },
  {
    rule: 'R2',
    path: 'link',
    name: 'dense + EP at 6 cascades into R14',
    config: { shardingStrategy: 'expert-parallel', numGPUs: 6 },
    model: L70,
    gpu: H100,
    expected: { shardingStrategy: 'tensor-parallel', numGPUs: 4 },
  },
  // R3: host-grace only on a Grace host
  {
    rule: 'R3',
    path: 'field',
    name: 'host-grace on HGX',
    config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' } },
    model: L70,
    gpu: H100,
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } },
  },
  {
    rule: 'R3',
    path: 'dependency',
    name: 'host-grace then HGX',
    config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' } },
    model: L70,
    gpu: H100,
    validUnder: { model: L70, gpu: NVL72 },
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } },
  },
  {
    rule: 'R3',
    path: 'link',
    name: 'host-grace with a 500% active share',
    config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 5 } },
    model: L70,
    gpu: H100,
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none', activeShare: 1 } },
  },
  // R4: KV tier bounds, host capacity positive or null (no model/GPU dependency: live in both modes)
  {
    rule: 'R4',
    path: 'field',
    name: 'active share 0',
    config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network', activeShare: 0 } },
    model: L70,
    gpu: H100,
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network', activeShare: 0.01 } },
  },
  {
    rule: 'R4',
    path: 'dependency',
    name: 'live in inference, inert once training hides the field',
    config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network', burstSeconds: 0 } },
    model: L70,
    gpu: H100,
    validUnder: { model: L70, gpu: H100, mode: 'training' },
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network', burstSeconds: 1 } },
  },
  {
    rule: 'R4',
    path: 'link',
    name: 'host capacity 0',
    config: { offloadHostCapacityGB: 0 },
    model: L70,
    gpu: H100,
    expected: { offloadHostCapacityGB: null },
  },
  // R5: interconnect override must be one of the GPU's options
  {
    rule: 'R5',
    path: 'field',
    name: 'override on a GPU without options',
    config: { interconnectOverride: 'pcie-5' },
    model: L70,
    gpu: H100,
    expected: { interconnectOverride: null },
  },
  {
    rule: 'R5',
    path: 'dependency',
    name: 'valid override then another GPU',
    config: { interconnectOverride: 'pcie-5' },
    model: L70,
    gpu: H100,
    validUnder: { model: L70, gpu: WITH_OPTIONS },
    expected: { interconnectOverride: null },
  },
  {
    rule: 'R5',
    path: 'link',
    name: 'garbage override',
    config: { interconnectOverride: 'nvlink-9' },
    model: L70,
    gpu: WITH_OPTIONS,
    expected: { interconnectOverride: null },
  },
  // R6: unified memory has no separate host
  {
    rule: 'R6',
    path: 'field',
    name: 'cpu-ram offload on unified memory',
    config: { offloadingEnabled: true, offloadTarget: 'cpu-ram' },
    model: L8,
    gpu: UNIFIED_M3,
    expected: { offloadingEnabled: false },
  },
  {
    rule: 'R6',
    path: 'dependency',
    name: 'offload + PCIe tier then unified GPU',
    config: {
      offloadingEnabled: true,
      offloadTarget: 'cpu-ram',
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' },
    },
    model: L8,
    gpu: UNIFIED_M3,
    validUnder: { model: L8, gpu: H100 },
    expected: { offloadingEnabled: false, kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } },
  },
  {
    rule: 'R6',
    path: 'link',
    name: 'training optimizer offload on unified memory',
    config: { mode: 'training', frameworkPreset: 'deepspeed-zero3', cpuOffloadOptimizer: true },
    model: L8,
    gpu: UNIFIED_M3,
    expected: { cpuOffloadOptimizer: false },
  },
  {
    rule: 'R6',
    path: 'dependency',
    name: 'PCIe tier then DGX Spark',
    config: { kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' } },
    model: L8,
    gpu: findGPU('nvidia-gb10'),
    validUnder: { model: L8, gpu: H100 },
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } },
  },
  // R7: inference-only presets cleared in training
  {
    rule: 'R7',
    path: 'field',
    name: 'vLLM in training',
    config: { mode: 'training', frameworkPreset: 'vllm' },
    model: L70,
    gpu: H100,
    expected: { frameworkPreset: 'none' },
  },
  {
    rule: 'R7',
    path: 'dependency',
    name: 'vLLM then training mode',
    config: { mode: 'training', frameworkPreset: 'vllm' },
    model: L70,
    gpu: H100,
    validUnder: { model: L70, gpu: H100, mode: 'inference' },
    expected: { frameworkPreset: 'none' },
  },
  {
    rule: 'R7',
    path: 'link',
    name: 'TGI + optimizer offload cascades into R8',
    config: { mode: 'training', frameworkPreset: 'tgi', cpuOffloadOptimizer: true },
    model: L70,
    gpu: H100,
    expected: { frameworkPreset: 'none', cpuOffloadOptimizer: false },
  },
  // R8: CPU optimizer offload needs a ZeRO preset
  {
    rule: 'R8',
    path: 'field',
    name: 'offload with Unsloth',
    config: { mode: 'training', frameworkPreset: 'unsloth', cpuOffloadOptimizer: true },
    model: L70,
    gpu: H100,
    expected: { cpuOffloadOptimizer: false },
  },
  {
    rule: 'R8',
    path: 'dependency',
    name: 'ZeRO-2 offload then no preset',
    config: { mode: 'training', frameworkPreset: 'none', cpuOffloadOptimizer: true },
    model: L70,
    gpu: H100,
    expected: { cpuOffloadOptimizer: false },
  },
  {
    rule: 'R8',
    path: 'link',
    name: 'offload flag with no preset',
    config: { mode: 'training', cpuOffloadOptimizer: true },
    model: L70,
    gpu: H100,
    expected: { cpuOffloadOptimizer: false },
  },
  // R9: offload amounts bounded by the model
  {
    rule: 'R9',
    path: 'field',
    name: '500 layers on an 80-layer model',
    config: { offloadLayers: 500 },
    model: L70,
    gpu: H100,
    expected: { offloadLayers: 80 },
  },
  {
    rule: 'R9',
    path: 'dependency',
    name: '60 layers then a 32-layer model',
    config: { offloadLayers: 60 },
    model: L8,
    gpu: H100,
    validUnder: { model: L70, gpu: H100 },
    expected: { offloadLayers: 32 },
  },
  {
    rule: 'R9',
    path: 'link',
    name: '150% and -2 layers',
    config: { offloadPercentage: 150, offloadLayers: -2 },
    model: L70,
    gpu: H100,
    expected: { offloadPercentage: 100, offloadLayers: 0 },
  },
  // R10: numeric bounds (no model/GPU dependency: live in both modes)
  {
    rule: 'R10',
    path: 'field',
    name: 'batch 0',
    config: { batchSize: 0 },
    model: L70,
    gpu: H100,
    expected: { batchSize: 1 },
  },
  {
    rule: 'R10',
    path: 'dependency',
    name: 'LoRA rank 0 in training',
    config: { mode: 'training', loraRank: 0 },
    model: L70,
    gpu: H100,
    expected: { loraRank: 1 },
  },
  {
    rule: 'R10',
    path: 'link',
    name: 'every bound at once',
    config: { sequenceLength: 100, numNodes: 12, concurrentUsers: 0 },
    model: L70,
    gpu: H100,
    expected: {
      sequenceLength: 512,
      numNodes: 8,
      concurrentUsers: 1,
    },
  },
  {
    rule: 'R10',
    path: 'link',
    name: 'gradient accumulation 0 in training',
    config: { mode: 'training', gradientAccumulationSteps: 0 },
    model: L70,
    gpu: H100,
    expected: { gradientAccumulationSteps: 1 },
  },
  // R12: KV cache offload excludes a KV tier
  {
    rule: 'R12',
    path: 'field',
    name: 'tier on top of KV offload',
    config: {
      offloadingEnabled: true,
      kvCacheOffload: true,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'network' },
    },
    model: L70,
    gpu: H100,
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } },
  },
  {
    rule: 'R12',
    path: 'dependency',
    name: 'tier then offloading turned on',
    config: {
      offloadingEnabled: true,
      kvCacheOffload: true,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'local-nvme' },
    },
    model: L70,
    gpu: H100,
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'none' } },
  },
  {
    rule: 'R12',
    path: 'link',
    name: 'stale kvCacheOffload with offloading off is inert',
    config: {
      offloadingEnabled: false,
      kvCacheOffload: true,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'network' },
    },
    model: L70,
    gpu: H100,
    expected: { kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } },
  },
  // R14: vLLM tensor-parallel divisibility
  {
    rule: 'R14',
    path: 'field',
    name: 'Llama 70B on 6',
    config: { numGPUs: 6 },
    model: L70,
    gpu: H100,
    expected: { numGPUs: 4 },
  },
  {
    rule: 'R14',
    path: 'dependency',
    name: 'PP at 6 then tensor parallel',
    config: { numGPUs: 6, shardingStrategy: 'tensor-parallel' },
    model: L70,
    gpu: H100,
    expected: { numGPUs: 4 },
  },
  {
    rule: 'R14',
    path: 'dependency',
    name: '8 then a 4-head model',
    config: { numGPUs: 8 },
    model: GEMMA_1B,
    gpu: H100,
    validUnder: { model: L70, gpu: H100 },
    expected: { numGPUs: 4 },
  },
  {
    rule: 'R14',
    path: 'link',
    name: 'Kimi K3 keeps 6 (96 heads)',
    config: { numGPUs: 6 },
    model: KIMI_K3,
    gpu: H100,
    expected: { numGPUs: 6 },
  },
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

  // Item B: R4, R9 and part of R10 used to run with modes: BOTH, so a shared link
  // opened in training silently "corrected" a field InputPanel never shows there
  // (ADR 0004: inert inputs are hidden and ignored, never reset).
  it('numNodes and concurrentUsers are inert in training: a link with nn:12 is not silently trimmed', () => {
    const training = cfg({ mode: 'training', numNodes: 12, concurrentUsers: 999999 })
    const result = normalizeConfig(training, L70, H100)
    expect(result.corrections).toEqual([])
    expect(result.config.numNodes).toBe(12)
    expect(result.config.concurrentUsers).toBe(999999)

    const inference = cfg({ mode: 'inference', numNodes: 12, concurrentUsers: 999999 })
    expect(normalizeConfig(inference, L70, H100).config).toMatchObject({
      numNodes: 8,
      concurrentUsers: MAX_CONCURRENT_USERS,
    })
  })

  it('offload fields (R9) and KV tier/host capacity bounds (R4) are inert in training', () => {
    const training = cfg({
      mode: 'training',
      offloadLayers: 500,
      offloadPercentage: 150,
      offloadHostCapacityGB: -1,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'network', activeShare: 5 },
    })
    const result = normalizeConfig(training, L70, H100)
    expect(result.corrections).toEqual([])
    expect(result.config).toEqual(training)

    const inference = { ...training, mode: 'inference' as const }
    const inferenceResult = normalizeConfig(inference, L70, H100)
    expect(inferenceResult.corrections.length).toBeGreaterThan(0)
  })

  it('R1: numGPUs beyond the GPU max is inert in training without a ZeRO preset (GPU count is hidden)', () => {
    const training = cfg({ mode: 'training', frameworkPreset: 'none', numGPUs: 999 })
    const result = normalizeConfig(training, L70, H100)
    expect(result.corrections).toEqual([])
    expect(result.config.numGPUs).toBe(999)
  })

  it('R1: numGPUs is still corrected in training with a ZeRO preset (GPU count is visible)', () => {
    const training = cfg({ mode: 'training', frameworkPreset: 'deepspeed-zero3', numGPUs: 999 })
    const result = normalizeConfig(training, L70, H100)
    expect(result.config.numGPUs).toBe(8) // H100's max_gpus_per_node
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
  {
    name: 'dense EP at 100 GPUs',
    config: cfg({ shardingStrategy: 'expert-parallel', numGPUs: 100 }),
    model: L70,
    gpu: H100,
  },
  {
    name: 'GLM-4.7-Flash at 100 on 8',
    config: cfg({ numGPUs: 100 }),
    model: findModel('zai-org-glm-4.7-flash'),
    gpu: H100,
  },
  {
    name: 'every bound broken',
    config: cfg({
      numGPUs: 2.5,
      batchSize: 0,
      sequenceLength: 100,
      numNodes: 12,
      concurrentUsers: 0,
      offloadLayers: 500,
      offloadPercentage: -5,
      offloadHostCapacityGB: -1,
      interconnectOverride: 'nvlink-9',
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 0 },
    }),
    model: L70,
    gpu: H100,
  },
  {
    name: 'unified memory, everything offloaded',
    config: cfg({
      offloadingEnabled: true,
      offloadTarget: 'cpu-ram',
      kvCacheOffload: true,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' },
    }),
    model: L8,
    gpu: UNIFIED_M3,
  },
  {
    name: 'training with inference preset',
    config: cfg({
      mode: 'training',
      frameworkPreset: 'tgi',
      cpuOffloadOptimizer: true,
      loraRank: 0,
      gradientAccumulationSteps: -1,
    }),
    model: L70,
    gpu: UNIFIED_M3,
  },
  { name: 'NaN count', config: cfg({ numGPUs: Number.NaN }), model: L70, gpu: NVL72 },
  {
    name: 'a billion GPUs (R14 must stay bounded in any rule order)',
    config: cfg({ numGPUs: 1e9 }),
    model: L70,
    gpu: H100,
  },
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

  it.each(HOSTILE)(
    'is independent of rule order and converges within 4 passes: $name',
    ({ config, model, gpu }) => {
      const reference = normalizeConfig(config, model, gpu).config
      for (const order of ruleOrders()) {
        const result = normalizeConfig(config, model, gpu, order)
        expect(result.config).toEqual(reference)
        expect(result.passes).toBeLessThanOrEqual(MAX_NORMALIZE_PASSES)
      }
    },
  )

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
    expect(
      allowedOptions({ ...input, shardingStrategy: 'pipeline-parallel' }, L70, H100).gpuCounts,
    ).toEqual(all8)
    expect(allowedOptions({ ...input, mode: 'training' }, L70, H100).gpuCounts).toEqual(all8)
    expect(allowedOptions(input, null, H100).gpuCounts).toEqual(all8)
  })

  it('offers expert parallel only for a splittable MoE (R2)', () => {
    expect(allowedOptions(input, L70, H100).strategies).toEqual([
      'tensor-parallel',
      'pipeline-parallel',
    ])
    expect(allowedOptions(input, DSR1, H100).strategies).toContain('expert-parallel')
    expect(allowedOptions(input, null, H100).strategies).not.toContain('expert-parallel')
  })

  it('filters KV tiers by Grace host, unified memory and KV offload (R3, R6, R12)', () => {
    expect(allowedOptions(input, L70, H100).kvTiers).toEqual([
      'none',
      'host-pcie',
      'local-nvme',
      'network',
    ])
    expect(allowedOptions(input, L70, NVL72).kvTiers).toContain('host-grace')
    expect(allowedOptions(input, L8, UNIFIED_M3).kvTiers).toEqual(['none', 'local-nvme', 'network'])
    expect(
      allowedOptions({ ...input, offloadingEnabled: true, kvCacheOffload: true }, L70, H100)
        .kvTiers,
    ).toEqual(['none'])
  })

  it('offers NVMe only on unified memory (R6)', () => {
    expect(allowedOptions(input, L8, UNIFIED_M3).offloadTargets).toEqual(['nvme'])
    expect(allowedOptions(input, L70, H100).offloadTargets).toEqual(['cpu-ram', 'nvme'])
  })

  it('offers interconnect variants only with two or more options on a multi-GPU part (R5, R13)', () => {
    expect(allowedOptions(input, L70, WITH_OPTIONS).interconnectOptions).toEqual([
      'nvlink-4',
      'pcie-5',
    ])
    expect(allowedOptions(input, L70, H100).interconnectOptions).toEqual([])
    const singleWithOptions = validateGPU({ ...M3, interconnect_options: ['pcie-5', 'nvlink-5'] })
    expect(allowedOptions(input, L70, singleWithOptions).interconnectOptions).toEqual([])
  })

  it('offers CPU optimizer offload only with a ZeRO preset and separate host memory (R6, R8)', () => {
    expect(
      allowedOptions({ ...input, frameworkPreset: 'deepspeed-zero3' }, L70, H100)
        .cpuOffloadOptimizer,
    ).toBe(true)
    expect(
      allowedOptions({ ...input, frameworkPreset: 'unsloth' }, L70, H100).cpuOffloadOptimizer,
    ).toBe(false)
    expect(
      allowedOptions({ ...input, frameworkPreset: 'deepspeed-zero3' }, L8, UNIFIED_M3)
        .cpuOffloadOptimizer,
    ).toBe(false)
  })

  it('never offers a value the store would correct', () => {
    const pairs: [Model, GPU][] = [
      [L70, H100],
      [KIMI_K3, H100],
      [GEMMA_1B, NVL72],
      [DSR1, NVL72],
      [L8, UNIFIED_M3],
    ]
    for (const [model, gpu] of pairs) {
      const options = allowedOptions(input, model, gpu)
      for (const n of options.gpuCounts) {
        expect(
          normalizeConfig(cfg({ numGPUs: n }), model, gpu).corrections,
          `${model.id} ${n}`,
        ).toEqual([])
      }
      for (const tier of options.kvTiers) {
        const tiered = cfg({ kvTier: { ...DEFAULT_KV_TIER, tier } })
        expect(normalizeConfig(tiered, model, gpu).corrections, `${gpu.id} ${tier}`).toEqual([])
      }
    }
  })
})

describe('softWarnings', () => {
  const base = {
    mode: 'inference' as const,
    numGPUs: 1,
    numNodes: 1,
    shardingStrategy: 'tensor-parallel' as const,
  }

  it('W3: multi-node clusters of single-GPU or unified-memory parts', () => {
    expect(softWarnings({ ...base, numNodes: 2 }, L70, M3).map((w) => w.id)).toEqual(['W3'])
    expect(softWarnings({ ...base, numNodes: 2 }, L70, H100)).toEqual([])
  })

  it('W6: more pipeline stages than layers', () => {
    const w = softWarnings(
      { ...base, numGPUs: 8, numNodes: 8, shardingStrategy: 'pipeline-parallel' },
      L8,
      H100,
    )
    expect(w).toEqual([
      {
        id: 'W6',
        message: `64 pipeline stages exceed ${L8.name}'s ${L8.num_hidden_layers} layers; some stages would be empty.`,
      },
    ])
  })

  it('W8: experts that do not divide by the EP degree', () => {
    const uneven = softWarnings(
      { ...base, numGPUs: 6, shardingStrategy: 'expert-parallel' },
      DSR1,
      H100,
    )
    expect(uneven).toEqual([
      { id: 'W8', message: `${DSR1.num_experts} experts don't split evenly across 6 GPUs.` },
    ])
    expect(
      softWarnings({ ...base, numGPUs: 8, shardingStrategy: 'expert-parallel' }, DSR1, H100),
    ).toEqual([])
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
    const r = normalizeConfig(
      cfg({ numGPUs: 64, kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' } }),
      L70,
      H100,
    )
    const notice = buildNotice(r.corrections, 'gpu', { model: L70, gpu: H100, config: r.config })
    expect(notice?.title).toBe(`Adjusted for ${H100.name}`)
    expect(notice?.lines).toEqual([
      `GPU count set to 8: ${H100.name} supports at most 8 per server.`,
      `KV tier turned off: ${H100.name} has no Grace host memory.`,
    ])
  })

  it('titles a restored link "Shared link adjusted"', () => {
    const r = normalizeConfig(cfg({ batchSize: 0 }), L70, H100)
    expect(
      buildNotice(r.corrections, 'link', { model: L70, gpu: H100, config: r.config })?.title,
    ).toBe('Shared link adjusted')
  })

  it('titles a mode switch after the mode', () => {
    const r = normalizeConfig(cfg({ mode: 'training', frameworkPreset: 'vllm' }), L70, H100)
    expect(
      buildNotice(r.corrections, 'mode', { model: L70, gpu: H100, config: r.config })?.title,
    ).toBe('Adjusted for fine-tuning mode')
  })

  it('titles a plain setting change after the first correction subject', () => {
    const r = normalizeConfig(cfg({ numGPUs: 6 }), L70, H100)
    expect(
      buildNotice(r.corrections, 'setting', { model: L70, gpu: H100, config: r.config })?.title,
    ).toBe(`Adjusted for ${L70.name}`)
  })

  it('titles a reset "Reset to defaults" (Task 2a\'s resetAdvancedSettings/resetAll)', () => {
    const r = normalizeConfig(cfg({ numGPUs: 6 }), L70, H100)
    expect(
      buildNotice(r.corrections, 'reset', { model: L70, gpu: H100, config: r.config })?.title,
    ).toBe('Reset to defaults')
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
