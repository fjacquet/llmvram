import { maxGPUsFor } from '@utils/gpuLimits'
import { type GPU, MAX_SEQUENCE_LENGTH, type Model } from '@utils/schemas'
import { INTERCONNECT_LABELS, MAX_CONCURRENT_USERS } from './constants'
import { FRAMEWORK_PRESETS, type FrameworkPreset } from './frameworks'
import { splitMoEParams } from './inference'
import {
  clampKVTier,
  DEFAULT_KV_TIER,
  graceLinkGBps,
  KV_TIER_TYPES,
  type KVTierSettings,
  type KVTierType,
} from './kv-tier'
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

export type RuleId =
  | 'R1'
  | 'R2'
  | 'R3'
  | 'R4'
  | 'R5'
  | 'R6'
  | 'R7'
  | 'R8'
  | 'R9'
  | 'R10'
  | 'R12'
  | 'R14'

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

type RangeBound = { field: keyof typeof RANGE_LABELS; min: number; max: number }

// Split by the mode(s) each field's input is visible in (InputPanel): numNodes and
// concurrentUsers only render in inference (NodeCountSelector, ConcurrentUsersInput);
// batchSize and sequenceLength render in both; loraRank and gradientAccumulationSteps
// only render in training (TrainingPanel) — loraRank has no widget at all yet, but it
// only ever feeds a training calculation (useTrainingCalculation), never inference's.
const RANGE_BOUNDS_INFERENCE: RangeBound[] = [
  { field: 'numNodes', min: 1, max: 8 },
  { field: 'concurrentUsers', min: 1, max: MAX_CONCURRENT_USERS },
]
const RANGE_BOUNDS_BOTH: RangeBound[] = [
  { field: 'batchSize', min: 1, max: Number.MAX_SAFE_INTEGER },
  { field: 'sequenceLength', min: 512, max: MAX_SEQUENCE_LENGTH },
]
const RANGE_BOUNDS_TRAINING: RangeBound[] = [
  { field: 'loraRank', min: 1, max: Number.MAX_SAFE_INTEGER },
  { field: 'gradientAccumulationSteps', min: 1, max: Number.MAX_SAFE_INTEGER },
]

function rangeCheck(bounds: readonly RangeBound[]): Rule['check'] {
  return (c) =>
    bounds.flatMap(({ field, min, max }) => {
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
    })
}

/**
 * The hard rules. Order is the dependency order (R1 and R2 before R14, R7 before R8),
 * but normalizeConfig loops to a fixpoint, so any order gives the same result.
 */
export const RULES: readonly Rule[] = [
  {
    id: 'R1',
    modes: BOTH,
    check: (c, { gpu }) => {
      // In training, numGPUs is hidden (InputPanel's showGPUCount) unless a ZeRO
      // preset is selected: no widget shows it, so it must not be corrected.
      if (c.mode === 'training' && FRAMEWORK_PRESETS[c.frameworkPreset].zeroStage === null) {
        return []
      }
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
    modes: INFERENCE,
    check: (c) => {
      const out: Correction[] = []
      const tier = clampKVTier(c.kvTier)
      if (!sameTier(tier, c.kvTier)) {
        out.push(
          fix(
            'R4',
            'kvTier',
            c.kvTier,
            tier,
            'range',
            'KV tier setting adjusted to its allowed range.',
          ),
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
      if (gpu?.unified_memory !== true) return []
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
    modes: INFERENCE,
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
    modes: INFERENCE,
    check: rangeCheck(RANGE_BOUNDS_INFERENCE),
  },
  {
    id: 'R10',
    modes: BOTH,
    check: rangeCheck(RANGE_BOUNDS_BOTH),
  },
  {
    id: 'R10',
    modes: TRAINING,
    check: rangeCheck(RANGE_BOUNDS_TRAINING),
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
export type NoticeSource = 'gpu' | 'model' | 'mode' | 'link' | 'setting' | 'reset'

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
 * "Shared link adjusted" on restore, or "Reset to defaults" (Section 4). One line per
 * rule and field, in order of first appearance, carrying that pair's last message (the
 * final value).
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
  else if (source === 'reset') title = 'Reset to defaults'
  else if (source === 'setting') title = subjectTitle(first.subject, ctx)
  else title = subjectTitle(source, ctx)
  return { title, lines: [...lines.values()] }
}
