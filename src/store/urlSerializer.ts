import type { FrameworkPreset } from '@engines/frameworks'
import { DEFAULT_KV_TIER, KV_TIER_TYPES, type KVTierSettings } from '@engines/kv-tier'
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
import type { UIConfig } from '@store/uiStore'
import type { CustomFabricInput, GPU, Model } from '@utils/schemas'
import { compressToEncodedURIComponent, decompressFromEncodedURIComponent } from 'lz-string'
import { z } from 'zod'

/**
 * Zod schema for URL-serializable state
 * Uses short keys to minimize URL size before compression
 */
export const URLStateSchema = z.object({
  // Model: ID for curated, full params for custom
  modelId: z.string().optional(),
  customModel: z
    .object({
      name: z.string(),
      // No correcting rule reads these fields: unlike sl/bs/ng/cu/nn below, a
      // non-positive value here has nowhere else to be caught before it reaches the
      // engines, so the schema rejects the whole link instead of opening it corrected.
      num_parameters_billion: z.number().positive(),
      hidden_size: z.number().int().positive(),
      num_hidden_layers: z.number().int().positive(),
      num_attention_heads: z.number().int().positive(),
      num_kv_heads: z.number().int().positive().optional(),
      intermediate_size: z.number().int().positive(),
    })
    .optional(),
  // GPU: ID for curated, full params for custom
  gpuId: z.string().optional(),
  customGPU: z
    .object({
      name: z.string(),
      vram_gb: z.number().positive(),
      // nonnegative, not positive: GPUSelector's custom-GPU form defaults an unfilled
      // bandwidth to 0 (createCustomGPU's `input.memory_bandwidth_gbps || 0`), so 0 is a
      // legitimate serialized value, not just a hostile edit.
      memory_bandwidth_gbps: z.number().nonnegative(),
      fp16_tflops: z.number().positive().optional(),
    })
    .optional(),
  // Calculation parameters (short keys)
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
  kt: z
    .object({
      t: z.enum(KV_TIER_TYPES), // tier
      // Not range-checked (same reasoning as sl/bs/cu above): config-rules R4's
      // clampKVTier corrects a, b, g and c after restore, with a "Shared link
      // adjusted" notice, instead of the schema silently rejecting the whole link.
      g: z.number().optional(), // customGBps
      a: z.number(), // activeShare (R4 clamps to 1-100%)
      b: z.number(), // burstSeconds (R4 clamps to >= 1 s)
      c: z.number().optional(), // capacityTB
    })
    .optional(), // KV storage tier (absent = none)
  // Multi-node (absent = single node, for backward compatibility with links
  // created before this feature, where ng meant the total GPU count)
  nn: z.number().optional(), // numNodes (R10 bounds it to 1-8)
  fab: z
    .enum([
      'ethernet-1600g',
      'ethernet-800g',
      'infiniband-xdr',
      'infiniband-ndr',
      'ethernet-400g',
      'ethernet-100g',
      'custom',
    ])
    .optional(), // interNodeFabric
  fabc: z
    .object({
      name: z.string(),
      port_gbps: z.number().positive(),
    })
    .optional(), // customFabric
  // Offloading (only if enabled)
  oe: z.boolean().optional(), // offloadingEnabled
  ot: z.string().optional(), // offloadTarget
  om: z.string().optional(), // offloadMode
  op: z.number().optional(), // offloadPercentage
  ol: z.number().optional(), // offloadLayers
  ko: z.boolean().optional(), // kvCacheOffload
  // offloadHostCapacityGB. Not range-checked (same reasoning as kt above): a
  // hand-edited hc <= 0 (or non-finite) must still open the link. It used to be
  // preprocessed to `undefined` here, which restored silently with no notice;
  // now it passes through and config-rules R4 resets it to the default (null)
  // with a "Shared link adjusted" notice, like every other rule-corrected field.
  hc: z.number().optional(),
  // Mode (only present if training; absence = inference for backward compat)
  m: z.enum(['inference', 'training']).optional(),
  // Training parameters (only present when mode=training)
  tm: z.enum(['full', 'lora', 'qlora']).optional(), // trainingMethod
  to: z.enum(['adamw', 'sgd-momentum', 'adamw-8bit', 'adafactor']).optional(), // optimizer
  tp: z.enum(['fp32', 'fp16', 'bf16']).optional(), // trainingPrecision
  lr: z.number().optional(), // loraRank
  la: z.number().positive().optional(), // loraAlpha
  tmp: z.number().positive().optional(), // targetModulesPercent
  ga: z.number().optional(), // gradientAccumulationSteps
  gc: z.boolean().optional(), // gradientCheckpointing
  fa: z.boolean().optional(), // flashAttention
  // Framework presets and CPU offload (training parameters group)
  fp: z
    .enum([
      'none',
      'deepspeed-zero1',
      'deepspeed-zero2',
      'deepspeed-zero3',
      'unsloth',
      'vllm',
      'tgi',
    ])
    .optional(), // frameworkPreset
  co: z.boolean().optional(), // cpuOffloadOptimizer
  io: z.string().optional(), // interconnectOverride (absent = the GPU's default)
})

export type URLState = z.infer<typeof URLStateSchema>

/**
 * Check if an ID represents a custom model/GPU
 */
export function isCustomId(id: string): boolean {
  return id.startsWith('custom-')
}

/**
 * Serialize store state to compressed URL hash
 */
export function serializeToURL(state: {
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  kvQuantization: KVCachePrecision
  numGPUs: number
  shardingStrategy: ShardingStrategy
  concurrentUsers: number
  kvTier: KVTierSettings
  numNodes: number
  interNodeFabric: FabricType
  customFabric: CustomFabricInput | null
  offloadingEnabled: boolean
  offloadTarget: OffloadTarget
  offloadMode: OffloadMode
  offloadPercentage: number
  offloadLayers: number
  kvCacheOffload: boolean
  offloadHostCapacityGB: number | null
  mode: 'inference' | 'training'
  trainingMethod: FineTuningMethod
  optimizer: OptimizerType
  trainingPrecision: TrainingPrecision
  loraRank: number
  loraAlpha: number
  targetModulesPercent: number
  gradientAccumulationSteps: number
  gradientCheckpointing: boolean
  flashAttention: boolean
  frameworkPreset: FrameworkPreset
  cpuOffloadOptimizer: boolean
  interconnectOverride: string | null
}): string {
  const urlState: URLState = {
    // Model serialization
    ...(state.selectedModel && isCustomId(state.selectedModel.id)
      ? {
          customModel: {
            name: state.selectedModel.name,
            num_parameters_billion: state.selectedModel.num_parameters_billion,
            hidden_size: state.selectedModel.hidden_size,
            num_hidden_layers: state.selectedModel.num_hidden_layers,
            num_attention_heads: state.selectedModel.num_attention_heads,
            num_kv_heads: state.selectedModel.num_kv_heads,
            intermediate_size: state.selectedModel.intermediate_size,
          },
        }
      : state.selectedModel
        ? { modelId: state.selectedModel.id }
        : {}),

    // GPU serialization
    ...(state.selectedGPU && isCustomId(state.selectedGPU.id)
      ? {
          customGPU: {
            name: state.selectedGPU.name,
            vram_gb: state.selectedGPU.vram_gb,
            memory_bandwidth_gbps: state.selectedGPU.memory_bandwidth_gbps,
            fp16_tflops: state.selectedGPU.fp16_tflops,
          },
        }
      : state.selectedGPU
        ? { gpuId: state.selectedGPU.id }
        : {}),

    // Calculation parameters
    q: state.quantization,
    sl: state.sequenceLength,
    bs: state.batchSize,
    kvq: state.kvQuantization,
    ng: state.numGPUs,
    ss: state.shardingStrategy,
    ...(state.concurrentUsers > 1 ? { cu: state.concurrentUsers } : {}),
    // JSON.stringify drops the undefined g / c
    ...(state.kvTier.tier !== 'none'
      ? {
          kt: {
            t: state.kvTier.tier,
            g: state.kvTier.customGBps ?? undefined,
            a: state.kvTier.activeShare,
            b: state.kvTier.burstSeconds,
            c: state.kvTier.capacityTB ?? undefined,
          },
        }
      : {}),

    // Multi-node (only when actually multi-node, to keep single-node links short)
    ...(state.numNodes > 1
      ? {
          nn: state.numNodes,
          fab: state.interNodeFabric,
          ...(state.interNodeFabric === 'custom' && state.customFabric
            ? { fabc: state.customFabric }
            : {}),
        }
      : {}),

    // Offloading (only if enabled)
    ...(state.offloadingEnabled
      ? {
          oe: state.offloadingEnabled,
          ot: state.offloadTarget,
          om: state.offloadMode,
          op: state.offloadPercentage,
          ol: state.offloadLayers,
          ko: state.kvCacheOffload,
          hc: state.offloadHostCapacityGB ?? undefined,
        }
      : {}),

    // Training state (only if mode is training)
    ...(state.mode === 'training'
      ? {
          m: state.mode,
          tm: state.trainingMethod,
          to: state.optimizer,
          tp: state.trainingPrecision,
          lr: state.loraRank,
          la: state.loraAlpha,
          tmp: state.targetModulesPercent,
          ga: state.gradientAccumulationSteps,
          gc: state.gradientCheckpointing,
          fa: state.flashAttention,
          fp: state.frameworkPreset,
          co: state.cpuOffloadOptimizer || undefined, // omit if false
        }
      : {}),

    // Framework preset for inference mode (vLLM/TGI)
    ...(state.frameworkPreset !== 'none' && state.mode === 'inference'
      ? { fp: state.frameworkPreset }
      : {}),

    // Interconnect variant (only when overridden)
    ...(state.interconnectOverride ? { io: state.interconnectOverride } : {}),
  }

  const json = JSON.stringify(urlState)
  return compressToEncodedURIComponent(json)
}

/**
 * Deserialize compressed URL hash to state object
 * Returns null on any failure (invalid format, parse error, schema validation)
 * Never throws exceptions
 */
export function deserializeFromURL(hash: string): URLState | null {
  try {
    // Decompress
    const decompressed = decompressFromEncodedURIComponent(hash)
    if (!decompressed) {
      return null
    }

    // Parse JSON
    const parsed = JSON.parse(decompressed)

    // Validate with schema
    const result = URLStateSchema.safeParse(parsed)
    if (!result.success) {
      return null
    }

    return result.data
  } catch {
    // Any error (decompress, parse, etc.) → return null
    return null
  }
}

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
