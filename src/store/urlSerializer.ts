import type {
  FineTuningMethod,
  KVCachePrecision,
  OffloadMode,
  OffloadTarget,
  OptimizerType,
  QuantizationFormat,
  ShardingStrategy,
  TrainingPrecision,
} from '@engines/types'
import type { GPU, Model } from '@utils/schemas'
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
      num_parameters_billion: z.number(),
      hidden_size: z.number(),
      num_hidden_layers: z.number(),
      num_attention_heads: z.number(),
      num_kv_heads: z.number().optional(),
      intermediate_size: z.number(),
    })
    .optional(),
  // GPU: ID for curated, full params for custom
  gpuId: z.string().optional(),
  customGPU: z
    .object({
      name: z.string(),
      vram_gb: z.number(),
      memory_bandwidth_gbps: z.number(),
      fp16_tflops: z.number().optional(),
    })
    .optional(),
  // Calculation parameters (short keys)
  q: z.string(), // quantization
  sl: z.number(), // sequenceLength
  bs: z.number(), // batchSize
  kvq: z.string(), // kvQuantization
  ng: z.number(), // numGPUs
  ss: z.string(), // shardingStrategy
  // Offloading (only if enabled)
  oe: z.boolean().optional(), // offloadingEnabled
  ot: z.string().optional(), // offloadTarget
  om: z.string().optional(), // offloadMode
  op: z.number().optional(), // offloadPercentage
  ol: z.number().optional(), // offloadLayers
  ko: z.boolean().optional(), // kvCacheOffload
  // Mode (only present if training; absence = inference for backward compat)
  m: z.enum(['inference', 'training']).optional(),
  // Training parameters (only present when mode=training)
  tm: z.enum(['full', 'lora', 'qlora']).optional(), // trainingMethod
  to: z.enum(['adamw', 'sgd-momentum', 'adamw-8bit', 'adafactor']).optional(), // optimizer
  tp: z.enum(['fp32', 'fp16', 'bf16']).optional(), // trainingPrecision
  lr: z.number().optional(), // loraRank
  la: z.number().optional(), // loraAlpha
  tmp: z.number().optional(), // targetModulesPercent
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
  offloadingEnabled: boolean
  offloadTarget: OffloadTarget
  offloadMode: OffloadMode
  offloadPercentage: number
  offloadLayers: number
  kvCacheOffload: boolean
  mode: 'inference' | 'training'
  trainingMethod: FineTuningMethod
  optimizer: OptimizerType
  trainingPrecision: TrainingPrecision
  loraRank: number
  loraAlpha: number
  targetModulesPercent: number
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

    // Offloading (only if enabled)
    ...(state.offloadingEnabled
      ? {
          oe: state.offloadingEnabled,
          ot: state.offloadTarget,
          om: state.offloadMode,
          op: state.offloadPercentage,
          ol: state.offloadLayers,
          ko: state.kvCacheOffload,
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
        }
      : {}),
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
