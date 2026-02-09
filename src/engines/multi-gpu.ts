import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import {
  BYTES_PER_GB,
  EMBEDDING_WEIGHT_FRACTION,
  INTERCONNECT_SPECS,
  MOE_MULTI_GPU_OVERHEAD,
  NCCL_BUFFER_PER_PEER_GB,
  PP_ACTIVATION_STASHING_OVERHEAD,
  PP_COMMUNICATION_OVERHEAD,
  TP_COMMUNICATION_OVERHEAD,
} from './constants'
import type {
  InferenceVRAMBreakdown,
  InterconnectType,
  InterconnectValidation,
  MultiGPUVRAMBreakdown,
  ShardingStrategy,
} from './types'

/**
 * Calculate replicated memory for tensor parallelism
 *
 * In TP, embeddings and layer norms must be replicated across all GPUs.
 * This function estimates the total replicated memory.
 *
 * @param model - Model configuration
 * @param modelWeightsGB - Total model weights in GB (Decimal)
 * @returns Replicated memory in GB as Decimal
 */
function calculateReplicatedMemory(model: Model, modelWeightsGB: Decimal): Decimal {
  // Layer norm memory: num_layers * 2 (pre/post) * hidden_size * 4 bytes (FP32)
  const layerNormMemory = new Decimal(model.num_hidden_layers)
    .mul(2)
    .mul(model.hidden_size)
    .mul(4)
    .div(BYTES_PER_GB)

  // Embedding memory estimate: ~3% of model weights
  const embeddingMemory = modelWeightsGB.mul(EMBEDDING_WEIGHT_FRACTION)

  return layerNormMemory.add(embeddingMemory)
}

/**
 * Calculate tensor parallelism VRAM distribution
 *
 * TP shards model weights, KV cache, and activations across GPUs.
 * Embeddings and layer norms are replicated.
 *
 * @param singleGPU - Single-GPU VRAM breakdown
 * @param model - Model configuration
 * @param gpuVramGB - GPU VRAM capacity in GB
 * @param numGPUs - Number of GPUs
 * @param isMoE - Whether model is MoE architecture
 * @returns Multi-GPU VRAM breakdown
 */
function calculateTensorParallelVRAM(
  singleGPU: InferenceVRAMBreakdown,
  model: Model,
  gpuVramGB: number,
  numGPUs: number,
  isMoE: boolean,
): MultiGPUVRAMBreakdown {
  // Calculate replicated memory (embeddings + layer norms)
  const replicatedMemory = calculateReplicatedMemory(model, singleGPU.modelWeights)

  // Shardable weights = total weights - replicated
  const shardableWeights = singleGPU.modelWeights.sub(replicatedMemory)

  // Weights per GPU = (shardable / numGPUs) + replicated
  const weightsPerGPU = shardableWeights.div(numGPUs).add(replicatedMemory)

  // KV cache divided across GPUs
  const kvCachePerGPU = singleGPU.kvCache.div(numGPUs)

  // Activations divided across GPUs
  const activationsPerGPU = singleGPU.activations.div(numGPUs)

  // NCCL buffers: 0.2 GB per peer GPU
  const ncclBuffers = NCCL_BUFFER_PER_PEER_GB.mul(numGPUs - 1)
  const frameworkOverheadPerGPU = singleGPU.frameworkOverhead.add(ncclBuffers)

  // Communication overhead: 12% of weights per GPU (15% extra for MoE)
  const moeMultiplier = isMoE ? new Decimal(1).add(MOE_MULTI_GPU_OVERHEAD) : new Decimal(1)
  const communicationOverhead = weightsPerGPU.mul(TP_COMMUNICATION_OVERHEAD).mul(moeMultiplier)

  // Total per GPU
  const totalPerGPU = weightsPerGPU
    .add(kvCachePerGPU)
    .add(activationsPerGPU)
    .add(frameworkOverheadPerGPU)
    .add(communicationOverhead)

  // Utilization percentage
  const utilizationPercent = totalPerGPU.div(gpuVramGB).mul(100)

  return {
    numGPUs,
    strategy: 'tensor-parallel',
    perGPU: {
      modelWeights: weightsPerGPU,
      kvCache: kvCachePerGPU,
      activations: activationsPerGPU,
      frameworkOverhead: frameworkOverheadPerGPU,
      communicationOverhead,
      total: totalPerGPU,
    },
    replicatedMemory,
    totalPerGPU,
    utilizationPercent,
    singleGPUBaseline: singleGPU.total,
  }
}

/**
 * Calculate pipeline parallelism VRAM distribution
 *
 * PP divides layers across GPUs. Each GPU needs full KV cache for its layers.
 * No weight replication (layers are split, not sharded).
 *
 * @param singleGPU - Single-GPU VRAM breakdown
 * @param model - Model configuration
 * @param gpuVramGB - GPU VRAM capacity in GB
 * @param numGPUs - Number of GPUs
 * @param isMoE - Whether model is MoE architecture
 * @returns Multi-GPU VRAM breakdown
 */
function calculatePipelineParallelVRAM(
  singleGPU: InferenceVRAMBreakdown,
  _model: Model,
  gpuVramGB: number,
  numGPUs: number,
  isMoE: boolean,
): MultiGPUVRAMBreakdown {
  // Weights divided evenly across layers
  const weightsPerGPU = singleGPU.modelWeights.div(numGPUs)

  // KV cache NOT divided (each GPU needs full cache for its layers)
  const kvCachePerGPU = singleGPU.kvCache

  // Activations divided with stashing overhead
  const baseActivationsPerGPU = singleGPU.activations.div(numGPUs)
  const activationsPerGPU = baseActivationsPerGPU.mul(
    new Decimal(1).add(PP_ACTIVATION_STASHING_OVERHEAD),
  )

  // Framework overhead (no NCCL buffers for PP)
  const frameworkOverheadPerGPU = singleGPU.frameworkOverhead

  // Communication overhead: 5% of weights per GPU (15% extra for MoE)
  const moeMultiplier = isMoE ? new Decimal(1).add(MOE_MULTI_GPU_OVERHEAD) : new Decimal(1)
  const communicationOverhead = weightsPerGPU.mul(PP_COMMUNICATION_OVERHEAD).mul(moeMultiplier)

  // Total per GPU
  const totalPerGPU = weightsPerGPU
    .add(kvCachePerGPU)
    .add(activationsPerGPU)
    .add(frameworkOverheadPerGPU)
    .add(communicationOverhead)

  // Utilization percentage
  const utilizationPercent = totalPerGPU.div(gpuVramGB).mul(100)

  return {
    numGPUs,
    strategy: 'pipeline-parallel',
    perGPU: {
      modelWeights: weightsPerGPU,
      kvCache: kvCachePerGPU,
      activations: activationsPerGPU,
      frameworkOverhead: frameworkOverheadPerGPU,
      communicationOverhead,
      total: totalPerGPU,
    },
    replicatedMemory: new Decimal(0), // PP has no replication
    totalPerGPU,
    utilizationPercent,
    singleGPUBaseline: singleGPU.total,
  }
}

/**
 * Calculate multi-GPU VRAM distribution
 *
 * Takes a single-GPU VRAM breakdown and distributes it across multiple GPUs
 * using either tensor parallelism or pipeline parallelism strategy.
 *
 * @param singleGPU - Single-GPU VRAM breakdown
 * @param model - Model configuration
 * @param gpuVramGB - GPU VRAM capacity in GB
 * @param numGPUs - Number of GPUs (1-8)
 * @param strategy - Sharding strategy
 * @returns Multi-GPU VRAM breakdown
 *
 * @throws Error if numGPUs < 1 or > 8
 *
 * @example
 * ```ts
 * const singleGPU = calculateInferenceVRAM({
 *   model: llama70b,
 *   quantization: 'gptq',
 *   sequenceLength: 4096,
 *   batchSize: 1,
 * })
 *
 * const multiGPU = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel')
 * // multiGPU.totalPerGPU: ~15 GB per GPU (vs 42 GB single GPU)
 * ```
 */
export function calculateMultiGPUVRAM(
  singleGPU: InferenceVRAMBreakdown,
  model: Model,
  gpuVramGB: number,
  numGPUs: number,
  strategy: ShardingStrategy,
): MultiGPUVRAMBreakdown {
  // Validate numGPUs range
  if (numGPUs < 1 || numGPUs > 8) {
    throw new Error(`numGPUs must be between 1 and 8, got ${numGPUs}`)
  }

  // Single GPU passthrough
  if (numGPUs === 1) {
    const utilizationPercent = singleGPU.total.div(gpuVramGB).mul(100)

    return {
      numGPUs: 1,
      strategy,
      perGPU: {
        modelWeights: singleGPU.modelWeights,
        kvCache: singleGPU.kvCache,
        activations: singleGPU.activations,
        frameworkOverhead: singleGPU.frameworkOverhead,
        communicationOverhead: new Decimal(0),
        total: singleGPU.total,
      },
      replicatedMemory: new Decimal(0),
      totalPerGPU: singleGPU.total,
      utilizationPercent,
      singleGPUBaseline: singleGPU.total,
    }
  }

  // Multi-GPU calculation
  const isMoE = model.architecture === 'moe'

  if (strategy === 'tensor-parallel') {
    return calculateTensorParallelVRAM(singleGPU, model, gpuVramGB, numGPUs, isMoE)
  } else {
    return calculatePipelineParallelVRAM(singleGPU, model, gpuVramGB, numGPUs, isMoE)
  }
}

/**
 * Resolve GPU interconnect string to engine InterconnectType
 *
 * Maps GPU.interconnect field to standardized InterconnectType enum.
 *
 * @param gpu - GPU configuration
 * @returns InterconnectType
 *
 * @example
 * ```ts
 * resolveInterconnect({ interconnect: 'nvlink-4', tier: 'datacenter' }) // 'nvlink-4'
 * resolveInterconnect({ interconnect: 'nvlink', tier: 'datacenter' }) // 'nvlink-4'
 * resolveInterconnect({ interconnect: undefined, tier: 'datacenter' }) // 'pcie-5'
 * resolveInterconnect({ interconnect: 'unified', tier: 'apple-silicon' }) // 'none'
 * ```
 */
export function resolveInterconnect(gpu: GPU): InterconnectType {
  const interconnect = gpu.interconnect

  // Direct mapping for specific types
  if (interconnect === 'nvlink-4') return 'nvlink-4'
  if (interconnect === 'nvlink-5') return 'nvlink-5'
  if (interconnect === 'pcie-4') return 'pcie-4'
  if (interconnect === 'pcie-5') return 'pcie-5'

  // Generic nvlink maps to nvlink-4
  if (interconnect === 'nvlink') return 'nvlink-4'

  // Apple unified memory (no multi-GPU support)
  if (interconnect === 'unified') return 'none'

  // AMD Infinity Fabric maps to pcie-5 (similar bandwidth)
  if (interconnect === 'infinity-fabric') return 'pcie-5'

  // Fallback based on GPU tier for undefined or 'none'
  if (interconnect === undefined || interconnect === 'none') {
    if (gpu.tier === 'datacenter') return 'pcie-5'
    if (gpu.tier === 'consumer') return 'pcie-4'
    return 'none' // apple-silicon
  }

  return 'none'
}

/**
 * Validate interconnect for multi-GPU configuration
 *
 * Checks if the GPU interconnect is suitable for the requested multi-GPU
 * configuration and strategy. Returns validation result with warning if
 * configuration is suboptimal but still valid.
 *
 * @param gpu - GPU configuration
 * @param numGPUs - Number of GPUs
 * @param strategy - Sharding strategy
 * @returns Validation result
 *
 * @example
 * ```ts
 * validateInterconnect(h100, 4, 'tensor-parallel')
 * // { valid: true, warning: null, interconnect: { type: 'nvlink-4', ... } }
 *
 * validateInterconnect(rtx4090, 4, 'tensor-parallel')
 * // { valid: true, warning: 'PCIe 4.0 may have...', interconnect: { type: 'pcie-4', ... } }
 *
 * validateInterconnect(m3Ultra, 2, 'tensor-parallel')
 * // { valid: false, warning: 'does not support multi-GPU', interconnect: { type: 'none', ... } }
 * ```
 */
export function validateInterconnect(
  gpu: GPU,
  numGPUs: number,
  strategy: ShardingStrategy,
): InterconnectValidation {
  const interconnectType = resolveInterconnect(gpu)
  const spec = INTERCONNECT_SPECS[interconnectType]

  // Single GPU is always valid
  if (numGPUs === 1) {
    return {
      valid: true,
      warning: null,
      interconnect: spec,
    }
  }

  // 'none' interconnect cannot support multi-GPU
  if (spec.type === 'none') {
    return {
      valid: false,
      warning: `${gpu.name} does not support multi-GPU configurations`,
      interconnect: spec,
    }
  }

  // Check if TP degree exceeds recommended maximum
  if (strategy === 'tensor-parallel' && numGPUs > spec.recommendedMaxTPDegree) {
    const interconnectName =
      spec.type === 'pcie-4' ? 'PCIe 4.0' : spec.type === 'pcie-5' ? 'PCIe 5.0' : spec.type

    return {
      valid: true,
      warning: `${interconnectName} may have significant communication overhead with ${numGPUs} GPUs (recommended max: ${spec.recommendedMaxTPDegree})`,
      interconnect: spec,
    }
  }

  // Valid with no warnings
  return {
    valid: true,
    warning: null,
    interconnect: spec,
  }
}
