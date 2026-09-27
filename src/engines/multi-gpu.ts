import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import {
  BYTES_PER_GB,
  INTERCONNECT_LABELS,
  INTERCONNECT_SPECS,
  MAX_GPUS_PER_NODE,
  NCCL_BUFFER_PER_GPU_GB,
  PP_ACTIVATION_STASHING_OVERHEAD,
  PP_COMMUNICATION_OVERHEAD,
} from './constants'
import { moeWeightSplit, splitMoEParams } from './inference'
import type {
  InferenceVRAMBreakdown,
  InterconnectType,
  InterconnectValidation,
  MultiGPUVRAMBreakdown,
  QuantizationFormat,
  ShardingStrategy,
} from './types'

/**
 * Calculate replicated memory for tensor parallelism
 *
 * In TP, layer norms are replicated across all GPUs. Embeddings and the LM head
 * are not: vLLM's VocabParallelEmbedding and ParallelLMHead split the vocabulary
 * across TP ranks, so they shard like every other weight.
 *
 * @param model - Model configuration
 * @returns Replicated memory in GB as Decimal
 */
function calculateReplicatedMemory(model: Model): Decimal {
  // Layer norm memory: num_layers * 2 (pre/post) * hidden_size * 4 bytes (FP32)
  return new Decimal(model.num_hidden_layers).mul(2).mul(model.hidden_size).mul(4).div(BYTES_PER_GB)
}

/**
 * How many ways tensor parallelism splits the KV cache, following vLLM's
 * ModelConfig.get_num_kv_heads: MLA caches one latent duplicated on every rank,
 * and GQA heads are replicated once TP exceeds the KV head count, so each GPU
 * holds at least one head.
 */
export function kvCacheTPShards(model: Model, numGPUs: number): number {
  if (model.use_mla) return 1
  return Math.min(numGPUs, model.num_kv_heads ?? model.num_attention_heads)
}

/**
 * Calculate tensor parallelism VRAM distribution
 *
 * TP shards model weights, KV cache, and activations across GPUs.
 * Layer norms are replicated, and the KV cache stops splitting at the KV head
 * count (MLA does not split at all). Interconnect bandwidth costs throughput,
 * not memory: it only sets the scaling efficiency used by the performance model.
 *
 * @param singleGPU - Single-GPU VRAM breakdown
 * @param model - Model configuration
 * @param gpuVramGB - GPU VRAM capacity in GB
 * @param numGPUs - Number of GPUs
 * @param interconnectType - Resolved interconnect type for scaling efficiency
 * @returns Multi-GPU VRAM breakdown
 */
function calculateTensorParallelVRAM(
  singleGPU: InferenceVRAMBreakdown,
  model: Model,
  gpuVramGB: number,
  numGPUs: number,
  interconnectType: InterconnectType,
): MultiGPUVRAMBreakdown {
  const interconnectSpec = INTERCONNECT_SPECS[interconnectType]
  const scalingEfficiency = interconnectSpec.tpScalingEfficiency

  // Calculate replicated memory (layer norms)
  const replicatedMemory = calculateReplicatedMemory(model)

  // Shardable weights = total weights - replicated
  const shardableWeights = singleGPU.modelWeights.sub(replicatedMemory)

  // Weights per GPU = (shardable / numGPUs) + replicated
  const weightsPerGPU = shardableWeights.div(numGPUs).add(replicatedMemory)

  // KV cache divided across the ranks that hold distinct KV heads; linear-attention
  // state splits across every rank (vLLM divides its heads by tp_world_size).
  // Clamped to kvCache: an offloaded KV cache carries no state on the device.
  const linearState = Decimal.min(singleGPU.linearState ?? 0, singleGPU.kvCache)
  const kvCachePerGPU = singleGPU.kvCache
    .sub(linearState)
    .div(kvCacheTPShards(model, numGPUs))
    .add(linearState.div(numGPUs))

  // Activations divided across GPUs
  const activationsPerGPU = singleGPU.activations.div(numGPUs)

  // Framework overhead is per process: every rank pays one CUDA/ROCm context.
  const frameworkOverheadPerGPU = singleGPU.frameworkOverhead

  // Communication memory is the NCCL buffers: flat per GPU, not per peer.
  // Ring/tree allreduce gives each rank a fixed handful of connections however
  // large the group is, so this does not grow with numGPUs. No numGPUs > 1
  // guard is needed: a single GPU forms no communicator and never reaches this
  // path, returning through the passthrough in calculateMultiGPUVRAM instead.
  const communicationOverhead = NCCL_BUFFER_PER_GPU_GB

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
    numNodes: 1,
    gpusPerNode: numGPUs,
    intraNodeEfficiency: scalingEfficiency,
    interNodeGBps: 0,
    bubbleEfficiency: 1,
    scalingEfficiency,
    prefillScalingEfficiency: scalingEfficiency,
    interconnectBandwidthGBps: interconnectSpec.bandwidthGBps,
  }
}

/**
 * Calculate pipeline parallelism VRAM distribution
 *
 * PP divides layers across GPUs. KV cache is sharded by layer: each stage
 * holds only the cache for the layers it owns.
 * No weight replication (layers are split, not sharded).
 *
 * @param singleGPU - Single-GPU VRAM breakdown
 * @param model - Model configuration
 * @param gpuVramGB - GPU VRAM capacity in GB
 * @param numGPUs - Number of GPUs
 * @returns Multi-GPU VRAM breakdown
 */
function calculatePipelineParallelVRAM(
  singleGPU: InferenceVRAMBreakdown,
  _model: Model,
  gpuVramGB: number,
  numGPUs: number,
): MultiGPUVRAMBreakdown {
  // Weights divided evenly across layers
  const weightsPerGPU = singleGPU.modelWeights.div(numGPUs)

  // KV cache is sharded by layer: PP assigns a contiguous slice of layers to
  // each stage, and the KV cache is per-layer, so a stage holds only its own
  // layers' cache. (singleGPU.kvCache is the all-layer total — kv-cache.ts
  // multiplies by num_hidden_layers.)
  const kvCachePerGPU = singleGPU.kvCache.div(numGPUs)

  // Activations divided with stashing overhead
  const baseActivationsPerGPU = singleGPU.activations.div(numGPUs)
  const activationsPerGPU = baseActivationsPerGPU.mul(
    new Decimal(1).add(PP_ACTIVATION_STASHING_OVERHEAD),
  )

  // Framework overhead (no NCCL buffers for PP)
  const frameworkOverheadPerGPU = singleGPU.frameworkOverhead

  // Stage-to-stage sends reuse activation buffers already counted above; the
  // pipeline's communication cost is throughput (scaling efficiency), not memory.
  const communicationOverhead = new Decimal(0)

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
    numNodes: 1,
    gpusPerNode: numGPUs,
    // PP has lower communication overhead than TP; use flat 95% efficiency
    intraNodeEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),
    interNodeGBps: 0,
    bubbleEfficiency: 1,
    scalingEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),
    prefillScalingEfficiency: 1 - PP_COMMUNICATION_OVERHEAD.toNumber(),
    interconnectBandwidthGBps: 0,
  }
}

/**
 * Calculate expert parallel + DP attention VRAM distribution (MoE only)
 *
 * vLLM Expert Parallel Deployment: "Expert (MoE) layers are sharded across all EP
 * ranks... Attention layers... replicated across DP ranks if TP=1". So the routed
 * experts divide by numGPUs, everything else (attention, embeddings, shared and
 * dense layers) is replicated, and each GPU serves its own sessions: the KV cache
 * divides by numGPUs with no MLA duplication.
 *
 * @throws Error for a model that is not a splittable MoE
 */
function calculateExpertParallelVRAM(
  singleGPU: InferenceVRAMBreakdown,
  model: Model,
  gpuVramGB: number,
  numGPUs: number,
  interconnectType: InterconnectType,
  quantization: QuantizationFormat | undefined,
): MultiGPUVRAMBreakdown {
  const split = splitMoEParams(model)
  if (!split) {
    throw new Error(`Expert parallelism needs a MoE model, got ${model.name}`)
  }
  const interconnectSpec = INTERCONNECT_SPECS[interconnectType]

  // The replicated base and the routed experts at their own rates (moeWeightSplit),
  // applied as shares of the on-device weights so offloading still scales them.
  // No quantization passed in: keep the parameter-fraction split unchanged.
  const weightSplit = quantization ? moeWeightSplit(model, quantization) : null
  const baseShare = weightSplit
    ? weightSplit.baseGiB.div(weightSplit.baseGiB.add(weightSplit.routedGiB))
    : new Decimal(split.baseB).div(split.baseB + split.routedB)
  const replicatedMemory = singleGPU.modelWeights.mul(baseShare)
  const routedWeights = singleGPU.modelWeights.sub(replicatedMemory)
  const weightsPerGPU = replicatedMemory.add(routedWeights.div(numGPUs))

  const kvCachePerGPU = singleGPU.kvCache.div(numGPUs)
  const activationsPerGPU = singleGPU.activations.div(numGPUs)
  const frameworkOverheadPerGPU = singleGPU.frameworkOverhead
  const communicationOverhead = NCCL_BUFFER_PER_GPU_GB

  const totalPerGPU = weightsPerGPU
    .add(kvCachePerGPU)
    .add(activationsPerGPU)
    .add(frameworkOverheadPerGPU)
    .add(communicationOverhead)

  return {
    numGPUs,
    strategy: 'expert-parallel',
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
    utilizationPercent: totalPerGPU.div(gpuVramGB).mul(100),
    singleGPUBaseline: singleGPU.total,
    numNodes: 1,
    gpusPerNode: numGPUs,
    intraNodeEfficiency: interconnectSpec.tpScalingEfficiency,
    interNodeGBps: 0,
    bubbleEfficiency: 1,
    scalingEfficiency: interconnectSpec.tpScalingEfficiency,
    prefillScalingEfficiency: interconnectSpec.tpScalingEfficiency,
    interconnectBandwidthGBps: interconnectSpec.bandwidthGBps,
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
 * @param numGPUs - Number of GPUs (1-72)
 * @param strategy - Sharding strategy
 * @param gpu - GPU configuration (used to derive interconnect bandwidth for TP overhead)
 * @returns Multi-GPU VRAM breakdown
 *
 * @throws Error if numGPUs < 1 or > MAX_GPUS_PER_NODE (72)
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
 * const multiGPU = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100)
 * // multiGPU.totalPerGPU: ~15 GB per GPU (vs 42 GB single GPU)
 * // multiGPU.scalingEfficiency: 0.92 (NVLink-4)
 * ```
 */
export function calculateMultiGPUVRAM(
  singleGPU: InferenceVRAMBreakdown,
  model: Model,
  gpuVramGB: number,
  numGPUs: number,
  strategy: ShardingStrategy,
  gpu: GPU,
  quantization?: QuantizationFormat,
): MultiGPUVRAMBreakdown {
  // Validate numGPUs range
  if (numGPUs < 1 || numGPUs > MAX_GPUS_PER_NODE) {
    throw new Error(`numGPUs must be between 1 and ${MAX_GPUS_PER_NODE}, got ${numGPUs}`)
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
      numNodes: 1,
      gpusPerNode: 1,
      intraNodeEfficiency: 1.0,
      interNodeGBps: 0,
      bubbleEfficiency: 1,
      scalingEfficiency: 1.0,
      prefillScalingEfficiency: 1.0,
      interconnectBandwidthGBps: 0,
    }
  }

  // Multi-GPU calculation
  const interconnectType = resolveInterconnect(gpu, numGPUs)

  if (strategy === 'tensor-parallel') {
    return calculateTensorParallelVRAM(singleGPU, model, gpuVramGB, numGPUs, interconnectType)
  }
  if (strategy === 'expert-parallel') {
    return calculateExpertParallelVRAM(
      singleGPU,
      model,
      gpuVramGB,
      numGPUs,
      interconnectType,
      quantization,
    )
  }
  return calculatePipelineParallelVRAM(singleGPU, model, gpuVramGB, numGPUs)
}

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

  // Direct mapping for specific types
  if (interconnect === 'nvlink-3') return 'nvlink-3'
  if (interconnect === 'nvlink-4') return 'nvlink-4'
  if (interconnect === 'nvlink-5') return 'nvlink-5'
  if (interconnect === 'pcie-4') return 'pcie-4'
  if (interconnect === 'pcie-5') return 'pcie-5'

  // Generic nvlink maps to nvlink-4
  if (interconnect === 'nvlink') return 'nvlink-4'

  // Apple unified memory (no multi-GPU support)
  if (interconnect === 'unified') return 'none'

  // AMD Infinity Fabric (xGMI) — its own tier, not a PCIe stand-in
  if (interconnect === 'infinity-fabric') return 'infinity-fabric'

  // Fallback based on GPU tier for undefined or 'none'
  if (interconnect === undefined || interconnect === 'none') {
    if (gpu.tier === 'datacenter') return 'pcie-5'
    if (gpu.tier === 'consumer') return 'pcie-4'
    return 'none' // apple-silicon
  }

  return 'none'
}

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
  const interconnectType = resolveInterconnect(gpu, numGPUs)
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

  // Check if the degree exceeds the interconnect's recommended maximum. This
  // is a soft "scales badly" warning (INTERCONNECT_SPECS.recommendedMaxTPDegree),
  // separate from the hard "cannot be built" bound (GPU.max_gpus_per_node)
  // enforced at the store boundary. It applies regardless of strategy: a
  // 72-way pipeline-parallel run (reachable now that the flat 8-GPU guard is
  // gone) degrades for a different reason than tensor parallelism, so it gets
  // its own wording rather than reusing the TP sentence verbatim.
  // Expert parallelism spans the whole scale-up domain by design (e.g. EP72 on an
  // NVL72); its all-to-all cost is priced in the decode model instead.
  if (strategy !== 'expert-parallel' && numGPUs > spec.recommendedMaxTPDegree) {
    const interconnectName = INTERCONNECT_LABELS[spec.type] ?? spec.type

    const warning =
      strategy === 'tensor-parallel'
        ? `${interconnectName} may have significant communication overhead with ${numGPUs} GPUs (recommended max: ${spec.recommendedMaxTPDegree})`
        : `Pipeline parallelism across ${numGPUs} GPUs may suffer from pipeline bubble overhead and stage imbalance beyond the recommended ${spec.recommendedMaxTPDegree}-way split on ${interconnectName}`

    return {
      valid: true,
      warning,
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
