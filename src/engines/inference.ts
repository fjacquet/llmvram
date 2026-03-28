import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { BYTES_PER_GB, FRAMEWORK_OVERHEAD_GB, PER_GPU_FRAMEWORK_OVERHEAD_GB } from './constants'
import { calculateKVCacheVRAM } from './kv-cache'
import { calculateModelWeightVRAM } from './quantization'
import type { InferenceVRAMBreakdown, KVCachePrecision, QuantizationFormat } from './types'

/**
 * Calculate active parameters for MoE models
 *
 * MoE models have two parameter pools:
 * - Shared parameters (embeddings, layer norms, output head): ~20% of total
 * - Expert parameters (FFN layers): ~80% of total
 *
 * Only a subset of experts are active per token (num_experts_per_token / num_experts).
 * This function returns the effective parameter count for activation memory sizing.
 *
 * IMPORTANT: This is NOT used for weight VRAM calculation. All expert weights must
 * be loaded in VRAM (total params). This is only for activation memory estimation.
 *
 * @param model - Model configuration
 * @returns Active parameters in billions
 *
 * @example
 * ```ts
 * // Mixtral 8x7B: 46.7B total, 8 experts, 2 active per token
 * // Shared: 46.7 * 0.2 = 9.34B
 * // Expert contribution: 46.7 * 0.8 * (2/8) = 9.34B
 * // Total active: 9.34 + 9.34 = 18.68B
 * calculateMoEActiveParams(mixtral) // ~18.68
 *
 * // Dense model (no MoE)
 * calculateMoEActiveParams(llama70b) // 70.0 (unchanged)
 * ```
 */
export function calculateMoEActiveParams(model: Model): number {
  // Dense model or missing MoE fields - return full param count
  if (model.architecture === 'dense' || !model.num_experts || !model.num_experts_per_token) {
    return model.num_parameters_billion
  }

  // MoE model - calculate active parameters
  const totalParams = model.num_parameters_billion
  const activeRatio = model.num_experts_per_token / model.num_experts

  // Approximate split: 20% shared, 80% in experts
  const sharedParams = totalParams * 0.2
  const expertParams = totalParams * 0.8

  // Active params = shared + (expert params * active ratio)
  return sharedParams + expertParams * activeRatio
}

/**
 * Calculate activation memory for forward pass
 *
 * Activations are intermediate tensors stored during the forward pass.
 * Size depends on batch size, sequence length, and intermediate layer size.
 *
 * For MoE models, uses active parameters (not total) since only active experts
 * contribute to activations.
 *
 * Formula: batchSize * sequenceLength * intermediateSize * 4 / BYTES_PER_GB
 * The factor of 4 is for FP32 activation storage (standard precision).
 *
 * @param model - Model configuration
 * @param sequenceLength - Maximum sequence length
 * @param batchSize - Number of concurrent sequences
 * @returns Activation memory in GB as Decimal
 *
 * @example
 * ```ts
 * // 7B model with 11008 intermediate size, seq=2048, batch=1
 * // = 1 * 2048 * 11008 * 4 / (1024^3) = ~0.083 GB
 * calculateActivationMemory(llama7b, 2048, 1)
 *
 * // MoE model uses reduced intermediate size based on active params
 * calculateActivationMemory(mixtral, 2048, 1) // Uses ~20.86B active, not 46.7B total
 * ```
 */
export function calculateActivationMemory(
  model: Model,
  sequenceLength: number,
  batchSize: number,
): Decimal {
  // For MoE models, scale intermediate_size by active param ratio
  let effectiveIntermediateSize = model.intermediate_size

  if (model.architecture === 'moe' && model.num_experts && model.num_experts_per_token) {
    const activeParams = calculateMoEActiveParams(model)
    const paramRatio = activeParams / model.num_parameters_billion
    effectiveIntermediateSize = Math.floor(model.intermediate_size * paramRatio)
  }

  // Activation memory: batch * seq_len * intermediate_size * 4 (FP32 bytes)
  const activationBytes = new Decimal(batchSize)
    .mul(sequenceLength)
    .mul(effectiveIntermediateSize)
    .mul(4) // FP32 bytes per activation

  return activationBytes.div(BYTES_PER_GB)
}

/**
 * Calculate total inference VRAM requirement with detailed breakdown
 *
 * Combines all VRAM components for inference workload:
 * 1. Model weights (post-quantization)
 * 2. KV cache (depends on sequence length, batch size, architecture)
 * 3. Activation memory (forward pass intermediate tensors)
 * 4. Framework overhead (PyTorch + CUDA runtime)
 *
 * CRITICAL NOTES:
 * - For MoE models, weights use TOTAL parameters (46.7B for Mixtral), NOT active (13B)
 * - KV cache applies GQA/MQA reduction automatically via calculateKVCacheVRAM
 * - Activations use active parameters for MoE (only active experts contribute)
 * - KV quantization is independent from weight quantization (INFER-05)
 *
 * @param params - Calculation parameters
 * @param params.model - Model configuration
 * @param params.quantization - Model weight quantization format
 * @param params.sequenceLength - Maximum sequence length
 * @param params.batchSize - Number of concurrent sequences
 * @param params.kvQuantization - KV cache quantization (defaults to fp16)
 * @returns VRAM breakdown with all components in GB as Decimal
 *
 * @example
 * ```ts
 * // Llama 3 70B with GPTQ on H100
 * const breakdown = calculateInferenceVRAM({
 *   model: llama70b,
 *   quantization: 'gptq',
 *   sequenceLength: 4096,
 *   batchSize: 1,
 *   kvQuantization: 'fp16',
 * })
 * // breakdown.modelWeights: ~39.12 GB (70B * 0.6 bytes/param)
 * // breakdown.kvCache: ~1.25 GB (with GQA 8x reduction)
 * // breakdown.activations: ~0.46 GB
 * // breakdown.frameworkOverhead: 1.0 GB
 * // breakdown.total: ~41.83 GB (fits on H100 80GB)
 *
 * // Mixtral 8x7B FP16 (MoE)
 * const moeBreakdown = calculateInferenceVRAM({
 *   model: mixtral,
 *   quantization: 'fp16',
 *   sequenceLength: 2048,
 *   batchSize: 1,
 * })
 * // breakdown.modelWeights: ~86.97 GB (46.7B total params, NOT 13B active)
 * // breakdown.activations: uses active params (~18.68B) for sizing
 * ```
 */
export function calculateInferenceVRAM(params: {
  model: Model
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  kvQuantization?: KVCachePrecision
  numGPUs?: number
}): InferenceVRAMBreakdown {
  const {
    model,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization = 'fp16',
    numGPUs = 1,
  } = params

  // 1. Model weights (uses TOTAL params for MoE, not active)
  const modelWeights = calculateModelWeightVRAM(model.num_parameters_billion, quantization)

  // 2. KV cache (applies GQA/MQA reduction automatically)
  const kvCache = calculateKVCacheVRAM({
    model,
    sequenceLength,
    batchSize,
    kvPrecision: kvQuantization,
  })

  // 3. Activation memory (uses active params for MoE)
  const activations = calculateActivationMemory(model, sequenceLength, batchSize)

  // 4. Framework overhead (PyTorch + CUDA), scaled by GPU count
  const frameworkOverhead = FRAMEWORK_OVERHEAD_GB.add(
    PER_GPU_FRAMEWORK_OVERHEAD_GB.mul(Math.max(0, numGPUs - 1)),
  )

  // 5. Total VRAM
  const total = modelWeights.add(kvCache).add(activations).add(frameworkOverhead)

  return {
    modelWeights,
    kvCache,
    activations,
    frameworkOverhead,
    total,
  }
}
