import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import {
  BYTES_PER_GB,
  FRAMEWORK_OVERHEAD_GB,
  PER_GPU_FRAMEWORK_OVERHEAD_GB,
  PREFILL_CHUNK_TOKENS,
} from './constants'
import { calculateKVCacheVRAM } from './kv-cache'
import { calculateModelWeightVRAM } from './quantization'
import type { InferenceVRAMBreakdown, KVCachePrecision, QuantizationFormat } from './types'

/**
 * Calculate active parameters for MoE models
 *
 * Three-tier resolution:
 * 1. Explicit `active_parameters_billion` on the model, when present (e.g. verified
 *    from the model card).
 * 2. Otherwise, derive it from the stored per-expert dimensions: expert parameters are
 *    layers x experts x 3 projections (gate, up, down) x hidden x per-expert intermediate,
 *    with the remainder treated as non-expert (shared) parameters.
 * 3. Dense models, or MoE models missing the expert fields needed for derivation, return
 *    the full parameter count.
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
 * // Tier 1 - explicit field wins
 * calculateMoEActiveParams({ ...qwen35bA3b, active_parameters_billion: 3 }) // 3
 *
 * // Tier 2 - derived from per-expert dimensions
 * calculateMoEActiveParams(qwen35bA3b) // ~4.79
 *
 * // Tier 3 - dense model
 * calculateMoEActiveParams(llama70b) // 70.0
 * ```
 */
export function calculateMoEActiveParams(model: Model): number {
  // Tier 1: explicit value verified from the model card
  if (model.active_parameters_billion) {
    return model.active_parameters_billion
  }

  // Tier 3: dense model, or MoE fields incomplete
  if (model.architecture === 'dense' || !model.num_experts || !model.num_experts_per_token) {
    return model.num_parameters_billion
  }

  // Tier 2: derive from stored dimensions. This assumes `intermediate_size` holds the
  // PER-EXPERT width, which is true only where the stored value sits well below
  // `hidden_size` — Qwen3.6 35B A3B (512 against a hidden size of 2048), Gemma 4 26B A4B
  // (2112), the MiniMax M2.x entries (1536), the Nemotron 3 entries (1856), and both
  // DeepSeek V4 entries (2048 / 3072). The premise does NOT hold for Kimi, GLM,
  // Qwen3-235B, Mistral, Ling, MiniMax M3, or Llama 4, where the field instead holds the
  // dense/shared FFN width (DeepSeek R1: 18432 against a real moe_intermediate_size of
  // 2048). For those architectures this derivation is a rough fallback, not a faithful
  // per-expert count. In practice it currently only runs for the two models without an
  // explicit `active_parameters_billion` (DeepSeek V4 Flash and Pro), and for those two
  // the stored width IS the per-expert one — verified against their config.json, whose
  // `moe_intermediate_size` reads 2048 and 3072 respectively. A separate defect does
  // remain for them: their stored `num_parameters_billion` (158.1 / 861.6) sits well
  // below the real total, so `expertParams` exceeds it, `nonExpertParams` clamps to 0,
  // and the shared expert (`n_shared_experts: 1`) goes uncounted — leaving 6.5B and
  // 24.2B, roughly 15-25% low. Every other MoE entry supplies a verified Tier 1 value
  // and never reaches this branch.
  // Expert parameters are layers x experts x 3 projections (gate, up, down) x hidden x
  // intermediate_size, under the (architecture-dependent) assumption above.
  const expertParams = new Decimal(model.num_hidden_layers)
    .mul(model.num_experts)
    .mul(3)
    .mul(model.hidden_size)
    .mul(model.intermediate_size)
    .div(1e9)

  const total = new Decimal(model.num_parameters_billion)
  // Guard against inconsistent data where the derivation exceeds the declared total
  const nonExpertParams = Decimal.max(total.sub(expertParams), 0)
  const activeRatio = new Decimal(model.num_experts_per_token).div(model.num_experts)

  return Decimal.min(nonExpertParams.add(expertParams.mul(activeRatio)), total).toNumber()
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
 * Formula: batch * min(sequenceLength, PREFILL_CHUNK_TOKENS) * intermediateSize * 4 / BYTES_PER_GB
 * The factor of 4 is 2 bytes (bf16) x ~2 live buffers per layer, not FP32 storage.
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
 * calculateActivationMemory(mixtral, 2048, 1) // Uses ~12.88B active, not 46.7B total
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

  // Peak activations are bounded by the prefill chunk, not the context window.
  // Decode activations are one token wide; prefill is processed PREFILL_CHUNK_TOKENS
  // at a time, so activations plateau once the prompt exceeds one chunk.
  const activeTokens = Math.min(sequenceLength, PREFILL_CHUNK_TOKENS)

  // batch * chunk_tokens * intermediate_size * 4
  // The factor 4 is 2 bytes (bf16 activations) x ~2 live buffers per layer.
  // NOT FP32 storage, despite what this comment used to claim.
  const activationBytes = new Decimal(batchSize)
    .mul(activeTokens)
    .mul(effectiveIntermediateSize)
    .mul(4)

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
 * // breakdown.activations: uses active params (~12.88B) for sizing
 * ```
 */
export function calculateInferenceVRAM(params: {
  model: Model
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  kvQuantization?: KVCachePrecision
  numGPUs?: number
  concurrentUsers?: number
}): InferenceVRAMBreakdown {
  const {
    model,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization = 'fp16',
    numGPUs = 1,
    concurrentUsers,
  } = params

  // 1. Model weights (uses TOTAL params for MoE, not active)
  const modelWeights = calculateModelWeightVRAM(model.num_parameters_billion, quantization)

  // 2. KV cache (applies GQA/MQA reduction automatically)
  // concurrentUsers drives KV cache sizing — each active session holds its full context
  const kvCache = calculateKVCacheVRAM({
    model,
    sequenceLength,
    batchSize: concurrentUsers ?? batchSize,
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
