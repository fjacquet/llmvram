import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import {
  BYTES_PER_GB,
  GRADIENT_BYTES,
  MASTER_WEIGHT_BYTES,
  TOTAL_TARGETABLE_MODULES_PER_LAYER,
  TRAINING_FRAMEWORK_OVERHEAD_GB,
} from './constants'
import { applyFlashAttention, applyGradientCheckpointing } from './optimizations'
import { calculateModelWeightVRAM } from './quantization'
import { calculateOptimizerStateMemory, calculateTrainingActivationMemory } from './training'
import type { LoRAVRAMBreakdown, OptimizerType, TrainingPrecision } from './types'

/**
 * Calculate LoRA adapter parameter count
 *
 * LoRA adds low-rank matrices to target modules. Each adapter consists of two
 * matrices: A (d × r) and B (r × d), where d is hidden_size and r is rank.
 * Total params per adapter = 2 * rank * hidden_size.
 *
 * Standard transformer has 7 targetable modules per layer:
 * - Attention: q_proj, k_proj, v_proj, o_proj (4 modules)
 * - MLP: gate_proj, up_proj, down_proj (3 modules)
 *
 * @param params - LoRA adapter configuration
 * @param params.hiddenSize - Model hidden dimension size
 * @param params.numHiddenLayers - Number of transformer layers
 * @param params.rank - LoRA rank (typically 8-256)
 * @param params.targetModulesPercent - Percentage of modules to target (0-100)
 * @returns Adapter parameter count as Decimal (NOT in billions)
 *
 * @example
 * ```ts
 * // 7B model (hidden=4096, layers=32), rank=16, target 50% modules
 * calculateLoRAAdapterParams({
 *   hiddenSize: 4096,
 *   numHiddenLayers: 32,
 *   rank: 16,
 *   targetModulesPercent: 50,
 * })
 * // Returns: ~16.8M parameters (0.24% of 7B base)
 * ```
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export function calculateLoRAAdapterParams(params: {
  hiddenSize: number
  numHiddenLayers: number
  rank: number
  targetModulesPercent: number
}): Decimal {
  const { hiddenSize, numHiddenLayers, rank, targetModulesPercent } = params

  // Calculate number of target modules (minimum 1)
  const targetModuleCount = Math.max(
    1,
    Math.round((TOTAL_TARGETABLE_MODULES_PER_LAYER * targetModulesPercent) / 100),
  )

  // Each LoRA adapter adds two low-rank matrices: A(d, r) + B(r, d) = 2 * rank * d params
  const paramsPerLayer = new Decimal(2).mul(rank).mul(hiddenSize).mul(targetModuleCount)

  // Total adapter params across all layers
  return paramsPerLayer.mul(numHiddenLayers)
}

/**
 * Calculate LoRA fine-tuning VRAM breakdown
 *
 * LoRA freezes the base model and trains only low-rank adapters (~1% of parameters).
 * Optimizer states and gradients apply ONLY to adapter parameters, not the frozen base.
 *
 * Memory components:
 * 1. Base model weights at training precision (frozen, no gradients)
 * 2. Adapter weights in FP16/BF16 (trainable)
 * 3. FP32 master weights for adapters (mixed precision only)
 * 4. Gradients for adapters (same precision as training)
 * 5. Optimizer states for adapters (always FP32)
 * 6. Training activations (batch-dependent, can be optimized)
 * 7. Framework overhead (PyTorch + CUDA + autograd)
 *
 * CRITICAL: This is NOT "full fine-tuning divided by N". Optimizer states
 * apply only to adapters, not the frozen 7B base model.
 *
 * Optional memory optimizations (applied to activations):
 * - gradientCheckpointing: Reduces activation memory by 60%
 * - flashAttention: Reduces activation memory by 15-70% (depends on sequence length)
 *
 * @param params - LoRA training configuration
 * @returns Complete VRAM breakdown with LoRA-specific fields
 *
 * @example
 * ```ts
 * // 7B model, BF16 mixed precision, rank=16, 50% target modules
 * const breakdown = calculateLoRAFineTuningVRAM({
 *   model: llama7b,
 *   trainingPrecision: 'bf16',
 *   optimizer: 'adamw',
 *   batchSize: 1,
 *   sequenceLength: 2048,
 *   loraRank: 16,
 *   targetModulesPercent: 50,
 * })
 * // breakdown.baseWeights: ~13.04 GB (frozen BF16 base)
 * // breakdown.adapterWeights: ~0.031 GB (trainable adapters)
 * // breakdown.optimizerStates: ~0.125 GB (only for adapters!)
 * // breakdown.total: ~19.8 GB (fits on 24GB GPU)
 * ```
 *
 * Reference: .planning/research/PITFALLS.md #3 (LoRA optimizer states)
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export function calculateLoRAFineTuningVRAM(params: {
  model: Model
  trainingPrecision: TrainingPrecision
  optimizer: OptimizerType
  batchSize: number
  sequenceLength: number
  loraRank: number
  targetModulesPercent: number
  gradientCheckpointing?: boolean
  flashAttention?: boolean
}): LoRAVRAMBreakdown {
  const {
    model,
    trainingPrecision,
    optimizer,
    batchSize,
    sequenceLength,
    loraRank,
    targetModulesPercent,
    gradientCheckpointing,
    flashAttention,
  } = params

  // 1. Calculate adapter parameter count
  const adapterParameters = calculateLoRAAdapterParams({
    hiddenSize: model.hidden_size,
    numHiddenLayers: model.num_hidden_layers,
    rank: loraRank,
    targetModulesPercent,
  })

  const adapterParamsBillion = adapterParameters.div(1e9)

  // 2. Base model weights at training precision (frozen, no gradients/optimizer states)
  const baseWeights = calculateModelWeightVRAM(
    model.num_parameters_billion,
    trainingPrecision === 'fp32' ? 'fp32' : trainingPrecision === 'fp16' ? 'fp16' : 'bf16',
  )

  // 3. Adapter weights (always FP16/BF16, typically 2 bytes)
  const adapterWeights = adapterParameters.mul(2).div(BYTES_PER_GB)

  // 4. Master weights (FP32 copy of adapters for mixed precision)
  const masterWeights =
    trainingPrecision === 'fp32'
      ? new Decimal(0)
      : adapterParameters.mul(MASTER_WEIGHT_BYTES).div(BYTES_PER_GB)

  // 5. Gradients for adapters (at training precision)
  const gradients = adapterParameters.mul(GRADIENT_BYTES[trainingPrecision]).div(BYTES_PER_GB)

  // 6. Optimizer states (ONLY for adapters, always FP32)
  const optimizerStates = calculateOptimizerStateMemory(adapterParamsBillion.toNumber(), optimizer)

  // 7. Training activations (batch-dependent, with optional optimizations)
  let activations = calculateTrainingActivationMemory(model, batchSize, sequenceLength)

  // Apply memory optimizations to activations (multiplicative stacking)
  if (gradientCheckpointing) {
    activations = applyGradientCheckpointing(activations, true)
  }
  if (flashAttention) {
    activations = applyFlashAttention(activations, sequenceLength, true)
  }

  // 8. Framework overhead
  const frameworkOverhead = TRAINING_FRAMEWORK_OVERHEAD_GB

  // 9. Total VRAM
  const total = baseWeights
    .add(adapterWeights)
    .add(masterWeights)
    .add(gradients)
    .add(optimizerStates)
    .add(activations)
    .add(frameworkOverhead)

  return {
    modelWeights: baseWeights, // For interface compatibility (this is the frozen base)
    masterWeights,
    gradients,
    optimizerStates,
    activations,
    frameworkOverhead,
    total,
    trainableParameters: adapterParamsBillion,
    totalParameters: new Decimal(model.num_parameters_billion),
    method: 'lora',
    baseWeights,
    adapterWeights,
    adapterParameters: adapterParamsBillion,
  }
}

/**
 * Calculate QLoRA fine-tuning VRAM breakdown
 *
 * QLoRA combines three precision levels for extreme memory efficiency:
 * 1. Frozen base model in 4-bit NF4 (~0.5 bytes/param)
 * 2. Trainable LoRA adapters in FP16 (2 bytes/param)
 * 3. Optimizer states in FP32 (8 bytes/param for AdamW)
 *
 * This is NOT configurable - QLoRA always uses this specific precision architecture
 * for numerical stability and memory efficiency.
 *
 * Memory breakdown:
 * - Base: 7B * 0.5 bytes = ~3.26 GB (vs ~13 GB for FP16)
 * - Adapters: ~16.8M * 2 bytes = ~0.031 GB (same as LoRA)
 * - Optimizer: ~16.8M * 8 bytes = ~0.125 GB (only adapters)
 * - Total: ~10 GB for 7B model (vs ~20 GB for LoRA)
 *
 * This enables fine-tuning 7B models on consumer GPUs (12GB VRAM) and
 * 70B models on single A100 (80GB VRAM).
 *
 * Optional memory optimizations (applied to activations):
 * - gradientCheckpointing: Reduces activation memory by 60%
 * - flashAttention: Reduces activation memory by 15-70% (depends on sequence length)
 *
 * @param params - QLoRA training configuration
 * @returns Complete VRAM breakdown with QLoRA-specific precision architecture
 *
 * @example
 * ```ts
 * // 7B model with QLoRA (NF4 base, FP16 adapters)
 * const breakdown = calculateQLoRAFineTuningVRAM({
 *   model: llama7b,
 *   optimizer: 'adamw',
 *   batchSize: 1,
 *   sequenceLength: 2048,
 *   loraRank: 16,
 *   targetModulesPercent: 50,
 * })
 * // breakdown.baseWeights: ~3.26 GB (NF4 frozen base)
 * // breakdown.adapterWeights: ~0.031 GB (FP16 adapters)
 * // breakdown.masterWeights: ~0.063 GB (FP32 master of adapters)
 * // breakdown.total: ~10 GB (fits on 12GB GPU!)
 * ```
 *
 * Reference: .planning/research/PITFALLS.md #4 (QLoRA three-precision architecture)
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export function calculateQLoRAFineTuningVRAM(params: {
  model: Model
  optimizer: OptimizerType
  batchSize: number
  sequenceLength: number
  loraRank: number
  targetModulesPercent: number
  gradientCheckpointing?: boolean
  flashAttention?: boolean
}): LoRAVRAMBreakdown {
  const {
    model,
    optimizer,
    batchSize,
    sequenceLength,
    loraRank,
    targetModulesPercent,
    gradientCheckpointing,
    flashAttention,
  } = params

  // 1. Calculate adapter parameter count
  const adapterParameters = calculateLoRAAdapterParams({
    hiddenSize: model.hidden_size,
    numHiddenLayers: model.num_hidden_layers,
    rank: loraRank,
    targetModulesPercent,
  })

  const adapterParamsBillion = adapterParameters.div(1e9)

  // 2. Base model weights in NF4 (4-bit, always)
  const baseWeights = calculateModelWeightVRAM(model.num_parameters_billion, 'nf4')

  // 3. Adapter weights in FP16 (always, 2 bytes)
  const adapterWeights = adapterParameters.mul(2).div(BYTES_PER_GB)

  // 4. Master weights in FP32 (always, for mixed precision stability)
  const masterWeights = adapterParameters.mul(MASTER_WEIGHT_BYTES).div(BYTES_PER_GB)

  // 5. Gradients in FP16 (always, 2 bytes)
  const gradients = adapterParameters.mul(2).div(BYTES_PER_GB)

  // 6. Optimizer states (ONLY for adapters, always FP32)
  const optimizerStates = calculateOptimizerStateMemory(adapterParamsBillion.toNumber(), optimizer)

  // 7. Training activations (batch-dependent, with optional optimizations)
  let activations = calculateTrainingActivationMemory(model, batchSize, sequenceLength)

  // Apply memory optimizations to activations (multiplicative stacking)
  if (gradientCheckpointing) {
    activations = applyGradientCheckpointing(activations, true)
  }
  if (flashAttention) {
    activations = applyFlashAttention(activations, sequenceLength, true)
  }

  // 8. Framework overhead
  const frameworkOverhead = TRAINING_FRAMEWORK_OVERHEAD_GB

  // 9. Total VRAM
  const total = baseWeights
    .add(adapterWeights)
    .add(masterWeights)
    .add(gradients)
    .add(optimizerStates)
    .add(activations)
    .add(frameworkOverhead)

  return {
    modelWeights: baseWeights, // For interface compatibility
    masterWeights,
    gradients,
    optimizerStates,
    activations,
    frameworkOverhead,
    total,
    trainableParameters: adapterParamsBillion,
    totalParameters: new Decimal(model.num_parameters_billion),
    method: 'qlora',
    baseWeights,
    adapterWeights,
    adapterParameters: adapterParamsBillion,
  }
}
