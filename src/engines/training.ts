import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import {
  BYTES_PER_GB,
  GRADIENT_BYTES,
  MASTER_WEIGHT_BYTES,
  OPTIMIZER_STATE_BYTES,
  TRAINING_FRAMEWORK_OVERHEAD_GB,
  WEIGHT_BYTES,
} from './constants'
import type { OptimizerType, TrainingPrecision, TrainingVRAMBreakdown } from './types'

/**
 * Calculate optimizer state memory for training
 *
 * CRITICAL: Optimizer states are ALWAYS stored in FP32 for numerical stability,
 * even during mixed precision (FP16/BF16) training. This is independent of the
 * trainingPrecision parameter.
 *
 * Memory by optimizer type:
 * - AdamW: 8 bytes/param (2 FP32 states: momentum + variance)
 * - SGD-momentum: 4 bytes/param (1 FP32 state: momentum)
 * - AdamW-8bit: 2 bytes/param (2 8-bit quantized states)
 * - Adafactor: 4 bytes/param (factored approximation)
 *
 * For LoRA/QLoRA: Only applies to adapter parameters, NOT frozen base weights.
 * Example: 70B model with 1% LoRA adapters → 0.7B trainable params → 5.6GB optimizer states (AdamW)
 *
 * @param trainableParamsBillion - Number of trainable parameters in billions
 * @param optimizer - Optimizer type
 * @returns Optimizer state memory in GB as Decimal
 *
 * @example
 * ```ts
 * // 7B model, full fine-tuning with AdamW
 * calculateOptimizerStateMemory(7.0, 'adamw')
 * // Returns: ~52.15 GB (7B * 8 bytes / 1024^3)
 *
 * // 7B model with 1% LoRA adapters (~0.07B trainable)
 * calculateOptimizerStateMemory(0.07, 'adamw')
 * // Returns: ~0.52 GB (only adapters have optimizer states)
 * ```
 *
 * Reference: .planning/research/PITFALLS.md #2 (30-60% underestimation from missing this)
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export function calculateOptimizerStateMemory(
  trainableParamsBillion: number,
  optimizer: OptimizerType,
): Decimal {
  const totalParams = new Decimal(trainableParamsBillion).mul(1e9)
  const bytesPerParam = OPTIMIZER_STATE_BYTES[optimizer]
  return totalParams.mul(bytesPerParam).div(BYTES_PER_GB)
}

/**
 * Calculate training activation memory
 *
 * Training activations are fundamentally different from inference activations:
 * - Must store full attention matrices O(N^2) for backward pass
 * - Include intermediate gradients and activations for all layers
 * - Scale linearly with batch size and sequence length
 *
 * Formula: batchSize * sequenceLength * hiddenSize * numLayers * 10 * 2 / BYTES_PER_GB
 * - Factor of 10: approximate multiplier per layer (attention QKV + MLP + residuals)
 * - Factor of 2: bytes per element (FP16/BF16 activations)
 *
 * For MoE models: scales with active parameter ratio (only active experts contribute)
 *
 * IMPORTANT: This is NOT the same as inference activation memory. Do NOT reuse
 * calculateActivationMemory from inference.ts.
 *
 * @param model - Model configuration
 * @param batchSize - Number of concurrent training sequences
 * @param sequenceLength - Maximum sequence length
 * @returns Activation memory in GB as Decimal
 *
 * @example
 * ```ts
 * // 7B dense model (hidden=4096, layers=32), batch=1, seq=2048
 * calculateTrainingActivationMemory(llama7b, 1, 2048)
 * // Returns: ~5.0 GB
 *
 * // Same model, batch=4 (linear scaling)
 * calculateTrainingActivationMemory(llama7b, 4, 2048)
 * // Returns: ~20.0 GB
 *
 * // MoE model (uses active param ratio)
 * calculateTrainingActivationMemory(mixtral, 1, 2048)
 * // Returns: scaled by active experts
 * ```
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 * Reference: .planning/research/PITFALLS.md #1 (training vs inference activation memory)
 */
export function calculateTrainingActivationMemory(
  model: Model,
  batchSize: number,
  sequenceLength: number,
): Decimal {
  let effectiveHiddenSize = model.hidden_size

  // For MoE models, scale hidden size by active parameter ratio
  if (model.architecture === 'moe' && model.num_experts && model.num_experts_per_token) {
    const activeRatio = model.num_experts_per_token / model.num_experts
    // Approximate: 20% shared, 80% in experts
    const effectiveActiveRatio = 0.2 + 0.8 * activeRatio
    effectiveHiddenSize = Math.floor(model.hidden_size * effectiveActiveRatio)
  }

  // Activation memory: batch * seq * hidden * layers * 10 * 2 (FP16/BF16 bytes)
  const activationBytes = new Decimal(batchSize)
    .mul(sequenceLength)
    .mul(effectiveHiddenSize)
    .mul(model.num_layers)
    .mul(10) // Approximate multiplier per layer
    .mul(2) // FP16/BF16 bytes per activation

  return activationBytes.div(BYTES_PER_GB)
}

/**
 * Calculate full fine-tuning VRAM breakdown
 *
 * Computes complete VRAM requirements for full model fine-tuning including:
 * 1. Model weights at training precision
 * 2. FP32 master weights (for mixed precision only)
 * 3. Gradients at training precision
 * 4. Optimizer states (always FP32)
 * 5. Training activations (batch-dependent)
 * 6. Framework overhead (PyTorch + CUDA + autograd)
 *
 * Memory formula:
 * Total = ModelWeights + MasterWeights + Gradients + OptimizerStates + Activations + Overhead
 *
 * CRITICAL: Mixed precision (FP16/BF16) requires FP32 master weights for numerical
 * stability. Pure FP32 training does NOT need master weights (they ARE the weights).
 *
 * @param params - Training configuration
 * @param params.model - Model to fine-tune
 * @param params.trainingPrecision - Weight precision (fp32/fp16/bf16)
 * @param params.optimizer - Optimizer type
 * @param params.batchSize - Training batch size
 * @param params.sequenceLength - Training sequence length
 * @returns Complete VRAM breakdown with all components in GB
 *
 * @example
 * ```ts
 * // 7B model, BF16 mixed precision, AdamW optimizer
 * const breakdown = calculateFullFineTuningVRAM({
 *   model: llama7b,
 *   trainingPrecision: 'bf16',
 *   optimizer: 'adamw',
 *   batchSize: 1,
 *   sequenceLength: 2048,
 * })
 * // breakdown.modelWeights: ~13.04 GB (7B * 2 bytes)
 * // breakdown.masterWeights: ~26.08 GB (7B * 4 bytes, FP32 master copy)
 * // breakdown.gradients: ~13.04 GB (7B * 2 bytes)
 * // breakdown.optimizerStates: ~52.15 GB (7B * 8 bytes, FP32 AdamW)
 * // breakdown.activations: ~5.0 GB
 * // breakdown.frameworkOverhead: 1.5 GB
 * // breakdown.total: ~110.81 GB
 *
 * // FP32 training (no master weights)
 * const fp32Breakdown = calculateFullFineTuningVRAM({
 *   model: llama7b,
 *   trainingPrecision: 'fp32',
 *   optimizer: 'adamw',
 *   batchSize: 1,
 *   sequenceLength: 2048,
 * })
 * // fp32Breakdown.masterWeights: 0 GB (no master copy needed)
 * ```
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 * Reference: .planning/research/PITFALLS.md #2 (optimizer states always FP32)
 */
export function calculateFullFineTuningVRAM(params: {
  model: Model
  trainingPrecision: TrainingPrecision
  optimizer: OptimizerType
  batchSize: number
  sequenceLength: number
}): TrainingVRAMBreakdown {
  const { model, trainingPrecision, optimizer, batchSize, sequenceLength } = params

  const paramsBillion = model.num_parameters_billion
  const totalParams = new Decimal(paramsBillion).mul(1e9)

  // 1. Model weights at training precision
  const modelWeights = totalParams.mul(WEIGHT_BYTES[trainingPrecision]).div(BYTES_PER_GB)

  // 2. Master weights (FP32 copy for mixed precision, zero for pure FP32)
  const masterWeights =
    trainingPrecision === 'fp32'
      ? new Decimal(0)
      : totalParams.mul(MASTER_WEIGHT_BYTES).div(BYTES_PER_GB)

  // 3. Gradients at training precision
  const gradients = totalParams.mul(GRADIENT_BYTES[trainingPrecision]).div(BYTES_PER_GB)

  // 4. Optimizer states (always FP32)
  const optimizerStates = calculateOptimizerStateMemory(paramsBillion, optimizer)

  // 5. Training activations (batch-dependent)
  const activations = calculateTrainingActivationMemory(model, batchSize, sequenceLength)

  // 6. Framework overhead (PyTorch + CUDA + autograd)
  const frameworkOverhead = TRAINING_FRAMEWORK_OVERHEAD_GB

  // 7. Total VRAM
  const total = modelWeights
    .add(masterWeights)
    .add(gradients)
    .add(optimizerStates)
    .add(activations)
    .add(frameworkOverhead)

  return {
    modelWeights,
    masterWeights,
    gradients,
    optimizerStates,
    activations,
    frameworkOverhead,
    total,
    trainableParameters: new Decimal(paramsBillion),
    totalParameters: new Decimal(paramsBillion),
    method: 'full',
  }
}
