import type Decimal from 'decimal.js'
import {
  CHECKPOINTING_RETENTION_FACTOR,
  FLASH_ATTENTION_LONG_RETENTION,
  FLASH_ATTENTION_LONG_THRESHOLD,
  FLASH_ATTENTION_MEDIUM_RETENTION,
  FLASH_ATTENTION_SHORT_RETENTION,
  FLASH_ATTENTION_SHORT_THRESHOLD,
} from './constants'

/**
 * Calculate effective batch size for gradient accumulation
 *
 * Gradient accumulation allows training with larger effective batch sizes
 * by accumulating gradients across multiple forward/backward passes before
 * updating weights. The effective batch size determines convergence properties
 * and is the key metric for training reproducibility.
 *
 * Formula: effectiveBatch = perDeviceBatch * accumulationSteps * numGPUs
 *
 * @param perDeviceBatchSize - Batch size per GPU device
 * @param gradientAccumulationSteps - Number of steps to accumulate gradients
 * @param numGPUs - Number of GPUs (defaults to 1)
 * @returns Effective batch size as integer
 *
 * @example
 * ```ts
 * // Single GPU, batch=2, 4 accumulation steps -> 8
 * calculateEffectiveBatchSize(2, 4, 1)
 *
 * // 4 GPUs, batch=2, 4 accumulation steps -> 32
 * calculateEffectiveBatchSize(2, 4, 4)
 * ```
 *
 * Reference: .planning/phases/08-memory-optimization-features/08-RESEARCH.md
 */
export function calculateEffectiveBatchSize(
  perDeviceBatchSize: number,
  gradientAccumulationSteps: number,
  numGPUs: number = 1,
): number {
  return perDeviceBatchSize * gradientAccumulationSteps * numGPUs
}

/**
 * Apply gradient checkpointing (activation checkpointing) reduction
 *
 * Gradient checkpointing trades compute for memory by storing only a subset
 * of activations during forward pass and recomputing the rest during backward.
 * This typically reduces activation memory by 60% (retains 40%) with ~20-30%
 * compute overhead.
 *
 * When enabled, activations are multiplied by CHECKPOINTING_RETENTION_FACTOR (0.4).
 * When disabled, returns input unchanged (identity function).
 *
 * @param activationMemoryGB - Base activation memory in GB (as Decimal)
 * @param enabled - Whether gradient checkpointing is enabled
 * @returns Optimized activation memory in GB (as Decimal)
 *
 * @example
 * ```ts
 * // 10 GB activations with checkpointing -> 4 GB (60% reduction)
 * applyGradientCheckpointing(new Decimal(10.0), true)
 *
 * // 10 GB activations without checkpointing -> 10 GB (identity)
 * applyGradientCheckpointing(new Decimal(10.0), false)
 * ```
 *
 * Reference: .planning/phases/08-memory-optimization-features/08-RESEARCH.md
 */
export function applyGradientCheckpointing(activationMemoryGB: Decimal, enabled: boolean): Decimal {
  if (!enabled) {
    return activationMemoryGB
  }

  return activationMemoryGB.mul(CHECKPOINTING_RETENTION_FACTOR)
}

/**
 * Apply Flash Attention memory reduction
 *
 * Flash Attention reduces memory by fusing attention operations and avoiding
 * materialization of large QK^T attention matrices. Memory savings scale with
 * sequence length due to quadratic attention memory O(N^2).
 *
 * Retention factors (fraction of activation memory kept):
 * - Short (<2048 tokens): 0.85 (15% reduction) - minimal benefit
 * - Medium (2048-8191 tokens): 0.5 (50% reduction) - moderate benefit
 * - Long (>=8192 tokens): 0.3 (70% reduction) - large benefit
 *
 * When disabled, returns input unchanged (identity function).
 *
 * @param activationMemoryGB - Base activation memory in GB (as Decimal)
 * @param sequenceLength - Training sequence length in tokens
 * @param enabled - Whether Flash Attention is enabled
 * @returns Optimized activation memory in GB (as Decimal)
 *
 * @example
 * ```ts
 * // 10 GB activations, 1024 seq (short) -> 8.5 GB (15% reduction)
 * applyFlashAttention(new Decimal(10.0), 1024, true)
 *
 * // 10 GB activations, 4096 seq (medium) -> 5.0 GB (50% reduction)
 * applyFlashAttention(new Decimal(10.0), 4096, true)
 *
 * // 10 GB activations, 16384 seq (long) -> 3.0 GB (70% reduction)
 * applyFlashAttention(new Decimal(10.0), 16384, true)
 * ```
 *
 * Reference: .planning/phases/08-memory-optimization-features/08-RESEARCH.md
 */
export function applyFlashAttention(
  activationMemoryGB: Decimal,
  sequenceLength: number,
  enabled: boolean,
): Decimal {
  if (!enabled) {
    return activationMemoryGB
  }

  // Determine retention factor based on sequence length
  let retentionFactor: Decimal
  if (sequenceLength < FLASH_ATTENTION_SHORT_THRESHOLD) {
    retentionFactor = FLASH_ATTENTION_SHORT_RETENTION
  } else if (sequenceLength < FLASH_ATTENTION_LONG_THRESHOLD) {
    retentionFactor = FLASH_ATTENTION_MEDIUM_RETENTION
  } else {
    retentionFactor = FLASH_ATTENTION_LONG_RETENTION
  }

  return activationMemoryGB.mul(retentionFactor)
}
