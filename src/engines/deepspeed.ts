import Decimal from 'decimal.js'
import type { CPUOffloadConfig, ZeROStage } from './frameworks'
import type { TrainingVRAMBreakdown, ZeROResult } from './types'

/**
 * DeepSpeed ZeRO memory calculation engine
 *
 * CRITICAL: ZeRO stages are NOT simple divide-by-N. Each stage has specific
 * partitioning rules for which components are sharded vs replicated:
 *
 * - ZeRO-1: Partition optimizer states only (2x reduction)
 * - ZeRO-2: Partition optimizer states + gradients (4x reduction)
 * - ZeRO-3: Partition ALL training state (8x reduction with gather overhead)
 *
 * Reference: .planning/phases/10-framework-presets-multi-gpu-training/10-RESEARCH.md
 * Paper: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
 * https://arxiv.org/abs/1910.02054
 */

/**
 * Calculate per-GPU memory after applying DeepSpeed ZeRO partitioning
 *
 * Takes a single-GPU training breakdown and partitions memory across GPUs
 * according to the specified ZeRO stage. Different stages partition different
 * components:
 *
 * ZeRO-1 (Pos - Optimizer State Partitioning):
 * - Partitioned: optimizer states, activations
 * - Replicated: model weights, master weights, gradients
 * - Expected reduction: ~2x
 *
 * ZeRO-2 (Pos+g - Optimizer State + Gradient Partitioning):
 * - Partitioned: optimizer states, gradients, activations
 * - Replicated: model weights, master weights
 * - Expected reduction: ~4x
 *
 * ZeRO-3 (Pos+g+p - Full Training State Partitioning):
 * - Partitioned: ALL components (model weights, master weights, gradients, optimizer states, activations)
 * - Framework overhead increased by 15% for parameter gather/scatter
 * - Expected reduction: ~8x
 *
 * Single GPU passthrough:
 * - Returns original breakdown unchanged for 1 GPU (no partitioning needed)
 *
 * @param singleGPU - Single-GPU training memory breakdown
 * @param numGPUs - Number of GPUs to partition across
 * @param zeroStage - ZeRO optimization stage
 * @returns Per-GPU memory breakdown with reduction factor
 */
export function calculateZeROMemoryPerGPU(
  singleGPU: TrainingVRAMBreakdown,
  numGPUs: number,
  zeroStage: ZeROStage,
): ZeROResult {
  // Single GPU passthrough: no partitioning needed
  if (numGPUs === 1) {
    return {
      perGPU: {
        modelWeights: singleGPU.modelWeights,
        masterWeights: singleGPU.masterWeights,
        gradients: singleGPU.gradients,
        optimizerStates: singleGPU.optimizerStates,
        activations: singleGPU.activations,
        frameworkOverhead: singleGPU.frameworkOverhead,
        total: singleGPU.total,
      },
      reductionFactor: new Decimal(1),
      numGPUs: 1,
      zeroStage,
    }
  }

  const n = new Decimal(numGPUs)

  let perGPU: ZeROResult['perGPU']

  switch (zeroStage) {
    case 'zero-1': {
      // ZeRO-1: Partition optimizer states and activations only
      // Model weights, master weights, gradients are replicated
      perGPU = {
        modelWeights: singleGPU.modelWeights, // replicated
        masterWeights: singleGPU.masterWeights, // replicated
        gradients: singleGPU.gradients, // replicated
        optimizerStates: singleGPU.optimizerStates.div(n), // partitioned
        activations: singleGPU.activations.div(n), // partitioned
        frameworkOverhead: singleGPU.frameworkOverhead, // unchanged
        total: new Decimal(0), // calculated below
      }
      break
    }

    case 'zero-2': {
      // ZeRO-2: Partition optimizer states, gradients, and activations
      // Model weights and master weights are replicated
      perGPU = {
        modelWeights: singleGPU.modelWeights, // replicated
        masterWeights: singleGPU.masterWeights, // replicated
        gradients: singleGPU.gradients.div(n), // partitioned
        optimizerStates: singleGPU.optimizerStates.div(n), // partitioned
        activations: singleGPU.activations.div(n), // partitioned
        frameworkOverhead: singleGPU.frameworkOverhead, // unchanged
        total: new Decimal(0), // calculated below
      }
      break
    }

    case 'zero-3': {
      // ZeRO-3: Partition ALL training state
      // Framework overhead increased by 15% for parameter gather/scatter
      const frameworkOverheadMultiplier = new Decimal(1.15)
      perGPU = {
        modelWeights: singleGPU.modelWeights.div(n), // partitioned
        masterWeights: singleGPU.masterWeights.div(n), // partitioned
        gradients: singleGPU.gradients.div(n), // partitioned
        optimizerStates: singleGPU.optimizerStates.div(n), // partitioned
        activations: singleGPU.activations.div(n), // partitioned
        frameworkOverhead: singleGPU.frameworkOverhead.mul(frameworkOverheadMultiplier), // 15% extra
        total: new Decimal(0), // calculated below
      }
      break
    }
  }

  // Calculate total per-GPU memory
  perGPU.total = perGPU.modelWeights
    .add(perGPU.masterWeights)
    .add(perGPU.gradients)
    .add(perGPU.optimizerStates)
    .add(perGPU.activations)
    .add(perGPU.frameworkOverhead)

  // Calculate reduction factor
  const reductionFactor = singleGPU.total.div(perGPU.total)

  return {
    perGPU,
    reductionFactor,
    numGPUs,
    zeroStage,
  }
}

/**
 * Calculate memory split between GPU and CPU when offloading is enabled
 *
 * DeepSpeed supports offloading optimizer states and/or model parameters
 * from GPU VRAM to CPU RAM, reducing GPU memory requirements at the cost
 * of training speed (CPU-GPU data transfer overhead).
 *
 * Typical use cases:
 * - Offload optimizer only: Reduces GPU memory by ~50% (optimizer states are typically half of training memory)
 * - Offload optimizer + params: Reduces GPU memory by ~60-70% (keeps only gradients + activations on GPU)
 *
 * Note: This operates on a SINGLE GPU's breakdown. Apply this AFTER ZeRO partitioning
 * if using both ZeRO and CPU offload.
 *
 * @param breakdown - Training memory breakdown (single GPU or ZeRO-partitioned)
 * @param config - CPU offload configuration
 * @returns Memory split between GPU and CPU
 */
export function calculateCPUOffloadMemory(
  breakdown: TrainingVRAMBreakdown,
  config: CPUOffloadConfig,
): { gpuMemory: Decimal; cpuMemory: Decimal } {
  let gpuMemory = breakdown.total
  let cpuMemory = new Decimal(0)

  // Offload optimizer states to CPU
  if (config.offloadOptimizer) {
    gpuMemory = gpuMemory.sub(breakdown.optimizerStates)
    cpuMemory = cpuMemory.add(breakdown.optimizerStates)
  }

  // Offload model parameters to CPU
  if (config.offloadParameters) {
    gpuMemory = gpuMemory.sub(breakdown.modelWeights)
    cpuMemory = cpuMemory.add(breakdown.modelWeights)
  }

  return { gpuMemory, cpuMemory }
}
