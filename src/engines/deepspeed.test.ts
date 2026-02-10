import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { calculateCPUOffloadMemory, calculateZeROMemoryPerGPU } from './deepspeed'
import type { TrainingVRAMBreakdown } from './types'

/**
 * Test suite for DeepSpeed ZeRO memory calculation engine
 *
 * Validates:
 * - ZeRO-1/2/3 partitioning logic (NOT simple divide-by-N)
 * - Single GPU passthrough
 * - CPU offload calculations
 */

describe('calculateZeROMemoryPerGPU', () => {
  // Fixture: Single-GPU training breakdown for 7B model
  const singleGPUBreakdown: TrainingVRAMBreakdown = {
    modelWeights: new Decimal(14),
    masterWeights: new Decimal(28),
    gradients: new Decimal(14),
    optimizerStates: new Decimal(56),
    activations: new Decimal(5),
    frameworkOverhead: new Decimal(1.5),
    total: new Decimal(118.5),
    trainableParameters: new Decimal(7),
    totalParameters: new Decimal(7),
    method: 'full',
  }

  describe('ZeRO-1: Partition optimizer states only', () => {
    it('should partition optimizer states and activations across 4 GPUs', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 4, 'zero-1')

      expect(result.numGPUs).toBe(4)
      expect(result.zeroStage).toBe('zero-1')

      // Model weights, master weights, gradients: REPLICATED
      expect(result.perGPU.modelWeights.toNumber()).toBe(14)
      expect(result.perGPU.masterWeights.toNumber()).toBe(28)
      expect(result.perGPU.gradients.toNumber()).toBe(14)

      // Optimizer states: PARTITIONED (56 / 4 = 14)
      expect(result.perGPU.optimizerStates.toNumber()).toBe(14)

      // Activations: PARTITIONED (5 / 4 = 1.25)
      expect(result.perGPU.activations.toNumber()).toBe(1.25)

      // Framework overhead: unchanged
      expect(result.perGPU.frameworkOverhead.toNumber()).toBe(1.5)

      // Total: 14 + 28 + 14 + 14 + 1.25 + 1.5 = 72.75
      expect(result.perGPU.total.toNumber()).toBeCloseTo(72.75, 2)

      // Reduction factor: 118.5 / 72.75 ≈ 1.63
      expect(result.reductionFactor.toNumber()).toBeCloseTo(1.63, 2)
    })

    it('should partition optimizer states and activations across 2 GPUs', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 2, 'zero-1')

      expect(result.numGPUs).toBe(2)
      expect(result.zeroStage).toBe('zero-1')

      // Replicated: modelWeights, masterWeights, gradients
      expect(result.perGPU.modelWeights.toNumber()).toBe(14)
      expect(result.perGPU.masterWeights.toNumber()).toBe(28)
      expect(result.perGPU.gradients.toNumber()).toBe(14)

      // Partitioned: optimizerStates (56 / 2 = 28), activations (5 / 2 = 2.5)
      expect(result.perGPU.optimizerStates.toNumber()).toBe(28)
      expect(result.perGPU.activations.toNumber()).toBe(2.5)

      // Total: 14 + 28 + 14 + 28 + 2.5 + 1.5 = 88
      expect(result.perGPU.total.toNumber()).toBeCloseTo(88, 2)
    })
  })

  describe('ZeRO-2: Partition optimizer states + gradients', () => {
    it('should partition optimizer states, gradients, and activations across 4 GPUs', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 4, 'zero-2')

      expect(result.numGPUs).toBe(4)
      expect(result.zeroStage).toBe('zero-2')

      // Model weights, master weights: REPLICATED
      expect(result.perGPU.modelWeights.toNumber()).toBe(14)
      expect(result.perGPU.masterWeights.toNumber()).toBe(28)

      // Gradients: PARTITIONED (14 / 4 = 3.5)
      expect(result.perGPU.gradients.toNumber()).toBe(3.5)

      // Optimizer states: PARTITIONED (56 / 4 = 14)
      expect(result.perGPU.optimizerStates.toNumber()).toBe(14)

      // Activations: PARTITIONED (5 / 4 = 1.25)
      expect(result.perGPU.activations.toNumber()).toBe(1.25)

      // Framework overhead: unchanged
      expect(result.perGPU.frameworkOverhead.toNumber()).toBe(1.5)

      // Total: 14 + 28 + 3.5 + 14 + 1.25 + 1.5 = 62.25
      expect(result.perGPU.total.toNumber()).toBeCloseTo(62.25, 2)

      // Reduction factor: 118.5 / 62.25 ≈ 1.90
      expect(result.reductionFactor.toNumber()).toBeCloseTo(1.9, 2)
    })
  })

  describe('ZeRO-3: Partition all training state', () => {
    it('should partition all components with 15% framework overhead increase', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 4, 'zero-3')

      expect(result.numGPUs).toBe(4)
      expect(result.zeroStage).toBe('zero-3')

      // ALL components PARTITIONED
      // Model weights: 14 / 4 = 3.5
      expect(result.perGPU.modelWeights.toNumber()).toBe(3.5)

      // Master weights: 28 / 4 = 7
      expect(result.perGPU.masterWeights.toNumber()).toBe(7)

      // Gradients: 14 / 4 = 3.5
      expect(result.perGPU.gradients.toNumber()).toBe(3.5)

      // Optimizer states: 56 / 4 = 14
      expect(result.perGPU.optimizerStates.toNumber()).toBe(14)

      // Activations: 5 / 4 = 1.25
      expect(result.perGPU.activations.toNumber()).toBe(1.25)

      // Framework overhead: 1.5 * 1.15 = 1.725 (15% extra for parameter gather)
      expect(result.perGPU.frameworkOverhead.toNumber()).toBeCloseTo(1.725, 3)

      // Total: 3.5 + 7 + 3.5 + 14 + 1.25 + 1.725 = 30.975
      expect(result.perGPU.total.toNumber()).toBeCloseTo(30.975, 2)

      // Reduction factor: 118.5 / 30.975 ≈ 3.83
      expect(result.reductionFactor.toNumber()).toBeCloseTo(3.83, 2)
    })
  })

  describe('Single GPU passthrough', () => {
    it('should return original breakdown unchanged for 1 GPU', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 1, 'zero-1')

      expect(result.numGPUs).toBe(1)
      expect(result.zeroStage).toBe('zero-1')

      // All values unchanged
      expect(result.perGPU.modelWeights.toNumber()).toBe(14)
      expect(result.perGPU.masterWeights.toNumber()).toBe(28)
      expect(result.perGPU.gradients.toNumber()).toBe(14)
      expect(result.perGPU.optimizerStates.toNumber()).toBe(56)
      expect(result.perGPU.activations.toNumber()).toBe(5)
      expect(result.perGPU.frameworkOverhead.toNumber()).toBe(1.5)
      expect(result.perGPU.total.toNumber()).toBe(118.5)

      // No reduction (factor = 1)
      expect(result.reductionFactor.toNumber()).toBe(1)
    })

    it('should return original breakdown for ZeRO-3 with 1 GPU', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 1, 'zero-3')

      expect(result.numGPUs).toBe(1)
      expect(result.perGPU.total.toNumber()).toBe(118.5)
      expect(result.reductionFactor.toNumber()).toBe(1)
    })
  })

  describe('Edge cases', () => {
    it('should handle zero master weights (FP32 training)', () => {
      const fp32Breakdown: TrainingVRAMBreakdown = {
        ...singleGPUBreakdown,
        masterWeights: new Decimal(0),
        total: new Decimal(90.5),
      }

      const result = calculateZeROMemoryPerGPU(fp32Breakdown, 4, 'zero-2')

      expect(result.perGPU.masterWeights.toNumber()).toBe(0)
      // Should still partition other components correctly
      expect(result.perGPU.gradients.toNumber()).toBe(3.5)
      expect(result.perGPU.optimizerStates.toNumber()).toBe(14)
    })

    it('should handle large GPU counts', () => {
      const result = calculateZeROMemoryPerGPU(singleGPUBreakdown, 8, 'zero-3')

      expect(result.numGPUs).toBe(8)
      // All components divided by 8
      expect(result.perGPU.modelWeights.toNumber()).toBe(14 / 8)
      expect(result.perGPU.optimizerStates.toNumber()).toBe(56 / 8)
    })
  })
})

describe('calculateCPUOffloadMemory', () => {
  const breakdown: TrainingVRAMBreakdown = {
    modelWeights: new Decimal(14),
    masterWeights: new Decimal(28),
    gradients: new Decimal(14),
    optimizerStates: new Decimal(56),
    activations: new Decimal(5),
    frameworkOverhead: new Decimal(1.5),
    total: new Decimal(118.5),
    trainableParameters: new Decimal(7),
    totalParameters: new Decimal(7),
    method: 'full',
  }

  it('should offload optimizer states only', () => {
    const result = calculateCPUOffloadMemory(breakdown, {
      offloadOptimizer: true,
      offloadParameters: false,
    })

    // GPU memory = total - optimizerStates = 118.5 - 56 = 62.5
    expect(result.gpuMemory.toNumber()).toBeCloseTo(62.5, 2)

    // CPU memory = optimizerStates = 56
    expect(result.cpuMemory.toNumber()).toBe(56)
  })

  it('should offload optimizer states and parameters', () => {
    const result = calculateCPUOffloadMemory(breakdown, {
      offloadOptimizer: true,
      offloadParameters: true,
    })

    // GPU memory = total - optimizerStates - modelWeights = 118.5 - 56 - 14 = 48.5
    expect(result.gpuMemory.toNumber()).toBeCloseTo(48.5, 2)

    // CPU memory = optimizerStates + modelWeights = 56 + 14 = 70
    expect(result.cpuMemory.toNumber()).toBe(70)
  })

  it('should not offload when both flags are false', () => {
    const result = calculateCPUOffloadMemory(breakdown, {
      offloadOptimizer: false,
      offloadParameters: false,
    })

    // GPU memory unchanged
    expect(result.gpuMemory.toNumber()).toBe(118.5)

    // CPU memory = 0
    expect(result.cpuMemory.toNumber()).toBe(0)
  })

  it('should offload parameters only', () => {
    const result = calculateCPUOffloadMemory(breakdown, {
      offloadOptimizer: false,
      offloadParameters: true,
    })

    // GPU memory = total - modelWeights = 118.5 - 14 = 104.5
    expect(result.gpuMemory.toNumber()).toBeCloseTo(104.5, 2)

    // CPU memory = modelWeights = 14
    expect(result.cpuMemory.toNumber()).toBe(14)
  })

  it('should handle ZeRO-partitioned breakdown', () => {
    const zeroBreakdown: TrainingVRAMBreakdown = {
      ...breakdown,
      optimizerStates: new Decimal(14), // Already partitioned (56 / 4)
      total: new Decimal(76.5),
    }

    const result = calculateCPUOffloadMemory(zeroBreakdown, {
      offloadOptimizer: true,
      offloadParameters: false,
    })

    // GPU memory = 76.5 - 14 = 62.5
    expect(result.gpuMemory.toNumber()).toBeCloseTo(62.5, 2)

    // CPU memory = 14 (partitioned optimizer states)
    expect(result.cpuMemory.toNumber()).toBe(14)
  })
})
