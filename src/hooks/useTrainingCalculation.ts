/**
 * React hook for training VRAM calculations
 *
 * Synchronously calculates training VRAM requirements based on store configuration.
 * Unlike inference calculations, training calculations are fast enough to run
 * synchronously without Web Workers.
 *
 * Usage:
 * ```tsx
 * const { result, error, zeroResult, cpuOffload } = useTrainingCalculation()
 *
 * if (error) return <ErrorMessage error={error} />
 * if (!result) return <NoSelection />
 *
 * return <TrainingBreakdown breakdown={result} zeroResult={zeroResult} cpuOffload={cpuOffload} />
 * ```
 */

import { calculateCPUOffloadMemory, calculateZeROMemoryPerGPU } from '@engines/deepspeed'
import { FRAMEWORK_PRESETS, type ZeROStage } from '@engines/frameworks'
import { calculateLoRAFineTuningVRAM, calculateQLoRAFineTuningVRAM } from '@engines/lora'
import { calculateFullFineTuningVRAM } from '@engines/training'
import type { LoRAVRAMBreakdown, TrainingVRAMBreakdown, ZeROResult } from '@engines/types'
import { useUIStore } from '@store/uiStore'
import type Decimal from 'decimal.js'
import { useMemo } from 'react'

/**
 * Hook return type
 */
export interface UseTrainingCalculationResult {
  /** Training VRAM breakdown (null if no model selected) */
  result: TrainingVRAMBreakdown | LoRAVRAMBreakdown | null
  /** Error message (null if no error) */
  error: string | null
  /** ZeRO per-GPU breakdown (null if not using DeepSpeed multi-GPU) */
  zeroResult: ZeROResult | null
  /** CPU offload info (null if not enabled) */
  cpuOffload: {
    gpuMemory: Decimal
    cpuMemory: Decimal
  } | null
}

/**
 * Calculate training VRAM requirements from Zustand store state
 *
 * Reads training configuration from the UI store and calls the appropriate
 * training engine function based on the selected fine-tuning method.
 *
 * Supports all three training methods:
 * - Full fine-tuning: All parameters trainable
 * - LoRA: Low-rank adapters only
 * - QLoRA: 4-bit base + FP16 adapters
 *
 * Memory optimizations are applied when enabled:
 * - gradientCheckpointing: 60% activation memory reduction
 * - flashAttention: 15-70% reduction (sequence-dependent)
 *
 * @returns Hook state with result and error
 *
 * @example
 * ```tsx
 * function TrainingResults() {
 *   const { result, error } = useTrainingCalculation()
 *
 *   if (error) return <div>Error: {error}</div>
 *   if (!result) return <div>Select a model to begin</div>
 *
 *   return (
 *     <div>
 *       <h3>Training VRAM: {result.total.toFixed(2)} GB</h3>
 *       <p>Method: {result.method}</p>
 *       <p>Trainable: {result.trainableParameters.toFixed(3)}B params</p>
 *     </div>
 *   )
 * }
 * ```
 */
export function useTrainingCalculation(): UseTrainingCalculationResult {
  // Read all training configuration from store
  const {
    selectedModel,
    trainingMethod,
    optimizer,
    trainingPrecision,
    batchSize,
    sequenceLength,
    loraRank,
    targetModulesPercent,
    gradientAccumulationSteps: _gradientAccumulationSteps,
    gradientCheckpointing,
    flashAttention,
    frameworkPreset,
    numGPUs,
    cpuOffloadOptimizer,
  } = useUIStore()

  // Calculate training VRAM with useMemo (memoized by dependencies)
  const { result, error, zeroResult, cpuOffload } = useMemo<UseTrainingCalculationResult>(() => {
    // Early return if no model selected
    if (!selectedModel) {
      return { result: null, error: null, zeroResult: null, cpuOffload: null }
    }

    try {
      let breakdown: TrainingVRAMBreakdown | LoRAVRAMBreakdown

      // Branch on training method
      switch (trainingMethod) {
        case 'full':
          breakdown = calculateFullFineTuningVRAM({
            model: selectedModel,
            trainingPrecision,
            optimizer,
            batchSize,
            sequenceLength,
            gradientCheckpointing,
            flashAttention,
          })
          break

        case 'lora':
          breakdown = calculateLoRAFineTuningVRAM({
            model: selectedModel,
            trainingPrecision,
            optimizer,
            batchSize,
            sequenceLength,
            loraRank,
            targetModulesPercent,
            gradientCheckpointing,
            flashAttention,
          })
          break

        case 'qlora':
          breakdown = calculateQLoRAFineTuningVRAM({
            model: selectedModel,
            optimizer,
            batchSize,
            sequenceLength,
            loraRank,
            targetModulesPercent,
            gradientCheckpointing,
            flashAttention,
          })
          break

        default: {
          // TypeScript exhaustiveness check
          const _exhaustive: never = trainingMethod
          throw new Error(`Unknown training method: ${_exhaustive}`)
        }
      }

      // Derive ZeRO stage from framework preset
      const presetConfig = FRAMEWORK_PRESETS[frameworkPreset]
      const zeroStage: ZeROStage | null = presetConfig.zeroStage

      // Calculate ZeRO partitioning if using DeepSpeed with multiple GPUs
      let zeroResult: ZeROResult | null = null
      if (zeroStage && numGPUs > 1) {
        zeroResult = calculateZeROMemoryPerGPU(breakdown, numGPUs, zeroStage)
      }

      // Calculate CPU offload if enabled
      let cpuOffload: { gpuMemory: Decimal; cpuMemory: Decimal } | null = null
      if (cpuOffloadOptimizer) {
        // Apply to ZeRO per-GPU breakdown if available, otherwise single-GPU
        const sourceBreakdown: TrainingVRAMBreakdown = zeroResult
          ? {
              method: breakdown.method,
              totalParameters: breakdown.totalParameters,
              trainableParameters: breakdown.trainableParameters,
              modelWeights: zeroResult.perGPU.modelWeights,
              masterWeights: zeroResult.perGPU.masterWeights,
              gradients: zeroResult.perGPU.gradients,
              optimizerStates: zeroResult.perGPU.optimizerStates,
              activations: zeroResult.perGPU.activations,
              frameworkOverhead: zeroResult.perGPU.frameworkOverhead,
              total: zeroResult.perGPU.total,
            }
          : breakdown

        cpuOffload = calculateCPUOffloadMemory(sourceBreakdown, {
          offloadOptimizer: true,
          offloadParameters: false,
        })
      }

      return { result: breakdown, error: null, zeroResult, cpuOffload }
    } catch (err) {
      return {
        result: null,
        error: err instanceof Error ? err.message : String(err),
        zeroResult: null,
        cpuOffload: null,
      }
    }
  }, [
    selectedModel,
    trainingMethod,
    optimizer,
    trainingPrecision,
    batchSize,
    sequenceLength,
    loraRank,
    targetModulesPercent,
    gradientCheckpointing,
    flashAttention,
    frameworkPreset,
    numGPUs,
    cpuOffloadOptimizer,
  ])

  return { result, error, zeroResult, cpuOffload }
}
