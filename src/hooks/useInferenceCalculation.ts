/**
 * React hook for non-blocking inference calculations
 *
 * Offloads calculations to a Web Worker when available (browser environment).
 * Falls back to synchronous execution when Workers are unavailable (SSR, old browsers).
 *
 * Usage:
 * ```tsx
 * const { result, loading, error } = useInferenceCalculation(
 *   model,
 *   gpu,
 *   'gptq',
 *   4096,
 *   1,
 *   'fp16'
 * )
 * ```
 */

import type {
  InferenceVRAMBreakdown,
  KVCachePrecision,
  MultiGPUVRAMBreakdown,
  OffloadedVRAMBreakdown,
  OffloadingConfig,
  PerformanceEstimate,
  QuantizationFormat,
  ShardingStrategy,
} from '@engines/types'
import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { useEffect, useState } from 'react'

/**
 * Hook return type
 */
export interface UseInferenceCalculationResult {
  result: {
    vram: InferenceVRAMBreakdown
    performance: PerformanceEstimate
    offloading: OffloadedVRAMBreakdown | null
    multiGPU: MultiGPUVRAMBreakdown | null
    interconnectWarning: string | null
  } | null
  loading: boolean
  error: string | null
}

/**
 * Reconstruct InferenceVRAMBreakdown from serialized strings
 *
 * Web Workers serialize Decimal values to strings for structured cloning.
 * This function reconstructs Decimal objects from those strings.
 */
function reconstructVRAMBreakdown(serialized: {
  modelWeights: string
  kvCache: string
  activations: string
  frameworkOverhead: string
  total: string
}): InferenceVRAMBreakdown {
  return {
    modelWeights: new Decimal(serialized.modelWeights),
    kvCache: new Decimal(serialized.kvCache),
    activations: new Decimal(serialized.activations),
    frameworkOverhead: new Decimal(serialized.frameworkOverhead),
    total: new Decimal(serialized.total),
  }
}

/**
 * Reconstruct PerformanceEstimate from serialized strings
 */
function reconstructPerformanceEstimate(serialized: {
  tokensPerSecond: string
  timeToFirstToken: string
  isComputeBound: boolean
  isMemoryBound: boolean
  bottleneck: 'compute' | 'memory' | 'balanced'
}): PerformanceEstimate {
  return {
    tokensPerSecond: new Decimal(serialized.tokensPerSecond),
    timeToFirstToken: new Decimal(serialized.timeToFirstToken),
    isComputeBound: serialized.isComputeBound,
    isMemoryBound: serialized.isMemoryBound,
    bottleneck: serialized.bottleneck,
  }
}

/**
 * Reconstruct MultiGPUVRAMBreakdown from serialized strings
 *
 * Web Workers serialize Decimal values to strings for structured cloning.
 * This function reconstructs Decimal objects from those strings.
 */
function reconstructMultiGPUBreakdown(
  serialized: {
    numGPUs: number
    strategy: string
    perGPU: {
      modelWeights: string
      kvCache: string
      activations: string
      frameworkOverhead: string
      communicationOverhead: string
      total: string
    }
    replicatedMemory: string
    totalPerGPU: string
    utilizationPercent: string
    singleGPUBaseline: string
  } | null,
): MultiGPUVRAMBreakdown | null {
  if (!serialized) return null
  return {
    numGPUs: serialized.numGPUs,
    strategy: serialized.strategy as ShardingStrategy,
    perGPU: {
      modelWeights: new Decimal(serialized.perGPU.modelWeights),
      kvCache: new Decimal(serialized.perGPU.kvCache),
      activations: new Decimal(serialized.perGPU.activations),
      frameworkOverhead: new Decimal(serialized.perGPU.frameworkOverhead),
      communicationOverhead: new Decimal(serialized.perGPU.communicationOverhead),
      total: new Decimal(serialized.perGPU.total),
    },
    replicatedMemory: new Decimal(serialized.replicatedMemory),
    totalPerGPU: new Decimal(serialized.totalPerGPU),
    utilizationPercent: new Decimal(serialized.utilizationPercent),
    singleGPUBaseline: new Decimal(serialized.singleGPUBaseline),
  }
}

/**
 * Reconstruct OffloadedVRAMBreakdown from serialized strings
 *
 * Web Workers serialize Decimal values to strings for structured cloning.
 * This function reconstructs Decimal objects from those strings.
 */
function reconstructOffloadingBreakdown(
  serialized: {
    onDevice: {
      modelWeights: string
      kvCache: string
      activations: string
      frameworkOverhead: string
      total: string
    }
    offloaded: {
      modelWeights: string
      kvCache: string
      total: string
    }
    performanceImpact: string
    slowdownFactor: number
  } | null,
): OffloadedVRAMBreakdown | null {
  if (!serialized) return null
  return {
    onDevice: reconstructVRAMBreakdown(serialized.onDevice),
    offloaded: {
      modelWeights: new Decimal(serialized.offloaded.modelWeights),
      kvCache: new Decimal(serialized.offloaded.kvCache),
      total: new Decimal(serialized.offloaded.total),
    },
    performanceImpact: serialized.performanceImpact,
    slowdownFactor: serialized.slowdownFactor,
  }
}

/**
 * Calculate inference VRAM and performance with Web Worker offloading
 *
 * @param model - Model configuration (null = no calculation)
 * @param gpu - GPU hardware specs (null = no calculation)
 * @param quantization - Model weight quantization format
 * @param sequenceLength - Maximum sequence length (prompt + generation)
 * @param batchSize - Number of concurrent sequences
 * @param kvQuantization - KV cache quantization precision (defaults to fp16)
 * @param numGPUs - Number of GPUs for multi-GPU calculation (defaults to 1)
 * @param shardingStrategy - Multi-GPU sharding strategy (defaults to tensor-parallel)
 * @param offloadingConfig - CPU/RAM or NVMe offloading configuration (optional)
 * @returns Hook state with result, loading, and error
 *
 * @example
 * ```tsx
 * const { result, loading, error } = useInferenceCalculation(
 *   selectedModel,
 *   selectedGPU,
 *   'gptq',
 *   4096,
 *   1,
 *   'fp16'
 * )
 *
 * if (loading) return <Spinner />
 * if (error) return <ErrorMessage error={error} />
 * if (!result) return <NoSelection />
 *
 * return <VRAMBreakdown vram={result.vram} performance={result.performance} />
 * ```
 */
export function useInferenceCalculation(
  model: Model | null,
  gpu: GPU | null,
  quantization: QuantizationFormat,
  sequenceLength: number,
  batchSize: number,
  kvQuantization?: KVCachePrecision,
  numGPUs?: number,
  shardingStrategy?: ShardingStrategy,
  offloadingConfig?: OffloadingConfig,
): UseInferenceCalculationResult {
  const [result, setResult] = useState<UseInferenceCalculationResult['result']>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Early return if model or GPU not selected
    if (!model || !gpu) {
      setResult(null)
      setLoading(false)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    // Web Worker path (preferred - non-blocking)
    if (typeof Worker !== 'undefined') {
      const worker = new Worker(new URL('../workers/calculation.worker.ts', import.meta.url), {
        type: 'module',
      })

      worker.onmessage = (event) => {
        const { type, payload, error: workerError } = event.data

        if (type === 'CALCULATION_RESULT') {
          // Reconstruct Decimal objects from serialized strings
          const vram = reconstructVRAMBreakdown(payload.vram)
          const performance = reconstructPerformanceEstimate(payload.performance)
          const offloading = reconstructOffloadingBreakdown(payload.offloading)
          const multiGPU = reconstructMultiGPUBreakdown(payload.multiGPU)
          const interconnectWarning = payload.interconnectWarning ?? null

          setResult({ vram, performance, offloading, multiGPU, interconnectWarning })
          setLoading(false)
        } else if (type === 'CALCULATION_ERROR') {
          setError(workerError)
          setLoading(false)
        }

        worker.terminate()
      }

      worker.onerror = (event) => {
        setError(`Worker error: ${event.message}`)
        setLoading(false)
        worker.terminate()
      }

      // Post calculation request
      worker.postMessage({
        type: 'CALCULATE_INFERENCE',
        payload: {
          model,
          gpu,
          quantization,
          sequenceLength,
          batchSize,
          kvQuantization,
          numGPUs: numGPUs ?? 1,
          shardingStrategy: shardingStrategy ?? 'tensor-parallel',
          offloadingEnabled: offloadingConfig?.enabled ?? false,
          offloadTarget: offloadingConfig?.target ?? 'cpu-ram',
          offloadMode: offloadingConfig?.mode ?? 'percentage',
          offloadPercentage: offloadingConfig?.offloadPercentage ?? 0,
          offloadLayers: offloadingConfig?.offloadLayers ?? 0,
          kvCacheOffload: offloadingConfig?.kvCacheOffload ?? false,
        },
      })

      // Cleanup on unmount or dependency change
      return () => {
        worker.terminate()
      }
    }

    // Sync fallback (SSR, old browsers)
    Promise.all([
      import('@engines/inference'),
      import('@engines/performance'),
      import('@engines/offloading'),
      import('@engines/multi-gpu'),
    ])
      .then(([inferenceModule, performanceModule, offloadingModule, multiGPUModule]) => {
        const vram = inferenceModule.calculateInferenceVRAM({
          model,
          quantization,
          sequenceLength,
          batchSize,
          kvQuantization,
        })

        const performance = performanceModule.estimatePerformance({
          model,
          gpu,
          quantization,
          batchSize,
        })

        // Offloading calculation (if enabled)
        let offloading = null
        if (offloadingConfig?.enabled) {
          offloading = offloadingModule.calculateOffloadedVRAM(
            vram,
            offloadingConfig,
            model.num_hidden_layers,
          )
        }

        // Multi-GPU calculation (only when numGPUs > 1)
        // IMPORTANT: If offloading is active, use the offloaded onDevice breakdown as the base
        let multiGPU = null
        let interconnectWarning = null
        const effectiveNumGPUs = numGPUs ?? 1
        const effectiveStrategy = shardingStrategy ?? 'tensor-parallel'
        if (effectiveNumGPUs > 1) {
          const baseBreakdown = offloading ? offloading.onDevice : vram
          multiGPU = multiGPUModule.calculateMultiGPUVRAM(
            baseBreakdown,
            model,
            gpu.vram_gb,
            effectiveNumGPUs,
            effectiveStrategy,
          )
          const validation = multiGPUModule.validateInterconnect(
            gpu,
            effectiveNumGPUs,
            effectiveStrategy,
          )
          interconnectWarning = validation.warning
        }

        setResult({ vram, performance, offloading, multiGPU, interconnectWarning })
        setLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
  }, [
    model,
    gpu,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization,
    numGPUs,
    shardingStrategy,
    offloadingConfig,
  ])

  return { result, loading, error }
}
