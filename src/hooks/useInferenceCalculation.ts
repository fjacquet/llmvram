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
  PerformanceEstimate,
  QuantizationFormat,
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
 * Calculate inference VRAM and performance with Web Worker offloading
 *
 * @param model - Model configuration (null = no calculation)
 * @param gpu - GPU hardware specs (null = no calculation)
 * @param quantization - Model weight quantization format
 * @param sequenceLength - Maximum sequence length (prompt + generation)
 * @param batchSize - Number of concurrent sequences
 * @param kvQuantization - KV cache quantization precision (defaults to fp16)
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

          setResult({ vram, performance })
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
        },
      })

      // Cleanup on unmount or dependency change
      return () => {
        worker.terminate()
      }
    }

    // Sync fallback (SSR, old browsers)
    Promise.all([import('@engines/inference'), import('@engines/performance')])
      .then(([inferenceModule, performanceModule]) => {
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

        setResult({ vram, performance })
        setLoading(false)
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
  }, [model, gpu, quantization, sequenceLength, batchSize, kvQuantization])

  return { result, loading, error }
}
