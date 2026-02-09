/**
 * Web Worker for non-blocking VRAM calculations
 *
 * Offloads heavy calculation logic to a background thread to keep UI responsive.
 * Handles structured cloning of Decimal.js values (serialized to strings).
 *
 * Protocol:
 * - Receives CALCULATE_INFERENCE messages with model/GPU/params
 * - Computes VRAM breakdown and performance estimate
 * - Returns CALCULATION_RESULT with serialized Decimal values as strings
 * - Returns CALCULATION_ERROR on failure
 */

import { calculateInferenceVRAM } from '../engines/inference'
import { estimatePerformance } from '../engines/performance'
import type { KVCachePrecision, QuantizationFormat } from '../engines/types'
import type { GPU, Model } from '../utils/schemas'

/**
 * Request message for calculation
 */
interface CalculationRequest {
  type: 'CALCULATE_INFERENCE'
  payload: {
    model: Model
    gpu: GPU
    quantization: QuantizationFormat
    sequenceLength: number
    batchSize: number
    kvQuantization?: KVCachePrecision
  }
}

/**
 * Success response with serialized Decimal values
 *
 * IMPORTANT: Decimal.js instances lose methods during structured cloning.
 * All Decimal values are serialized to strings using .toString().
 * The receiving hook must reconstruct Decimal objects.
 */
interface CalculationSuccessResponse {
  type: 'CALCULATION_RESULT'
  payload: {
    vram: {
      modelWeights: string // Decimal serialized as string
      kvCache: string
      activations: string
      frameworkOverhead: string
      total: string
    }
    performance: {
      tokensPerSecond: string
      timeToFirstToken: string
      isComputeBound: boolean
      isMemoryBound: boolean
      bottleneck: 'compute' | 'memory' | 'balanced'
    }
  }
}

/**
 * Error response
 */
interface CalculationErrorResponse {
  type: 'CALCULATION_ERROR'
  error: string
}

type WorkerMessage = CalculationRequest

/**
 * Message handler
 *
 * Listens for CALCULATE_INFERENCE messages, runs calculations, and posts results.
 */
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data

  if (type === 'CALCULATE_INFERENCE') {
    try {
      const { model, gpu, quantization, sequenceLength, batchSize, kvQuantization } = payload

      // 1. Calculate VRAM breakdown
      const vramBreakdown = calculateInferenceVRAM({
        model,
        quantization,
        sequenceLength,
        batchSize,
        kvQuantization,
      })

      // 2. Estimate performance
      const performance = estimatePerformance({
        model,
        gpu,
        quantization,
        batchSize,
      })

      // 3. Serialize Decimal values to strings for structured cloning
      const response: CalculationSuccessResponse = {
        type: 'CALCULATION_RESULT',
        payload: {
          vram: {
            modelWeights: vramBreakdown.modelWeights.toString(),
            kvCache: vramBreakdown.kvCache.toString(),
            activations: vramBreakdown.activations.toString(),
            frameworkOverhead: vramBreakdown.frameworkOverhead.toString(),
            total: vramBreakdown.total.toString(),
          },
          performance: {
            tokensPerSecond: performance.tokensPerSecond.toString(),
            timeToFirstToken: performance.timeToFirstToken.toString(),
            isComputeBound: performance.isComputeBound,
            isMemoryBound: performance.isMemoryBound,
            bottleneck: performance.bottleneck,
          },
        },
      }

      self.postMessage(response)
    } catch (error) {
      const errorResponse: CalculationErrorResponse = {
        type: 'CALCULATION_ERROR',
        error: error instanceof Error ? error.message : String(error),
      }
      self.postMessage(errorResponse)
    }
  }
}
