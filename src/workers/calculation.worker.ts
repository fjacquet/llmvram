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
import { calculateMultiGPUVRAM, validateInterconnect } from '../engines/multi-gpu'
import { calculateOffloadedVRAM } from '../engines/offloading'
import { estimatePerformance } from '../engines/performance'
import type {
  KVCachePrecision,
  OffloadingConfig,
  QuantizationFormat,
  ShardingStrategy,
} from '../engines/types'
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
    numGPUs: number
    shardingStrategy: ShardingStrategy
    offloadingEnabled: boolean
    offloadTarget: string
    offloadMode: string
    offloadPercentage: number
    offloadLayers: number
    kvCacheOffload: boolean
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
    offloading: {
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
    } | null
    multiGPU: {
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
    } | null
    interconnectWarning: string | null
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
      const {
        model,
        gpu,
        quantization,
        sequenceLength,
        batchSize,
        kvQuantization,
        numGPUs,
        shardingStrategy,
        offloadingEnabled,
        offloadTarget,
        offloadMode,
        offloadPercentage,
        offloadLayers,
        kvCacheOffload,
      } = payload

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

      // 3. Offloading calculation (if enabled)
      let offloadingResult = null

      if (offloadingEnabled) {
        const offloadingConfig: OffloadingConfig = {
          enabled: offloadingEnabled,
          target: offloadTarget as 'cpu-ram' | 'nvme',
          mode: offloadMode as 'percentage' | 'layers',
          offloadPercentage,
          offloadLayers,
          kvCacheOffload,
        }

        offloadingResult = calculateOffloadedVRAM(vramBreakdown, offloadingConfig, model.num_layers)
      }

      // 4. Multi-GPU calculation (only when numGPUs > 1)
      // IMPORTANT: If offloading is active, use the offloaded onDevice breakdown as the base
      let multiGPUResult = null
      let interconnectWarning = null

      if (numGPUs > 1) {
        const baseBreakdown = offloadingResult ? offloadingResult.onDevice : vramBreakdown

        multiGPUResult = calculateMultiGPUVRAM(
          baseBreakdown,
          model,
          gpu.vram_gb,
          numGPUs,
          shardingStrategy,
        )

        const validation = validateInterconnect(gpu, numGPUs, shardingStrategy)
        interconnectWarning = validation.warning
      }

      // 5. Serialize Decimal values to strings for structured cloning
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
          offloading: offloadingResult
            ? {
                onDevice: {
                  modelWeights: offloadingResult.onDevice.modelWeights.toString(),
                  kvCache: offloadingResult.onDevice.kvCache.toString(),
                  activations: offloadingResult.onDevice.activations.toString(),
                  frameworkOverhead: offloadingResult.onDevice.frameworkOverhead.toString(),
                  total: offloadingResult.onDevice.total.toString(),
                },
                offloaded: {
                  modelWeights: offloadingResult.offloaded.modelWeights.toString(),
                  kvCache: offloadingResult.offloaded.kvCache.toString(),
                  total: offloadingResult.offloaded.total.toString(),
                },
                performanceImpact: offloadingResult.performanceImpact,
                slowdownFactor: offloadingResult.slowdownFactor,
              }
            : null,
          multiGPU: multiGPUResult
            ? {
                numGPUs: multiGPUResult.numGPUs,
                strategy: multiGPUResult.strategy,
                perGPU: {
                  modelWeights: multiGPUResult.perGPU.modelWeights.toString(),
                  kvCache: multiGPUResult.perGPU.kvCache.toString(),
                  activations: multiGPUResult.perGPU.activations.toString(),
                  frameworkOverhead: multiGPUResult.perGPU.frameworkOverhead.toString(),
                  communicationOverhead: multiGPUResult.perGPU.communicationOverhead.toString(),
                  total: multiGPUResult.perGPU.total.toString(),
                },
                replicatedMemory: multiGPUResult.replicatedMemory.toString(),
                totalPerGPU: multiGPUResult.totalPerGPU.toString(),
                utilizationPercent: multiGPUResult.utilizationPercent.toString(),
                singleGPUBaseline: multiGPUResult.singleGPUBaseline.toString(),
              }
            : null,
          interconnectWarning,
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
