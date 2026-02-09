import Decimal from 'decimal.js'
import type { InferenceVRAMBreakdown, OffloadingConfig, OffloadedVRAMBreakdown } from './types'

/**
 * Calculate VRAM breakdown after offloading model weights and/or KV cache.
 *
 * When offloading by percentage: offloadPercentage% of model weights move off GPU.
 * When offloading by layers: (offloadLayers / totalLayers) fraction of weights move off GPU.
 * KV cache offloading moves entire KV cache to CPU/RAM (independent toggle).
 *
 * Performance impact estimates:
 * - CPU/RAM offloading: PCIe bandwidth limited, ~2-5x slower for small offloads, ~15-50x for large
 * - NVMe offloading: Even slower, ~10-100x depending on SSD speed and amount
 * - KV cache offload adds latency per token (every decode step reads from CPU)
 *
 * @param breakdown - Base VRAM breakdown (before offloading)
 * @param config - Offloading configuration
 * @param totalLayers - Total number of layers in model (for layer-based offloading)
 * @returns Breakdown showing on-device vs offloaded memory with performance impact
 */
export function calculateOffloadedVRAM(
  breakdown: InferenceVRAMBreakdown,
  config: OffloadingConfig,
  totalLayers?: number,
): OffloadedVRAMBreakdown {
  // If offloading disabled, return original breakdown with zero offloaded
  if (!config.enabled) {
    return {
      onDevice: breakdown,
      offloaded: {
        modelWeights: new Decimal(0),
        kvCache: new Decimal(0),
        total: new Decimal(0),
      },
      performanceImpact: 'No offloading (GPU only)',
      slowdownFactor: 1.0,
    }
  }

  // Calculate effective offload fraction for model weights
  let offloadFraction: number

  if (config.mode === 'percentage') {
    // Percentage mode: direct percentage of model weights
    offloadFraction = config.offloadPercentage / 100
  } else {
    // Layers mode: fraction based on layer count
    const layers = totalLayers ?? 1
    offloadFraction = Math.min(config.offloadLayers / layers, 1.0)
  }

  // Clamp to valid range [0, 1]
  offloadFraction = Math.max(0, Math.min(1, offloadFraction))

  // Calculate offloaded and remaining model weights
  const offloadedModelWeights = breakdown.modelWeights.mul(offloadFraction)
  const onDeviceModelWeights = breakdown.modelWeights.mul(1 - offloadFraction)

  // Calculate KV cache offloading
  let offloadedKVCache: Decimal
  let onDeviceKVCache: Decimal

  if (config.kvCacheOffload) {
    // Full KV cache offloaded to CPU/RAM
    offloadedKVCache = breakdown.kvCache
    onDeviceKVCache = new Decimal(0)
  } else {
    // KV cache remains on GPU
    offloadedKVCache = new Decimal(0)
    onDeviceKVCache = breakdown.kvCache
  }

  // Activations and framework overhead always remain on GPU
  const onDeviceActivations = breakdown.activations
  const onDeviceFramework = breakdown.frameworkOverhead

  // Calculate totals
  const onDeviceTotal = onDeviceModelWeights
    .add(onDeviceKVCache)
    .add(onDeviceActivations)
    .add(onDeviceFramework)

  const offloadedTotal = offloadedModelWeights.add(offloadedKVCache)

  // Estimate performance impact
  const offloadedGB = offloadedTotal.toNumber()
  let slowdownFactor: number
  let performanceImpact: string

  if (config.target === 'cpu-ram') {
    // CPU/RAM offloading performance impact
    if (offloadedGB <= 4) {
      slowdownFactor = 2
      performanceImpact = '~2x slower'
    } else if (offloadedGB <= 8) {
      slowdownFactor = 3
      performanceImpact = '~2-5x slower'
    } else if (offloadedGB <= 24) {
      slowdownFactor = 8
      performanceImpact = '~5-15x slower'
    } else {
      slowdownFactor = 25
      performanceImpact = '~15-50x slower'
    }
  } else {
    // NVMe offloading is significantly slower
    if (offloadedGB <= 8) {
      slowdownFactor = 10
      performanceImpact = '~5-15x slower'
    } else {
      slowdownFactor = 50
      performanceImpact = '~30-100x slower'
    }
  }

  // Add note about KV cache offloading if enabled
  if (config.kvCacheOffload) {
    performanceImpact += ' (KV cache on CPU adds per-token latency)'
  }

  return {
    onDevice: {
      modelWeights: onDeviceModelWeights,
      kvCache: onDeviceKVCache,
      activations: onDeviceActivations,
      frameworkOverhead: onDeviceFramework,
      total: onDeviceTotal,
    },
    offloaded: {
      modelWeights: offloadedModelWeights,
      kvCache: offloadedKVCache,
      total: offloadedTotal,
    },
    performanceImpact,
    slowdownFactor,
  }
}
