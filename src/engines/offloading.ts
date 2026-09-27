import type { GPU } from '@utils/schemas'
import Decimal from 'decimal.js'
import { graceLinkGBps, KV_TIER_PRESETS } from './kv-tier'
import type {
  InferenceVRAMBreakdown,
  OffloadedVRAMBreakdown,
  OffloadingConfig,
  OffloadTarget,
} from './types'

/**
 * Fraction of model weights offloaded off-GPU (0-1), from either offload mode.
 * Shared by `calculateOffloadedVRAM` (VRAM split) and `estimatePerformance`
 * (decode bytes over the host link), so both agree on the on-device split.
 */
export function offloadWeightFraction(config: OffloadingConfig, totalLayers?: number): number {
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
  return Math.max(0, Math.min(1, offloadFraction))
}

/**
 * Per-GPU read bandwidth (decimal GB/s) of the link decode bytes travel over
 * when offloaded: the same presets the KV storage tier uses, so the two
 * features never quote different numbers for the same physical link.
 *
 * - 'nvme': local NVMe (KV_TIER_PRESETS['local-nvme'], 12 GB/s).
 * - 'cpu-ram': the exact Grace NVLink-C2C figure for this GPU when it has one
 *   (graceLinkGBps: 225 NVL72, 396 Desktop Superchip), else generic PCIe 5
 *   (KV_TIER_PRESETS['host-pcie'], 50 GB/s).
 */
export function hostLinkGBps(target: OffloadTarget, gpuId: string): number {
  if (target === 'nvme') return KV_TIER_PRESETS['local-nvme'].gbpsPerGPU
  return graceLinkGBps(gpuId) ?? KV_TIER_PRESETS['host-pcie'].gbpsPerGPU
}

/**
 * Default host capacity (decimal GB) offered as a placeholder for "Host
 * capacity per server" when the user hasn't overridden it, per offload
 * target and GPU tier.
 *
 * Basis: DGX B300: 2 TB system memory standard (up to 4 TB); 8 x 3.84 TB
 * E1.S NVMe (NVIDIA DGX B300 user guide / datasheet). Non-datacenter:
 * typical workstation assumption.
 */
export function defaultHostCapacityGB(target: OffloadTarget, gpu: Pick<GPU, 'tier'>): number {
  const datacenter = gpu.tier === 'datacenter'
  if (target === 'nvme') return datacenter ? 30720 : 2000
  return datacenter ? 2048 : 128
}

/**
 * Rounds a decode offload-slowdown ratio to 2 significant figures; null
 * (either no offload, or a ratio under 1.05x — indistinguishable from
 * all-in-GPU) means "no measurable slowdown" to the caller.
 */
export function roundOffloadSlowdown(slowdown: number | null): number | null {
  if (slowdown === null || slowdown < 1.05) return null
  return Number(slowdown.toPrecision(2))
}

/**
 * Calculate VRAM breakdown after offloading model weights and/or KV cache.
 *
 * When offloading by percentage: offloadPercentage% of model weights move off GPU.
 * When offloading by layers: (offloadLayers / totalLayers) fraction of weights move off GPU.
 * KV cache offloading moves entire KV cache to CPU/RAM (independent toggle).
 *
 * The performance impact of offloading is estimated by `estimatePerformance`
 * (decode reads offloaded bytes over the host link — see `hostLinkGBps` and
 * `PerformanceEstimate.offloadSlowdown`), not by this function.
 *
 * @param breakdown - Base VRAM breakdown (before offloading)
 * @param config - Offloading configuration
 * @param totalLayers - Total number of layers in model (for layer-based offloading)
 * @returns Breakdown showing on-device vs offloaded memory
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
    }
  }

  const offloadFraction = offloadWeightFraction(config, totalLayers)

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

  return {
    onDevice: {
      modelWeights: onDeviceModelWeights,
      kvCache: onDeviceKVCache,
      // the state travels with the KV cache
      linearState: config.kvCacheOffload ? new Decimal(0) : breakdown.linearState,
      activations: onDeviceActivations,
      frameworkOverhead: onDeviceFramework,
      total: onDeviceTotal,
    },
    offloaded: {
      modelWeights: offloadedModelWeights,
      kvCache: offloadedKVCache,
      total: offloadedTotal,
    },
  }
}
