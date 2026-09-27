import gpusData from '@data/gpus.json'
import { validateGPUs } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import {
  calculateOffloadedVRAM,
  defaultHostCapacityGB,
  hostLinkGBps,
  offloadWeightFraction,
  roundOffloadSlowdown,
} from './offloading'
import type { InferenceVRAMBreakdown, OffloadingConfig } from './types'

const gpus = validateGPUs(gpusData)
function findGPU(id: string) {
  const gpu = gpus.find((g) => g.id === id)
  if (!gpu) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return gpu
}

describe('calculateOffloadedVRAM', () => {
  // Mock baseline breakdown
  const baselineBreakdown: InferenceVRAMBreakdown = {
    modelWeights: new Decimal(40),
    kvCache: new Decimal(8),
    activations: new Decimal(2),
    frameworkOverhead: new Decimal(1),
    total: new Decimal(51),
  }

  it('disabled offloading returns original breakdown unchanged', () => {
    const config: OffloadingConfig = {
      enabled: false,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 50,
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.onDevice).toEqual(baselineBreakdown)
    expect(result.offloaded.total.toNumber()).toBe(0)
  })

  it('50% percentage offload moves half of model weights', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 50,
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.offloaded.modelWeights.toNumber()).toBe(20)
    expect(result.onDevice.modelWeights.toNumber()).toBe(20)
    expect(result.offloaded.kvCache.toNumber()).toBe(0)
    expect(result.onDevice.kvCache.toNumber()).toBe(8)
  })

  it('100% percentage offload moves all model weights', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 100,
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.offloaded.modelWeights.toNumber()).toBe(40)
    expect(result.onDevice.modelWeights.toNumber()).toBe(0)
    expect(result.offloaded.total.toNumber()).toBe(40)
  })

  it('0% percentage offload moves nothing', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 0,
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.offloaded.total.toNumber()).toBe(0)
    expect(result.onDevice.modelWeights.toNumber()).toBe(40)
  })

  it('KV cache offload zeroes KV cache on device', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 0,
      offloadLayers: 0,
      kvCacheOffload: true,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.onDevice.kvCache.toNumber()).toBe(0)
    expect(result.offloaded.kvCache.toNumber()).toBe(8)
    expect(result.offloaded.total.toNumber()).toBe(8)
  })

  it('both model weights and KV cache offloaded together', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 50,
      offloadLayers: 0,
      kvCacheOffload: true,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.offloaded.modelWeights.toNumber()).toBe(20)
    expect(result.offloaded.kvCache.toNumber()).toBe(8)
    expect(result.offloaded.total.toNumber()).toBe(28)
    expect(result.onDevice.modelWeights.toNumber()).toBe(20)
    expect(result.onDevice.kvCache.toNumber()).toBe(0)
  })

  it('layer-based offload: 10 of 80 layers = 12.5% of weights offloaded', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'layers',
      offloadPercentage: 0,
      offloadLayers: 10,
      kvCacheOffload: false,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config, 80)

    // 10/80 = 0.125 = 12.5%
    expect(result.offloaded.modelWeights.toNumber()).toBe(5)
    expect(result.onDevice.modelWeights.toNumber()).toBe(35)
  })

  it('activations and framework overhead always remain on GPU', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 100,
      offloadLayers: 0,
      kvCacheOffload: true,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.onDevice.activations.toNumber()).toBe(2)
    expect(result.onDevice.frameworkOverhead.toNumber()).toBe(1)
  })

  it('total on-device equals sum of on-device components', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 50,
      offloadLayers: 0,
      kvCacheOffload: true,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    const expectedTotal = result.onDevice.modelWeights
      .add(result.onDevice.kvCache)
      .add(result.onDevice.activations)
      .add(result.onDevice.frameworkOverhead)

    expect(result.onDevice.total.toNumber()).toBe(expectedTotal.toNumber())
  })
})

describe('offloadWeightFraction', () => {
  it('percentage mode returns the direct fraction', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 50,
      offloadLayers: 0,
      kvCacheOffload: false,
    }
    expect(offloadWeightFraction(config)).toBe(0.5)
  })

  it('layers mode divides by total layers, clamped to 1', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'layers',
      offloadPercentage: 0,
      offloadLayers: 10,
      kvCacheOffload: false,
    }
    expect(offloadWeightFraction(config, 80)).toBe(0.125)
    expect(offloadWeightFraction({ ...config, offloadLayers: 999 }, 80)).toBe(1)
  })
})

describe('hostLinkGBps', () => {
  it('nvme uses the local-nvme KV tier preset (12 GB/s), regardless of GPU', () => {
    expect(hostLinkGBps('nvme', 'nvidia-h100-80gb-sxm')).toBe(12)
    expect(hostLinkGBps('nvme', 'nvidia-gb300-nvl72')).toBe(12)
  })

  it('cpu-ram uses the exact Grace link for a Grace-host GPU', () => {
    expect(hostLinkGBps('cpu-ram', 'nvidia-gb300-nvl72')).toBe(225)
    expect(hostLinkGBps('cpu-ram', 'nvidia-gb300-desktop-252gb')).toBe(396)
  })

  it('cpu-ram falls back to generic PCIe 5 (50 GB/s) for a non-Grace GPU', () => {
    expect(hostLinkGBps('cpu-ram', 'nvidia-h100-80gb-sxm')).toBe(50)
  })
})

describe('defaultHostCapacityGB', () => {
  it('datacenter tier: 2048 GB cpu-ram, 30720 GB nvme', () => {
    const gpu = findGPU('nvidia-h100-80gb-sxm')
    expect(gpu.tier).toBe('datacenter')
    expect(defaultHostCapacityGB('cpu-ram', gpu)).toBe(2048)
    expect(defaultHostCapacityGB('nvme', gpu)).toBe(30720)
  })

  it('non-datacenter tiers: 128 GB cpu-ram, 2000 GB nvme', () => {
    const consumer = { tier: 'consumer' as const }
    const apple = { tier: 'apple-silicon' as const }
    expect(defaultHostCapacityGB('cpu-ram', consumer)).toBe(128)
    expect(defaultHostCapacityGB('nvme', consumer)).toBe(2000)
    expect(defaultHostCapacityGB('cpu-ram', apple)).toBe(128)
    expect(defaultHostCapacityGB('nvme', apple)).toBe(2000)
  })
})

describe('roundOffloadSlowdown', () => {
  it('rounds to 2 significant figures', () => {
    expect(roundOffloadSlowdown(31.4)).toBe(31)
    expect(roundOffloadSlowdown(5.678)).toBe(5.7)
    expect(roundOffloadSlowdown(123.4)).toBe(120)
  })

  it('null below the 1.05x threshold or when there is no offload', () => {
    expect(roundOffloadSlowdown(null)).toBeNull()
    expect(roundOffloadSlowdown(1)).toBeNull()
    expect(roundOffloadSlowdown(1.04)).toBeNull()
  })

  it('1.05x and above is a measurable slowdown', () => {
    expect(roundOffloadSlowdown(1.05)).toBe(1.1)
  })
})
