import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { calculateOffloadedVRAM } from './offloading'
import type { InferenceVRAMBreakdown, OffloadingConfig } from './types'

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
    expect(result.slowdownFactor).toBe(1.0)
    expect(result.performanceImpact).toBe('No offloading (GPU only)')
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
    expect(result.performanceImpact).toContain('KV cache on CPU adds per-token latency')
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

  it('NVMe target has higher slowdown than CPU/RAM', () => {
    const configCPU: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 50,
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const configNVMe: OffloadingConfig = {
      ...configCPU,
      target: 'nvme',
    }

    const resultCPU = calculateOffloadedVRAM(baselineBreakdown, configCPU)
    const resultNVMe = calculateOffloadedVRAM(baselineBreakdown, configNVMe)

    expect(resultNVMe.slowdownFactor).toBeGreaterThan(resultCPU.slowdownFactor)
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

  it('performance impact scales with offloaded amount for CPU/RAM', () => {
    // Small offload (4GB or less)
    const smallConfig: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 10, // ~4GB
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const smallResult = calculateOffloadedVRAM(baselineBreakdown, smallConfig)
    expect(smallResult.slowdownFactor).toBe(2)
    expect(smallResult.performanceImpact).toBe('~2x slower')

    // Large offload (>24GB)
    const largeConfig: OffloadingConfig = {
      enabled: true,
      target: 'cpu-ram',
      mode: 'percentage',
      offloadPercentage: 100, // 40GB
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const largeResult = calculateOffloadedVRAM(baselineBreakdown, largeConfig)
    expect(largeResult.slowdownFactor).toBe(25)
    expect(largeResult.performanceImpact).toBe('~15-50x slower')
  })

  it('NVMe offloading has extreme slowdown for large amounts', () => {
    const config: OffloadingConfig = {
      enabled: true,
      target: 'nvme',
      mode: 'percentage',
      offloadPercentage: 100, // 40GB to NVMe
      offloadLayers: 0,
      kvCacheOffload: false,
    }

    const result = calculateOffloadedVRAM(baselineBreakdown, config)

    expect(result.slowdownFactor).toBe(50)
    expect(result.performanceImpact).toBe('~30-100x slower')
  })
})
