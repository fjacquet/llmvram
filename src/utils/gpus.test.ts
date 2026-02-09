import gpusData from '@data/gpus.json'
import { describe, expect, it } from 'vitest'
import { GPUSchema, validateGPUs } from './schemas'

describe('GPU Database Validation', () => {
  it('should have at least 18 GPUs', () => {
    expect(gpusData.length).toBeGreaterThanOrEqual(18)
  })

  it('should validate all GPU entries against schema', () => {
    const result = validateGPUs(gpusData)
    expect(result.length).toBe(gpusData.length)
  })

  it('should include NVIDIA datacenter GPUs (H100, H200, B200, A100)', () => {
    const nvidiaDC = gpusData.filter(
      (gpu) => gpu.manufacturer === 'nvidia' && gpu.tier === 'datacenter',
    )
    expect(nvidiaDC.length).toBeGreaterThanOrEqual(7)

    const h100 = gpusData.find((gpu) => gpu.id.includes('h100'))
    const h200 = gpusData.find((gpu) => gpu.id.includes('h200'))
    const b200 = gpusData.find((gpu) => gpu.id.includes('b200'))
    const a100 = gpusData.find((gpu) => gpu.id.includes('a100'))
    expect(h100).toBeDefined()
    expect(h200).toBeDefined()
    expect(b200).toBeDefined()
    expect(a100).toBeDefined()
  })

  it('should include NVIDIA consumer GPUs (RTX series)', () => {
    const nvidiaConsumer = gpusData.filter(
      (gpu) => gpu.manufacturer === 'nvidia' && gpu.tier === 'consumer',
    )
    expect(nvidiaConsumer.length).toBeGreaterThanOrEqual(3)

    const rtx5090 = gpusData.find((gpu) => gpu.id.includes('5090'))
    const rtx4090 = gpusData.find((gpu) => gpu.id.includes('4090'))
    const rtx3090 = gpusData.find((gpu) => gpu.id.includes('3090'))
    expect(rtx5090).toBeDefined()
    expect(rtx4090).toBeDefined()
    expect(rtx3090).toBeDefined()
  })

  it('should include AMD MI300X', () => {
    const mi300x = gpusData.find((gpu) => gpu.id.includes('mi300x'))
    expect(mi300x).toBeDefined()
    expect(mi300x?.manufacturer).toBe('amd')
    expect(mi300x?.vram_gb).toBe(192)
  })

  it('should include Apple Silicon GPUs', () => {
    const appleSilicon = gpusData.filter((gpu) => gpu.manufacturer === 'apple')
    expect(appleSilicon.length).toBeGreaterThanOrEqual(4)

    const m1Ultra = gpusData.find((gpu) => gpu.id.includes('m1-ultra'))
    const m4Max = gpusData.find((gpu) => gpu.id.includes('m4-max'))
    expect(m1Ultra).toBeDefined()
    expect(m4Max).toBeDefined()
  })

  it('should have valid VRAM specifications', () => {
    for (const gpu of gpusData) {
      expect(gpu.vram_gb).toBeGreaterThan(0)
      expect(gpu.memory_bandwidth_gbps).toBeGreaterThan(0)
    }
  })

  it('should have consistent interconnect types', () => {
    const validInterconnects = [
      'none',
      'nvlink',
      'nvlink-4',
      'nvlink-5',
      'infinity-fabric',
      'unified',
      undefined,
    ]
    for (const gpu of gpusData) {
      expect(validInterconnects).toContain(gpu.interconnect)
    }
  })

  it('should validate individual GPU entry structure', () => {
    const sampleGPU = gpusData[0]
    const result = GPUSchema.safeParse(sampleGPU)

    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data).toHaveProperty('id')
      expect(result.data).toHaveProperty('name')
      expect(result.data).toHaveProperty('vram_gb')
      expect(result.data).toHaveProperty('memory_bandwidth_gbps')
    }
  })

  it('should have unique GPU IDs', () => {
    const ids = gpusData.map((gpu) => gpu.id)
    const uniqueIds = new Set(ids)
    expect(uniqueIds.size).toBe(ids.length)
  })

  it('should have performance specs for most GPUs', () => {
    const withFP16 = gpusData.filter((gpu) => gpu.fp16_tflops !== undefined)
    // At least 80% should have FP16 specs
    expect(withFP16.length).toBeGreaterThanOrEqual(Math.floor(gpusData.length * 0.8))
  })
})
