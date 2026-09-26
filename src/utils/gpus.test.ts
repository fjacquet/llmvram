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

  it('every GPU row declares max_gpus_per_node as a positive integer', () => {
    const result = validateGPUs(gpusData)
    for (const gpu of result) {
      expect(Number.isInteger(gpu.max_gpus_per_node)).toBe(true)
      expect(gpu.max_gpus_per_node).toBeGreaterThan(0)
    }
  })

  it('bounds every Apple Silicon row at a single GPU', () => {
    const result = validateGPUs(gpusData)
    const apple = result.filter((g) => g.tier === 'apple-silicon')
    expect(apple.length).toBeGreaterThan(0)
    for (const gpu of apple) {
      expect(gpu.max_gpus_per_node).toBe(1)
    }
  })

  it('offers GB300 as both an 8-GPU HGX baseboard and a 72-GPU NVL72 rack', () => {
    const result = validateGPUs(gpusData)
    const hgx = result.find((g) => g.id === 'nvidia-gb300-288gb')
    const nvl72 = result.find((g) => g.id === 'nvidia-gb300-nvl72')

    expect(hgx).toBeDefined()
    expect(nvl72).toBeDefined()
    expect(hgx?.max_gpus_per_node).toBe(8)
    expect(nvl72?.max_gpus_per_node).toBe(72)
  })

  it('gives the two Blackwell Ultra rows the same bandwidth and interconnect', () => {
    const result = validateGPUs(gpusData)
    const hgx = result.find((g) => g.id === 'nvidia-gb300-288gb')
    const nvl72 = result.find((g) => g.id === 'nvidia-gb300-nvl72')

    expect(nvl72?.memory_bandwidth_gbps).toBe(hgx?.memory_bandwidth_gbps)
    expect(nvl72?.interconnect).toBe(hgx?.interconnect)
  })

  it('lists HGX B300 at its air-cooled baseboard figures, not the NVL72 ones', () => {
    // NVIDIA HGX page: 2.1 TB total, 36 PF FP16 sparse, 600 TF FP32, across 8 GPUs
    const hgx = validateGPUs(gpusData).find((g) => g.id === 'nvidia-gb300-288gb')
    expect(hgx?.vram_gb).toBe(262.5)
    expect(hgx?.fp16_tflops).toBe(2250)
    expect(hgx?.fp32_tflops).toBe(75)
    expect(hgx?.tdp_watts).toBe(1100)
    expect(hgx?.name).not.toContain('GB300')
    expect(hgx?.spec_url).toBe('https://www.nvidia.com/en-us/data-center/hgx/')
  })

  it('lists GB300 NVL72 at dense FP16 (360 PF sparse / 72 / 2)', () => {
    const nvl72 = gpusData.find((g) => g.id === 'nvidia-gb300-nvl72')
    expect(nvl72?.fp16_tflops).toBe(2500)
  })

  it('lists the GB300 Desktop Superchip at dense FP16 (5 PF sparse / 2)', () => {
    const desktop = gpusData.find((g) => g.id === 'nvidia-gb300-desktop-252gb')
    expect(desktop?.fp16_tflops).toBe(2500)
  })

  it('stores dense FP16 for every GPU, never a with-sparsity figure', () => {
    // The densest part in the database is ~2.5 PF (GB300, MI355X). Anything above
    // 2600 means a vendor's with-sparsity number leaked in.
    for (const gpu of gpusData) {
      expect(gpu.fp16_tflops ?? 0, gpu.id).toBeLessThan(2600)
    }
  })
})

describe('AMD Instinct MI350 / MI325 series', () => {
  it('includes MI355X with 288 GB HBM3E at 8 TB/s', () => {
    const gpu = gpusData.find((g) => g.id === 'amd-mi355x')
    expect(gpu?.vram_gb).toBe(288)
    expect(gpu?.memory_bandwidth_gbps).toBe(8000)
    expect(gpu?.memory_type).toBe('HBM3E')
    expect(gpu?.tdp_watts).toBe(1400)
  })

  it('includes MI350X as the air-cooled sibling: same memory, lower TBP', () => {
    const mi350x = gpusData.find((g) => g.id === 'amd-mi350x')
    const mi355x = gpusData.find((g) => g.id === 'amd-mi355x')
    expect(mi350x?.vram_gb).toBe(mi355x?.vram_gb)
    expect(mi350x?.memory_bandwidth_gbps).toBe(mi355x?.memory_bandwidth_gbps)
    expect(mi350x?.tdp_watts).toBe(1000)
    expect(mi350x?.fp16_tflops ?? 0).toBeLessThan(mi355x?.fp16_tflops ?? 0)
  })

  it('includes MI325X with 256 GB and MI300X compute', () => {
    const mi325x = gpusData.find((g) => g.id === 'amd-mi325x')
    const mi300x = gpusData.find((g) => g.id === 'amd-mi300x')
    expect(mi325x?.vram_gb).toBe(256)
    expect(mi325x?.memory_bandwidth_gbps).toBe(6000)
    expect(mi325x?.fp16_tflops).toBe(mi300x?.fp16_tflops)
  })

  it('stores dense FP16, not AMD sparse marketing figures', () => {
    // Sparse would be ~5033 for MI355X. Anything above 3000 means a sparse
    // number leaked into the database.
    for (const gpu of gpusData.filter((g) => g.manufacturer === 'amd')) {
      expect(gpu.fp16_tflops ?? 0).toBeLessThan(3000)
    }
  })

  it('puts every AMD accelerator on Infinity Fabric', () => {
    for (const gpu of gpusData.filter((g) => g.manufacturer === 'amd')) {
      expect(gpu.interconnect).toBe('infinity-fabric')
    }
  })
})

describe('NVIDIA B200', () => {
  it('lists B200 at its allocatable 180GB, not the 192GB stack size', () => {
    const result = validateGPUs(gpusData)
    const b200 = result.find((g) => g.id === 'nvidia-b200-192gb')
    expect(b200).toBeDefined()
    expect(b200?.vram_gb).toBe(180)
    expect(b200?.name).not.toContain('192')
  })

  it('lists B200 at dense FP16 (HGX B200: 36 PF sparse / 8 / 2)', () => {
    const b200 = gpusData.find((g) => g.id === 'nvidia-b200-192gb')
    expect(b200?.fp16_tflops).toBe(2250)
  })
})
