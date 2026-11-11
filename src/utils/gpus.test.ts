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
      'nvlink-3',
      'nvlink-4',
      'nvlink-5',
      'nvlink-6',
      'pcie-4',
      'pcie-5',
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

  it('accepts dense fp8/fp4 TFLOPS and rejects zero', () => {
    const row = gpusData[0]
    expect(() => GPUSchema.parse({ ...row, fp8_tflops: 5000, fp4_tflops: 15000 })).not.toThrow()
    expect(() => GPUSchema.parse({ ...row, fp8_tflops: 0 })).toThrow()
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

  it('locks exactly the HGX/OAM baseboards to the 8-GPU machine', () => {
    const result = validateGPUs(gpusData)
    const locked = result.filter((g) => g.min_gpus_per_node !== undefined).map((g) => g.id)
    expect(locked.sort()).toEqual(
      [
        'amd-mi300x',
        'amd-mi325x',
        'amd-mi350x',
        'amd-mi355x',
        'nvidia-a100-80gb-sxm',
        'nvidia-b200-192gb',
        'nvidia-gb300-288gb',
        'nvidia-h100-80gb-sxm',
        'nvidia-h200-141gb',
        'nvidia-hgx-rubin-nvl8',
      ].sort(),
    )
    for (const gpu of result) {
      if (gpu.min_gpus_per_node === undefined) continue
      expect(gpu.min_gpus_per_node).toBe(8)
      expect(gpu.min_gpus_per_node).toBeLessThanOrEqual(gpu.max_gpus_per_node)
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
    // The densest pre-Rubin part is ~2.5 PF (GB300, MI355X). Anything above 2600
    // means a vendor's with-sparsity number leaked in. NVIDIA's Rubin pages
    // footnote FP16/BF16 as a dense specification, so those two are allowed.
    const allowed: Record<string, number> = {
      'nvidia-vera-rubin-nvl72': 4000,
      'nvidia-hgx-rubin-nvl8': 4000,
    }
    for (const gpu of gpusData) {
      const fp16 = gpu.fp16_tflops ?? 0
      if (gpu.id in allowed) expect(fp16, gpu.id).toBe(allowed[gpu.id])
      else expect(fp16, gpu.id).toBeLessThan(2600)
    }
  })

  it.each([
    // id, vram, GB/s, fp16, fp8, fp4, interconnect, max per node
    ['nvidia-gb200-nvl72', 186, 8000, 2500, 5000, 10000, 'nvlink-5', 72],
    ['nvidia-vera-rubin-nvl72', 278, 19200, 4000, 17500, 35000, 'nvlink-6', 72],
    // HGX page system totals / 8 for every figure (130 PF FP8, 270 PF NVFP4 training)
    ['nvidia-hgx-rubin-nvl8', 250, 16250, 4000, 16250, 33750, 'nvlink-6', 8],
  ] as const)(
    'lists %s at usable memory and dense compute',
    (id, vram, bw, f16, f8, f4, ic, max) => {
      const gpu = validateGPUs(gpusData).find((g) => g.id === id)
      expect(gpu).toMatchObject({
        manufacturer: 'nvidia',
        tier: 'datacenter',
        vram_gb: vram,
        memory_bandwidth_gbps: bw,
        fp16_tflops: f16,
        fp8_tflops: f8,
        fp4_tflops: f4,
        interconnect: ic,
        max_gpus_per_node: max,
      })
    },
  )

  it('lists GB300 NVL72 at usable memory (20 TB / 72 ≈ 278, InferenceX 278), not 288', () => {
    const nvl72 = gpusData.find((g) => g.id === 'nvidia-gb300-nvl72')
    expect(nvl72?.vram_gb).toBe(278)
    expect(nvl72?.name).not.toContain('288')
    expect(nvl72?.fp8_tflops).toBe(5000)
    expect(nvl72?.fp4_tflops).toBe(15000)
  })

  it('tags every Rubin row preliminary', () => {
    const rubin = gpusData.filter((g) => g.id.includes('rubin'))
    expect(rubin.length).toBe(2)
    for (const gpu of rubin) expect(gpu.name, gpu.id).toContain('preliminary')
  })

  it('lists B200 and HGX B300 dense FP8/FP4 from the HGX page', () => {
    const b200 = gpusData.find((g) => g.id === 'nvidia-b200-192gb')
    const b300 = gpusData.find((g) => g.id === 'nvidia-gb300-288gb')
    expect([b200?.fp8_tflops, b200?.fp4_tflops]).toEqual([4500, 9000])
    expect([b300?.fp8_tflops, b300?.fp4_tflops]).toEqual([4500, 13500])
  })

  it('lists dense FP8 for Hopper and AMD, FP4 only where vLLM runs it natively', () => {
    const fp = (id: string) => {
      const g = gpusData.find((x) => x.id === id)
      return [g?.fp8_tflops, g?.fp4_tflops]
    }
    // 2026-11-10: dense FP8 is half the published sparse figure; odd halves round down (lower wins)
    expect(fp('nvidia-h100-80gb-pcie')).toEqual([1513, undefined])
    expect(fp('nvidia-h100-80gb-sxm')).toEqual([1979, undefined])
    expect(fp('nvidia-h200-141gb')).toEqual([1979, undefined])
    expect(fp('nvidia-h200-nvl-141gb')).toEqual([1670, undefined])
    expect(fp('nvidia-gb300-desktop-252gb')).toEqual([5000, 15000])
    expect(fp('amd-mi300x')).toEqual([2614, undefined])
    expect(fp('amd-mi325x')).toEqual([2614, undefined])
    expect(fp('amd-mi350x')).toEqual([4600, 9200])
    expect(fp('amd-mi355x')).toEqual([5033, 10066])
  })

  it('never stores fp8/fp4 without fp16, and orders fp16 <= fp8 <= fp4', () => {
    for (const gpu of validateGPUs(gpusData)) {
      if (gpu.fp8_tflops !== undefined) {
        expect(gpu.fp16_tflops, gpu.id).toBeDefined()
        expect(gpu.fp8_tflops, gpu.id).toBeGreaterThanOrEqual(gpu.fp16_tflops ?? Infinity)
      }
      if (gpu.fp4_tflops !== undefined) {
        expect(gpu.fp8_tflops, gpu.id).toBeDefined()
        expect(gpu.fp4_tflops, gpu.id).toBeGreaterThanOrEqual(gpu.fp8_tflops ?? Infinity)
      }
    }
  })

  it('gives no fp8/fp4 to workstation, consumer, GB10 or Apple rows', () => {
    // An explicit id list, not tier: GB300 Desktop is tier 'consumer' yet gets fp8/fp4.
    const ids = [
      'nvidia-l40s',
      'nvidia-rtx-6000-ada',
      'nvidia-rtx-pro-6000-server',
      'nvidia-rtx-5090',
      'nvidia-rtx-4090',
      'nvidia-rtx-3090',
      'nvidia-gb10',
      ...gpusData.filter((g) => g.id.startsWith('apple-')).map((g) => g.id),
    ]
    for (const id of ids) {
      const gpu = gpusData.find((g) => g.id === id)
      expect(gpu, id).toBeDefined()
      expect([gpu?.fp8_tflops, gpu?.fp4_tflops], id).toEqual([undefined, undefined])
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

describe('GPU topology (v2 per-card audit, spec Section 3)', () => {
  const gpus = validateGPUs(gpusData)
  const byId = (id: string) => {
    const gpu = gpus.find((g) => g.id === id)
    if (!gpu) throw new Error(`GPU not found: ${id}`)
    return gpu
  }

  // 29 after the v2 audit (H200 NVL split from H200 SXM), plus GB200 NVL72,
  // Vera Rubin NVL72 and HGX Rubin NVL8.
  it('lists 32 GPUs (29 after the v2 audit, plus GB200 NVL72 and two Rubin rows)', () => {
    expect(gpus).toHaveLength(32)
  })

  it('H100 PCIe: PCIe 5 with a 2-card NVLink bridge at 600 GB/s, dense FP16 756', () => {
    const g = byId('nvidia-h100-80gb-pcie')
    expect(g.interconnect).toBe('pcie-5')
    expect(g.nvlink_bridge).toEqual({ type: 'nvlink-3', size: 2 })
    expect(g.fp16_tflops).toBe(756)
  })

  it('A100 PCIe: PCIe 4 with a 2-card NVLink bridge; A100 SXM: NVLink 3', () => {
    expect(byId('nvidia-a100-80gb-pcie').interconnect).toBe('pcie-4')
    expect(byId('nvidia-a100-80gb-pcie').nvlink_bridge).toEqual({ type: 'nvlink-3', size: 2 })
    expect(byId('nvidia-a100-80gb-sxm').interconnect).toBe('nvlink-3')
  })

  it('H200 id is the SXM product with no interconnect variants', () => {
    const g = byId('nvidia-h200-141gb')
    expect(g.interconnect).toBe('nvlink-4')
    expect(g.interconnect_options).toBeUndefined()
    expect(g.nvlink_bridge).toBeUndefined()
  })

  it('H200 NVL: PCIe 5 card with a 4-way NVLink 4 bridge, dense FP16 835', () => {
    const g = byId('nvidia-h200-nvl-141gb')
    expect(g).toMatchObject({
      vram_gb: 141,
      memory_bandwidth_gbps: 4800,
      fp16_tflops: 835,
      fp32_tflops: 60,
      interconnect: 'pcie-5',
      nvlink_bridge: { type: 'nvlink-4', size: 4 },
      max_gpus_per_node: 8,
      tier: 'datacenter',
    })
  })

  it('GB10 (DGX Spark) is one GPU per node, no scale-up link, unified memory, no GPUDirect RDMA', () => {
    const g = byId('nvidia-gb10')
    expect(g.max_gpus_per_node).toBe(1)
    expect(g.interconnect).toBe('none')
    expect(g.interconnect_options).toBeUndefined()
    expect(g.unified_memory).toBe(true)
    expect(g.gpudirect_rdma).toBe(false)
  })

  it('no other database GPU sets gpudirect_rdma (undefined reads as RDMA-capable)', () => {
    expect(gpus.filter((g) => g.gpudirect_rdma !== undefined).map((g) => g.id)).toEqual([
      'nvidia-gb10',
    ])
  })

  it('GB300 Desktop Superchip has no GPU-to-GPU link (single GPU)', () => {
    expect(byId('nvidia-gb300-desktop-252gb').interconnect).toBe('none')
  })

  it('L40S and RTX 6000 Ada are explicit PCIe 4', () => {
    expect(byId('nvidia-l40s').interconnect).toBe('pcie-4')
    expect(byId('nvidia-rtx-6000-ada').interconnect).toBe('pcie-4')
  })

  it('marks every Apple Silicon row and GB10 as unified memory, and nothing else', () => {
    const unified = gpus.filter((g) => g.unified_memory === true).map((g) => g.id)
    const expected = gpus.filter((g) => g.manufacturer === 'apple').map((g) => g.id)
    expect(unified.sort()).toEqual([...expected, 'nvidia-gb10'].sort())
  })

  it('no database GPU offers interconnect variants any more', () => {
    expect(gpus.filter((g) => g.interconnect_options !== undefined)).toEqual([])
  })
})
