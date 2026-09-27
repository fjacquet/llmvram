import gpusData from '@data/gpus.json'
import {
  effectiveFraction,
  FABRIC_HOP_LATENCY_S,
  FABRIC_SPECS,
  fabricDecodeEfficiency,
  fabricHopSeconds,
  fabricPrefillEfficiency,
  GB10_EFFECTIVE_FRACTION,
  HGX_EFFECTIVE_FRACTION,
  interNodeGBps,
  maxNumBatchedTokens,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
  prefillMicrobatches,
  prefillPipelineFill,
  resolveFabricSpec,
} from '@engines/fabric'
import { validateGPU, validateGPUs } from '@utils/schemas'
import { describe, expect, it } from 'vitest'

const realGPUs = validateGPUs(gpusData)
function findGPU(id: string) {
  const gpu = realGPUs.find((g) => g.id === id)
  if (!gpu) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return gpu
}

describe('fabricHopSeconds (spec Section 3b)', () => {
  it('Llama 70B at 8k over one GB10 on 200 GbE: 268,435,456 B at 25 x 0.37 GB/s + 10 us = 29.03 ms', () => {
    const gbps = interNodeGBps(
      FABRIC_SPECS['ethernet-200g'].portGBps,
      1,
      effectiveFraction(findGPU('nvidia-gb10')),
    )
    expect(gbps).toBeCloseTo(9.25, 10)
    expect(8192 * 8192 * 2 * 2).toBe(268_435_456)
    expect(fabricHopSeconds(8192, 8192, gbps)).toBeCloseTo(0.02903, 6)
  })

  it('is the latency floor alone for zero tokens', () => {
    expect(fabricHopSeconds(0, 8192, 100)).toBe(FABRIC_HOP_LATENCY_S)
  })
})

describe('effectiveFraction', () => {
  it('uses the GB10 measurement for DGX Spark and the HGX assumption elsewhere', () => {
    expect(effectiveFraction(findGPU('nvidia-gb10'))).toBe(GB10_EFFECTIVE_FRACTION)
    expect(effectiveFraction(findGPU('nvidia-h100-80gb-sxm'))).toBe(HGX_EFFECTIVE_FRACTION)
  })

  it('is data-driven (gpudirect_rdma), not an id check: an H100 without GPUDirect RDMA still gets the secondary-source value', () => {
    // Derived row, validated by the real schema: proves effectiveFraction reads
    // gpu.gpudirect_rdma, not gpu.id === 'nvidia-gb10' (nvidia-h100-80gb-sxm normally
    // has no gpudirect_rdma override and reads as RDMA-capable, see the H100 case above).
    const noRdma = validateGPU({ ...findGPU('nvidia-h100-80gb-sxm'), gpudirect_rdma: false })
    expect(effectiveFraction(noRdma)).toBe(GB10_EFFECTIVE_FRACTION)
  })
})

describe('interNodeGBps', () => {
  it('is port x GPUs per node x eta', () => {
    expect(interNodeGBps(100, 8, 0.8)).toBeCloseTo(640, 10)
  })
})

describe('maxNumBatchedTokens (vLLM engine/arg_utils.py defaults)', () => {
  it('scales with GPU memory and excludes A100', () => {
    expect(maxNumBatchedTokens(findGPU('nvidia-b200-192gb'))).toBe(16384) // 180 GB
    expect(maxNumBatchedTokens(findGPU('nvidia-h100-80gb-sxm'))).toBe(8192)
    expect(maxNumBatchedTokens(findGPU('nvidia-gb10'))).toBe(8192) // 128 GB
    expect(maxNumBatchedTokens(findGPU('nvidia-a100-80gb-sxm'))).toBe(2048)
    expect(maxNumBatchedTokens(findGPU('nvidia-rtx-4090'))).toBe(2048)
  })
})

describe('prefillMicrobatches and prefillPipelineFill', () => {
  it('M = ceil(B x T / C), at least 1', () => {
    expect(prefillMicrobatches(1, 8192, 8192)).toBe(1)
    expect(prefillMicrobatches(32, 8192, 8192)).toBe(32)
    expect(prefillMicrobatches(1, 32768, 8192)).toBe(4)
    expect(prefillMicrobatches(3, 1000, 8192)).toBe(1)
  })

  it('fill is M / (M + N - 1): one microbatch walks the stages serially', () => {
    expect(prefillPipelineFill(1, 1)).toBe(1)
    expect(prefillPipelineFill(1, 2)).toBe(0.5)
    expect(prefillPipelineFill(32, 2)).toBeCloseTo(32 / 33, 12)
  })
})

describe('ethernet-200g preset', () => {
  it('is 200 Gb/s = 25 GB/s per port (ConnectX-7)', () => {
    expect(FABRIC_SPECS['ethernet-200g'].portGBps).toBe(25)
  })
})

describe('perNodeFabricGBps', () => {
  it('multiplies port speed by one NIC per GPU', () => {
    expect(perNodeFabricGBps(100, 8)).toBe(800)
    expect(perNodeFabricGBps(200, 8)).toBe(1600)
    expect(perNodeFabricGBps(50, 4)).toBe(200)
  })
})

describe('fabricPrefillEfficiency', () => {
  it('is capped by the pipeline stage-boundary base at the reference bandwidth', () => {
    expect(fabricPrefillEfficiency(1600, 1.0)).toBeCloseTo(0.95, 2)
  })

  it('degrades superlinearly as bandwidth halves', () => {
    const at1600 = fabricPrefillEfficiency(1600, 1.0)
    const at800 = fabricPrefillEfficiency(800, 1.0)
    const at400 = fabricPrefillEfficiency(400, 1.0)
    expect(at1600 - at800).toBeLessThan(at800 - at400)
  })

  it('matches the documented table values', () => {
    expect(fabricPrefillEfficiency(800, 1.0)).toBeCloseTo(0.874, 3)
    expect(fabricPrefillEfficiency(400, 1.0)).toBeCloseTo(0.76, 3)
    expect(fabricPrefillEfficiency(100, 1.0)).toBeCloseTo(0.418, 3)
  })

  it('gives InfiniBand an edge over Ethernet at the same line rate', () => {
    expect(fabricPrefillEfficiency(800, 1.02)).toBeGreaterThan(fabricPrefillEfficiency(800, 1.0))
  })

  it('never exceeds 1 or falls below the floor', () => {
    expect(fabricPrefillEfficiency(100_000, 1.02)).toBeLessThanOrEqual(1)
    expect(fabricPrefillEfficiency(0.5, 1.0)).toBeGreaterThanOrEqual(0.05)
  })
})

describe('fabricDecodeEfficiency', () => {
  it('stays near-flat: a decode hop ships kilobytes and is latency-bound', () => {
    expect(fabricDecodeEfficiency(1600)).toBeCloseTo(0.99, 2)
    expect(fabricDecodeEfficiency(100)).toBeCloseTo(0.95, 2)
  })

  it('is always far above the prefill efficiency at the same bandwidth', () => {
    expect(fabricDecodeEfficiency(100)).toBeGreaterThan(fabricPrefillEfficiency(100, 1.0))
  })
})

describe('pipelineBubbleEfficiency', () => {
  it('is 1.0 for a single node — no pipeline, no bubble', () => {
    expect(pipelineBubbleEfficiency(1, 1)).toBe(1)
  })

  it('costs real throughput at batch 1 across 4 nodes', () => {
    // M = max(1, 4) = 4, so 4 / (4 + 3) = 0.571
    expect(pipelineBubbleEfficiency(1, 4)).toBeCloseTo(0.571, 3)
  })

  it('improves as batch size feeds more microbatches into the pipeline', () => {
    expect(pipelineBubbleEfficiency(64, 4)).toBeGreaterThan(pipelineBubbleEfficiency(4, 4))
  })
})

describe('resolveFabricSpec', () => {
  it('returns the preset for a known type', () => {
    expect(resolveFabricSpec('ethernet-800g', null)).toEqual(FABRIC_SPECS['ethernet-800g'])
  })

  it('builds a spec from custom input', () => {
    const spec = resolveFabricSpec('custom', { name: 'Lab fabric', port_gbps: 25 })
    expect(spec.portGBps).toBe(25)
    expect(spec.label).toContain('Lab fabric')
    expect(spec.classFactor).toBe(1.0)
  })

  it('falls back to the 800G Ethernet preset when custom is selected with no input', () => {
    expect(resolveFabricSpec('custom', null)).toEqual(FABRIC_SPECS['ethernet-800g'])
  })
})

describe('FABRIC_SPECS', () => {
  it('orders 1.6TbE fastest and 100GbE slowest', () => {
    expect(FABRIC_SPECS['ethernet-1600g'].portGBps).toBe(200)
    expect(FABRIC_SPECS['ethernet-100g'].portGBps).toBe(12.5)
  })

  it('marks only the InfiniBand entries with the class bonus', () => {
    expect(FABRIC_SPECS['infiniband-xdr'].classFactor).toBe(1.02)
    expect(FABRIC_SPECS['infiniband-ndr'].classFactor).toBe(1.02)
    expect(FABRIC_SPECS['ethernet-800g'].classFactor).toBe(1.0)
  })
})
