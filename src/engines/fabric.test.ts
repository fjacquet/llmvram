import {
  FABRIC_SPECS,
  fabricDecodeEfficiency,
  fabricPrefillEfficiency,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
  resolveFabricSpec,
} from '@engines/fabric'
import { describe, expect, it } from 'vitest'

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
