import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { resolveFabricSpec } from '@engines/fabric'
import { DEFAULT_KV_TIER, type KVTierSettings } from '@engines/kv-tier'
import { sizeDeployment } from '@engines/sizing'
import type { SizingInput, SizingResult } from '@engines/types'
import { render, screen } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { SizingResults } from './SizingResults'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => {
  const m = models.find((x) => x.id === id)
  if (!m) throw new Error(`no model ${id}`)
  return m
}
const gpu = (id: string) => {
  const g = gpus.find((x) => x.id === id)
  if (!g) throw new Error(`no gpu ${id}`)
  return g
}
const gb300 = gpu('nvidia-gb300-nvl72')
const h100 = gpu('nvidia-h100-80gb-sxm')

function size(patch: Partial<SizingInput> & Pick<SizingInput, 'gpu' | 'model'>) {
  return sizeDeployment({
    quantization: 'fp8',
    kvQuantization: 'fp16',
    sequenceLength: 32768,
    users: 2000,
    fabric: resolveFabricSpec('ethernet-800g', null),
    kvTier: DEFAULT_KV_TIER,
    ...patch,
  })
}
const show = (result: SizingResult, g = gb300, kvTier: KVTierSettings = DEFAULT_KV_TIER) =>
  render(<SizingResults result={result} gpu={g} kvTier={kvTier} />)

describe('SizingResults', () => {
  // Tier is DEFAULT_KV_TIER ('none'): the plain sessions line applies
  it('rack: headline, replica line, packing and speed with ceiling', () => {
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300 })
    const c = r.chosen
    if (!c) throw new Error('expected a fit')
    show(r)
    expect(
      screen.getByRole('heading', {
        name: new RegExp(`^${r.units} racks? NVIDIA GB300 NVL72`),
      }),
    ).toBeVisible()
    expect(screen.getByText(`replica = ${c.tp} GPUs (tensor parallel)`)).toBeVisible()
    expect(screen.getByTestId('sizing-replicas')).toHaveTextContent(String(r.replicas))
    expect(screen.getByTestId('sizing-per-unit')).toHaveTextContent(String(r.replicasPerUnit))
    expect(screen.getByTestId('sizing-idle')).toHaveTextContent(String(r.idleGPUsPerUnit))
    expect(screen.getByText(`${c.sessionsPerReplica} sessions per replica`)).toBeVisible()
    expect(screen.getAllByText(/Theoretical ceiling:/)).toHaveLength(2)
  })

  it('uses the singular for one rack', () => {
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300, users: 1 })
    show(r)
    expect(screen.getByRole('heading', { name: /^1 rack NVIDIA/ })).toBeVisible()
  })

  it('servers: "N servers <GPU>" and the multi-server replica', () => {
    const r = size({ model: model('zai-org-glm-5.2'), gpu: h100, users: 50 })
    const c = r.chosen
    if (!c) throw new Error('expected a fit')
    show(r, h100)
    expect(
      screen.getByRole('heading', { name: new RegExp(`^${r.units} servers? NVIDIA H100`) }),
    ).toBeVisible()
    const nodes = c.nodes > 1 ? ` × ${c.nodes} servers` : ''
    expect(screen.getByText(`replica = ${c.tp} GPUs (tensor parallel)${nodes}`)).toBeVisible()
  })

  it('does not fit: names the closest candidate and the shortfall', () => {
    const m3 = gpu('apple-m3-ultra')
    const r = size({ model: model('zai-org-glm-5.2'), gpu: m3, quantization: 'fp16' })
    const closest = r.closest
    if (r.fits || !closest) throw new Error('expected no fit')
    show(r, m3)
    expect(
      screen.getByText(
        new RegExp(
          `^Does not fit: closest is TP ${closest.candidate.tp} × ${closest.candidate.nodes}, lacking \\d+\\.\\d GB per GPU$`,
        ),
      ),
    ).toBeVisible()
  })

  it('shows the tier assumption beside the open-session count when a tier is used', () => {
    const tier: KVTierSettings = { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 0.25 }
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300, kvTier: tier })
    const c = r.chosen
    if (!c?.tierUsed) throw new Error('expected a tier-backed fit')
    show(r, gb300, tier)
    expect(
      screen.getByText(
        new RegExp(
          `${c.sessionsPerReplica} open sessions per replica \\(${c.hotSessions} in HBM, the rest parked in`,
        ),
      ),
    ).toBeVisible()
    expect(screen.getByText('assumes 25% of sessions decode at once')).toBeVisible()
    expect(screen.getByText(/Resume .* vs recompute/)).toBeVisible()
  })

  it('says so when the tier cannot sustain the churn', () => {
    const tier: KVTierSettings = {
      ...DEFAULT_KV_TIER,
      tier: 'network',
      activeShare: 1,
      burstSeconds: 1,
    }
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300, kvTier: tier })
    if (!r.chosen?.tierOverloaded) throw new Error('fixture no longer overloads the tier')
    show(r, gb300, tier)
    expect(
      screen.getByText('Tier cannot sustain the session churn; sized on HBM only'),
    ).toBeVisible()
  })
})
