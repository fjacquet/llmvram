import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { defaultMix } from '@engines/context-mix'
import { resolveFabricSpec } from '@engines/fabric'
import { DEFAULT_KV_TIER, type KVTierSettings } from '@engines/kv-tier'
import { sizeDeployment } from '@engines/sizing'
import type { SizingInput, SizingResult } from '@engines/types'
import { render, screen } from '@testing-library/react'
import { formatDuration } from '@utils/formatDuration'
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
    contextMix: defaultMix(32768),
    ttftTargetSeconds: null,
    ...patch,
  })
}
const show = (
  result: SizingResult,
  g = gb300,
  mix = defaultMix(32768),
  target: number | null = null,
) =>
  render(
    <SizingResults
      result={result}
      gpu={g}
      sequenceLength={32768}
      contextMix={mix}
      ttftTargetSeconds={target}
    />,
  )

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
    show(r)
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
    show(r)
    expect(
      screen.getByText('Tier cannot sustain the session churn; sized on HBM only'),
    ).toBeVisible()
  })

  // 2026-11-11, sizing final review I1: a decode batch above vLLM's default max_num_seqs
  const wide = () =>
    size({
      model: model('meta-llama-llama-3.1-8b'),
      gpu: gb300,
      sequenceLength: 512,
      contextMix: defaultMix(512),
      users: 20000,
    })

  it('notes --max-num-seqs when more than 256 sequences decode at once (soft, not an alert)', () => {
    const r = wide()
    const n = r.chosen?.decodeBatch ?? 0
    expect(n).toBeGreaterThan(256)
    show(r)
    const note = screen.getByText(new RegExp(`^${n} sequences decode at once per replica`))
    expect(note).toHaveTextContent('--max-num-seqs')
    expect(note).toHaveTextContent('per-user speed beyond 128 rests on 1 measured point')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('shows no batch note at 256 sequences or fewer', () => {
    show(size({ model: model('zai-org-glm-5.2'), gpu: gb300 }))
    expect(screen.queryByText(/sequences decode at once per replica/)).toBeNull()
  })

  it('prints the solver active share, not the raw store value', () => {
    const tier: KVTierSettings = { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 0.001 }
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300, kvTier: tier })
    if (!r.chosen?.tierUsed) throw new Error('expected a tier-backed fit')
    expect(r.chosen.activeShare).toBe(0.01)
    show(r)
    expect(screen.getByText('assumes 1% of sessions decode at once')).toBeVisible()
  })

  it('default mix, no target: sized by memory, no target line, no mix note', () => {
    show(size({ model: model('zai-org-glm-5.2'), gpu: gb300 }))
    expect(screen.getByTestId('sized-by')).toHaveTextContent(/^Sized by memory$/)
    expect(screen.queryByText(/^Target /)).toBeNull()
    expect(
      screen.getByText(
        /^worst case: all \d+ prompts at once, first come first served, no prefix cache$/,
      ),
    ).toBeVisible()
    expect(screen.queryByText(/expected/i)).toBeNull()
  })

  it('a mix names the memory basis', () => {
    const mix = [
      { tokens: 4096, share: 0.9 },
      { tokens: 262144, share: 0.1 },
    ]
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300, contextMix: mix })
    show(r, gb300, mix)
    expect(screen.getByTestId('sized-by')).toHaveTextContent(
      'memory sized on the mix average, mix: 90% 4K, 10% 256K',
    )
  })

  it('a target that binds: Sized by TTFT target, target line and max users', () => {
    const m = model('zai-org-glm-5.2')
    const lone = size({ model: m, gpu: gb300, users: 1 }).chosen?.ttft.lastExpected.toNumber()
    if (!lone) throw new Error('no lone ttft')
    const r = size({ model: m, gpu: gb300, ttftTargetSeconds: lone * 2 })
    if (r.chosen?.boundBy !== 'ttft') throw new Error('expected a TTFT-bound sizing')
    show(r, gb300, defaultMix(32768), lone * 2)
    expect(screen.getByTestId('sized-by')).toHaveTextContent(/^Sized by TTFT target$/)
    expect(screen.getByText(/^Max users within target: \d+$/)).toBeVisible()
    expect(screen.getByText(/^Target .*: (met|missed by .*)$/)).toBeVisible()
  })

  it('never meets the target: says so, never "0 racks"', () => {
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300, ttftTargetSeconds: 1e-6 })
    expect(r.neverMeetsTarget).not.toBeNull()
    show(r, gb300, defaultMix(32768), 1e-6)
    expect(
      screen.getByRole('heading', { name: /^No rack replica meets .* at 32K tokens$/ }),
    ).toBeVisible()
    expect(screen.queryByText(/0 racks/)).toBeNull()
  })

  it('headline is the first-served figure, equal to the single-prompt TTFT', () => {
    const r = size({ model: model('zai-org-glm-5.2'), gpu: gb300 })
    const c = r.chosen
    if (!c) throw new Error('expected a fit')
    expect(c.ttft.firstExpected.toNumber()).toBe(c.timeToFirstToken.toNumber())
    expect(c.ttft.firstCeiling.toNumber()).toBe(c.ceilingTimeToFirstToken.toNumber())
    show(r)
    expect(screen.getByText(formatDuration(c.timeToFirstToken))).toBeVisible()
  })
})
