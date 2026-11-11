import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import type { GPU, Model } from '@utils/schemas'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { validTPDegrees } from './config-rules'
import { resolveFabricSpec } from './fabric'
import { estimatePerformance } from './performance'
import { evaluateReplica, isRackGPU, replicaCandidates, sizeDeployment } from './sizing'
import type { SizingInput } from './types'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string): Model => {
  const m = models.find((x) => x.id === id)
  if (!m) throw new Error(`no model ${id}`)
  return m
}
const gpu = (id: string): GPU => {
  const g = gpus.find((x) => x.id === id)
  if (!g) throw new Error(`no gpu ${id}`)
  return g
}
const fabric = resolveFabricSpec('ethernet-800g', null)
const input = (m: Model, g: GPU, users: number, quantization = 'fp16' as const): SizingInput => ({
  model: m,
  gpu: g,
  quantization,
  kvQuantization: 'fp16',
  sequenceLength: 8192,
  users,
  fabric,
})

const h100 = gpu('nvidia-h100-80gb-sxm')
const gb300 = gpu('nvidia-gb300-nvl72')
const llama8b = model('meta-llama-llama-3.1-8b')

describe('replicaCandidates', () => {
  it('rack: every valid TP up to 72, one node', () => {
    const m = model('deepseek-r1')
    expect(replicaCandidates(m, gb300)).toEqual(
      validTPDegrees(m, 72).map((tp) => ({ tp, nodes: 1 })),
    )
  })

  it('server: valid TPs up to 8 plus 2..8 full-server stages', () => {
    const c = replicaCandidates(llama8b, h100)
    expect(c.filter((x) => x.nodes === 1).map((x) => x.tp)).toEqual(validTPDegrees(llama8b, 8))
    expect(c.filter((x) => x.nodes > 1).map((x) => x.nodes)).toEqual([2, 3, 4, 5, 6, 7, 8])
    expect(c.every((x) => x.nodes === 1 || x.tp === 8)).toBe(true)
  })

  it('never offers a TP vLLM refuses', () => {
    const m = model('deepseek-r1')
    expect(replicaCandidates(m, gb300).some((x) => x.tp === 3)).toBe(false)
  })

  it('a one-GPU machine offers only tp 1, nodes 1', () => {
    expect(replicaCandidates(llama8b, gpu('apple-m3-ultra'))).toEqual([{ tp: 1, nodes: 1 }])
  })
})

describe('sizeDeployment', () => {
  it('small model on HGX: tp 1, eight replicas per server', () => {
    const r = sizeDeployment(input(llama8b, h100, 100))
    expect(r.fits).toBe(true)
    expect(r.unit).toBe('server')
    expect(r.chosen?.tp).toBe(1)
    expect(r.replicasPerUnit).toBe(8)
    expect(r.replicas).toBe(2)
    expect(r.units).toBe(1)
    expect(r.units).toBe(Math.ceil(r.replicas / 8))
  })

  it('model larger than one HGX spans servers', () => {
    const r = sizeDeployment(input(model('deepseek-r1'), h100, 10))
    expect(r.fits).toBe(true)
    expect(r.chosen?.nodes ?? 0).toBeGreaterThanOrEqual(2)
    expect(r.replicasPerUnit).toBe(0)
    expect(r.units).toBe(r.replicas * (r.chosen?.nodes ?? 0))
  })

  it.each([
    // 2026-11-11, sizing T4: literals recomputed from the solver; the 48-way split is non-power-of-2
    [
      'deepseek-v4-pro',
      'nvidia-gb300-nvl72',
      { tp: 16, replicasPerUnit: 4, idleGPUsPerUnit: 8, replicas: 3, units: 1 },
    ],
    [
      'moonshotai-kimi-k3',
      'nvidia-gb200-nvl72',
      { tp: 48, replicasPerUnit: 1, idleGPUsPerUnit: 24, replicas: 8, units: 8 },
    ],
  ])('rack split: %s on %s', (m, g, expected) => {
    const r = sizeDeployment(input(model(m), gpu(g), 2000))
    expect(r.unit).toBe('rack')
    expect({
      tp: r.chosen?.tp,
      replicasPerUnit: r.replicasPerUnit,
      idleGPUsPerUnit: r.idleGPUsPerUnit,
      replicas: r.replicas,
      units: r.units,
    }).toEqual(expected)
  })

  it('doubling users doubles replicas within one', () => {
    const a = sizeDeployment(input(llama8b, h100, 1000))
    const b = sizeDeployment(input(llama8b, h100, 2000))
    expect(Math.abs(b.replicas - 2 * a.replicas)).toBeLessThanOrEqual(1)
  })

  it('one user: one replica, one unit, batch-1 speed from the engine', () => {
    const r = sizeDeployment(input(llama8b, h100, 1))
    expect(r.replicas).toBe(1)
    expect(r.units).toBe(1)
    expect(r.chosen?.sessionsPerReplica).toBe(1)
    const direct = estimatePerformance({
      model: llama8b,
      gpu: h100,
      quantization: 'fp16',
      sequenceLength: 8192,
      batchSize: 1,
      kvQuantization: 'fp16',
      multiGPUResult: null,
    })
    expect(r.chosen?.tp).toBe(1)
    expect(r.chosen?.perUserTokensPerSecond.toNumber()).toBe(
      direct.expected.tokensPerSecond.toNumber(),
    )
  })

  it('reports the closest candidate when nothing fits', () => {
    const r = sizeDeployment(input(model('zai-org-glm-5.2'), gpu('nvidia-rtx-4090'), 10))
    expect(r.fits).toBe(false)
    expect(r.units).toBe(0)
    expect(r.chosen).toBeNull()
    expect(r.closest?.lackingGB ?? 0).toBeGreaterThan(0)
  })

  it('a one-GPU machine sizes without dividing by zero', () => {
    const r = sizeDeployment(input(llama8b, gpu('apple-m3-ultra'), 20))
    expect(r.unit).toBe('server')
    expect(r.replicasPerUnit).toBe(1)
    expect(r.idleGPUsPerUnit).toBe(0)
    expect(Number.isFinite(r.units)).toBe(true)
  })

  it('expected speed is below the ceiling', () => {
    const c = sizeDeployment(input(llama8b, h100, 100)).chosen
    expect(c?.perUserTokensPerSecond.lt(c?.perUserCeilingTokensPerSecond ?? 0)).toBe(true)
  })

  it('tie-break: fewest total GPUs, then smallest replica', () => {
    const inp = input(model('meta-llama-llama-3.1-70b'), gb300, 300)
    const ranked = replicaCandidates(inp.model, inp.gpu)
      .map(({ tp, nodes }) => evaluateReplica(inp, tp, nodes))
      .filter((c) => c.fits)
      .map((c) => {
        const replicas = Math.ceil(inp.users / c.sessionsPerReplica)
        const per = Math.floor(72 / c.tp)
        return { c, units: Math.ceil(replicas / per), gpus: replicas * c.gpusPerReplica }
      })
    const best = Math.min(...ranked.map((x) => x.units))
    const tied = ranked.filter((x) => x.units === best)
    const minGpus = Math.min(...tied.map((x) => x.gpus))
    const minTp = Math.min(...tied.filter((x) => x.gpus === minGpus).map((x) => x.c.gpusPerReplica))
    expect(sizeDeployment(inp).chosen?.gpusPerReplica).toBe(minTp)
  })

  it('isRackGPU flags NVL72 only', () => {
    expect(isRackGPU(gb300)).toBe(true)
    expect(isRackGPU(h100)).toBe(false)
  })
})
