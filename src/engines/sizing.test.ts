import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import type { GPU, Model } from '@utils/schemas'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { fitsInference, maxConcurrentSessions } from './concurrency'
import { isRackGPU, validTPDegrees } from './config-rules'
import { defaultMix } from './context-mix'
import { resolveFabricSpec } from './fabric'
import { calculateInferenceVRAM } from './inference'
import { DEFAULT_KV_TIER, kvTierSummary } from './kv-tier'
import { calculateMultiGPUVRAM } from './multi-gpu'
import { estimatePerformance } from './performance'
import { evaluateReplica, perGPUBreakdown, replicaCandidates, sizeDeployment } from './sizing'
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
  kvTier: DEFAULT_KV_TIER,
  contextMix: defaultMix(8192),
  ttftTargetSeconds: null,
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

describe('KV storage tier', () => {
  const glm = model('zai-org-glm-5.2')
  const tiered = (tier: 'host-grace' | 'network', customGBps: number | null = null) => ({
    ...input(glm, gb300, 2000, 'fp8'),
    sequenceLength: 32768,
    kvTier: { ...DEFAULT_KV_TIER, tier, customGBps },
  })
  const none = { ...tiered('host-grace'), kvTier: DEFAULT_KV_TIER }

  it('tier none: hot = open = decode batch', () => {
    const c = sizeDeployment(none).chosen
    expect(c?.hotSessions).toBe(c?.sessionsPerReplica)
    expect(c?.decodeBatch).toBe(c?.hotSessions)
    expect(c?.tierUsed).toBeNull()
    expect(c?.tierOverloaded).toBe(false)
    expect(c?.resumeSeconds).toBeNull()
  })

  it('host-grace holds sessionsHeld open sessions, decodes a share of them', () => {
    const r = sizeDeployment(tiered('host-grace'))
    const c = r.chosen
    const base = sizeDeployment(none)
    if (!c) throw new Error('no chosen')
    const one = evaluateReplica({ ...none, users: 1 }, c.tp, c.nodes)
    expect(c.tierUsed).toBe('host-grace')
    expect(c.sessionsPerReplica).toBeGreaterThan(c.hotSessions)
    expect(c.decodeBatch).toBe(Math.min(c.hotSessions, Math.ceil(c.sessionsPerReplica * 0.25)))
    expect(c.resumeSeconds).toBeGreaterThan(0)
    expect(one.hotSessions).toBe(1)
    expect(r.units).toBeLessThanOrEqual(base.units)
  })

  // Mirrors evaluateReplica's one-session inputs for tp 1 on one GPU (multi null)
  const summaryFor = (inp: SizingInput, capacityTB: number | null) => {
    const single = calculateInferenceVRAM({
      model: inp.model,
      quantization: inp.quantization,
      sequenceLength: inp.sequenceLength,
      batchSize: 1,
      kvQuantization: inp.kvQuantization,
      concurrentUsers: 1,
    })
    const hotMax = maxConcurrentSessions({
      totalPerGPUGB: single.total.toNumber(),
      kvPerGPUGB: single.kvCache.toNumber(),
      concurrentUsers: 1,
      gpuVramGB: inp.gpu.vram_gb,
    })
    return kvTierSummary({
      settings: { ...inp.kvTier, capacityTB },
      maxHotSessions: hotMax ?? 0,
      perGPUKVGB: single.kvCache.toNumber(),
      totalKVGB: single.kvCache.toNumber(),
      concurrentUsers: 1,
      multi: { strategy: 'tensor-parallel', gpusPerNode: 1, numNodes: 1, numGPUs: 1 },
      recomputeSeconds: null,
      gpuId: inp.gpu.id,
    })
  }
  const pcie = (users: number, capacityTB: number | null = null): SizingInput => ({
    ...input(llama8b, h100, users),
    kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie', capacityTB },
  })

  it('open sessions = min(users, kvTierSummary sessionsHeld)', () => {
    const inp = pcie(100000)
    expect(evaluateReplica(inp, 1, 1).sessionsPerReplica).toBe(
      Math.min(100000, summaryFor(inp, null)?.sessionsHeld ?? 0),
    )
  })

  it('capacityTB is per unit: a tp 1 replica on HGX gets 1/8', () => {
    const inp = pcie(100000, 0.08)
    const c = evaluateReplica(inp, 1, 1)
    const perReplica = summaryFor(inp, 0.08 / 8)?.sessionsHeld
    const perServer = summaryFor(inp, 0.08)?.sessionsHeld
    expect(perReplica).toBeLessThan(perServer ?? 0)
    expect(c.sessionsPerReplica).toBe(perReplica)
    const scaled = sizeDeployment(inp)
    const unscaled = sizeDeployment({ ...inp, kvTier: { ...inp.kvTier, capacityTB: 0.08 * 8 } })
    expect(scaled.units).toBeGreaterThanOrEqual(unscaled.units)
  })

  it('nothing parked: speed equals tier none', () => {
    const t = evaluateReplica(pcie(20), 1, 1)
    const n = evaluateReplica(input(llama8b, h100, 20), 1, 1)
    expect(t.tierUsed).toBeNull()
    expect(t.decodeBatch).toBe(t.sessionsPerReplica)
    expect(t.perUserTokensPerSecond.toNumber()).toBe(n.perUserTokensPerSecond.toNumber())
  })

  it('a tier never needs more units than none (TP may change)', () => {
    expect(sizeDeployment(pcie(1000)).units).toBeLessThanOrEqual(
      sizeDeployment(input(llama8b, h100, 1000)).units,
    )
  })

  it('overloaded tier falls back to the HBM count', () => {
    const c = sizeDeployment(tiered('network', 0.001)).chosen
    expect(c?.tierOverloaded).toBe(true)
    expect(c?.tierUsed).toBeNull()
    expect(c?.sessionsPerReplica).toBe(c?.hotSessions)
    expect(c?.decodeBatch).toBe(c?.hotSessions)
  })

  it('per-user speed is aggregate at the decode batch', () => {
    const inp = {
      ...input(llama8b, h100, 1000),
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-pcie' as const },
    }
    const c = evaluateReplica(inp, 1, 1)
    expect(c.decodeBatch).toBeLessThan(c.sessionsPerReplica)
    const direct = estimatePerformance({
      model: llama8b,
      gpu: h100,
      quantization: 'fp16',
      sequenceLength: 8192,
      batchSize: c.decodeBatch,
      kvQuantization: 'fp16',
      multiGPUResult: null,
    })
    expect(c.perUserTokensPerSecond.toNumber()).toBeCloseTo(
      direct.expected.tokensPerSecond.div(c.decodeBatch).toNumber(),
      6,
    )
  })
})

describe('activations at the real batch (final review C2)', () => {
  const llama70 = model('meta-llama-llama-3.3-70b')
  const at = (g: string, ctx: number, users: number, q: SizingInput['quantization']) =>
    sizeDeployment({
      ...input(llama70, gpu(g), users, q),
      sequenceLength: ctx,
      contextMix: defaultMix(ctx),
    })

  // 2026-11-11, sizing final review C2: was 1 server, now 2 because the one-session
  // breakdown sized activations at batch 1; at 100 users they hit their cap and the
  // per-GPU fixed cost no longer leaves room for the KV the count assumed.
  it('Llama 3.3 70B int4 on RTX 4090, 4096 context, 100 users: 2 servers', () => {
    expect(at('nvidia-rtx-4090', 4096, 100, 'int4').units).toBe(2)
  })

  // 2026-11-11, sizing final review C2: was 3 servers, now 4, same cause.
  it('Llama 3.3 70B fp8 on L40S, 1024 context, 3000 users: 4 servers', () => {
    expect(at('nvidia-l40s', 1024, 3000, 'fp8').units).toBe(4)
  })

  it('the chosen candidate fits at its real batch', () => {
    for (const [g, ctx, users, q] of [
      ['nvidia-rtx-4090', 4096, 100, 'int4'],
      ['nvidia-l40s', 1024, 3000, 'fp8'],
      ['nvidia-h100-80gb-sxm', 8192, 500, 'fp16'],
    ] as const) {
      const r = at(g, ctx, users, q)
      const c = r.chosen
      expect(c).not.toBeNull()
      if (!c) continue
      const inp = {
        ...input(llama70, gpu(g), users, q),
        sequenceLength: ctx,
        contextMix: defaultMix(ctx),
      }
      const real = perGPUBreakdown(inp, c.tp, c.nodes, c.hotSessions)
      expect(fitsInference(real.perGPU.total.toNumber(), inp.gpu.vram_gb)).toBe(true)
    }
  })
})

describe('active share is the solver value (final review M2)', () => {
  it('carries the clamped share, not the raw store value', () => {
    const raw = { ...input(llama8b, h100, 1000), kvTier: { ...DEFAULT_KV_TIER, activeShare: 5 } }
    expect(evaluateReplica(raw, 1, 1).activeShare).toBe(1)
    expect(evaluateReplica(input(llama8b, h100, 1000), 1, 1).activeShare).toBe(0.25)
  })
})

describe('final review I4: Review Focus cases', () => {
  it('DSA model with FP8 KV: hot sessions equal the Inference composition at the same replica', () => {
    const glm = model('zai-org-glm-5.2')
    const users = 1000
    const inp = {
      ...input(glm, gb300, users, 'fp8'),
      kvQuantization: 'fp8' as const,
      sequenceLength: 32768,
      contextMix: defaultMix(32768),
    }
    const c = evaluateReplica(inp, 8, 1)
    // Inference path: the same model, one replica of TP 8, activations at the real batch
    const vram = calculateInferenceVRAM({
      model: glm,
      quantization: 'fp8',
      sequenceLength: 32768,
      batchSize: users,
      kvQuantization: 'fp8',
      concurrentUsers: 1,
    })
    const multi = calculateMultiGPUVRAM(
      vram,
      glm,
      gb300.vram_gb,
      8,
      'tensor-parallel',
      gb300,
      'fp8',
    )
    const sessions = maxConcurrentSessions({
      totalPerGPUGB: multi.perGPU.total.toNumber(),
      kvPerGPUGB: multi.perGPU.kvCache.toNumber(),
      concurrentUsers: 1,
      gpuVramGB: gb300.vram_gb,
    })
    expect(sessions).not.toBeNull()
    expect(c.hotSessions).toBe(Math.min(users, sessions ?? users))
    // 2026-11-11, sizing final review I4: 97 sessions, as the probe found, recomputed by the engine
    expect(c.hotSessions).toBe(97)
  })

  it('a single-GPU workstation part cannot hold Kimi K2: fits false, TP1 x 1, lacking GB', () => {
    const r = sizeDeployment(input(model('moonshotai-kimi-k2-instruct'), gpu('nvidia-gb10'), 10))
    expect(r.fits).toBe(false)
    expect(r.closest?.candidate).toMatchObject({ tp: 1, nodes: 1 })
    expect(r.closest?.lackingGB).toBeGreaterThan(0)
  })
})

// Roster refresh (2026-11): each added model sizes on one server GPU and one rack GPU, no NaN
describe.each([
  'zai-org-glm-5.3',
  'moonshotai-kimi-k3.1-pro',
  'deepseek-ai-deepseek-v4.1-flash-1017',
  'inclusionai-ling-3.1-flash-1019',
])('%s sizes', (id) => {
  it.each([
    ['server', 'nvidia-h100-80gb-sxm'],
    ['rack', 'nvidia-gb300-nvl72'],
  ])('on a %s GPU', (unit, gpuId) => {
    const r = sizeDeployment(input(model(id), gpu(gpuId), 20, 'fp8'))
    expect(r.fits).toBe(true)
    expect(r.unit).toBe(unit)
    expect(Number.isFinite(r.units) && r.units > 0).toBe(true)
    expect(Number.isFinite(r.chosen?.perUserTokensPerSecond.toNumber() ?? Number.NaN)).toBe(true)
  })
})

describe('TTFT target (rack GPU)', () => {
  const llama70 = model('meta-llama-llama-3.3-70b')
  const base = (users: number): SizingInput => ({
    ...input(llama70, gb300, users, 'fp8' as never),
    sequenceLength: 4096,
    contextMix: defaultMix(4096),
  })
  const lone = sizeDeployment(base(2000)).chosen?.classes[0]?.expectedTTFT.toNumber() ?? 0
  const none = sizeDeployment(base(2000))

  it('no target: nothing is bounded by TTFT', () => {
    expect(none.chosen?.burstCap).toBeNull()
    expect(none.chosen?.boundBy).toBe('memory')
    expect(none.neverMeetsTarget).toBeNull()
  })

  it('a tight target caps users per replica and never lowers the unit count', () => {
    const r = sizeDeployment({ ...base(2000), ttftTargetSeconds: lone * 2 })
    expect(r.chosen?.boundBy).toBe('ttft')
    expect(r.chosen?.sessionsPerReplica).toBeLessThanOrEqual(2)
    expect(r.units).toBeGreaterThanOrEqual(none.units)
    expect(r.chosen?.ttft.lastExpected.toNumber()).toBeLessThanOrEqual(lone * 2 + 1e-9)
  })

  it('a generous target changes nothing', () => {
    const r = sizeDeployment({ ...base(2000), ttftTargetSeconds: 3600 })
    expect(r.units).toBe(none.units)
    expect(r.replicas).toBe(none.replicas)
    expect(r.chosen?.sessionsPerReplica).toBe(none.chosen?.sessionsPerReplica)
    expect(r.chosen?.boundBy).toBe('memory')
  })

  it('below a lone prompt: neverMeetsTarget names the largest class', () => {
    const mix = [
      { tokens: 4096, share: 0.9 },
      { tokens: 262144, share: 0.1 },
    ]
    const r = sizeDeployment({
      ...base(100),
      sequenceLength: 262144,
      contextMix: mix,
      ttftTargetSeconds: 1e-6,
    })
    expect(r.neverMeetsTarget).toEqual({ largestClassTokens: 262144 })
    expect(r.chosen).toBeNull()
  })

  it('with a tier the burst is the decode batch, not the open sessions', () => {
    const r = sizeDeployment({
      ...base(2000),
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace', activeShare: 0.1 },
      ttftTargetSeconds: lone * 20,
    })
    const c = r.chosen
    expect(c?.tierUsed).toBe('host-grace')
    expect(c?.ttft.burst).toBe(c?.decodeBatch)
    expect(c?.decodeBatch).toBeLessThanOrEqual(c?.burstCap ?? 0)
    expect(c?.sessionsPerReplica).toBeGreaterThan(c?.decodeBatch ?? 0)
  })
})
