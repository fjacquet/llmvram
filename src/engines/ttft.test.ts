import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { validateGPUs, validateModels } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { FABRIC_SPECS } from './fabric'
import { calculateInferenceVRAM } from './inference'
import { calculateMultiNodeVRAM } from './multi-node'
import { estimatePerformance } from './performance'
import { burstTTFT, classPrefills, maxUsersWithinTarget, meetsTarget } from './ttft'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const find = <T extends { id: string }>(xs: T[], id: string): T => {
  const x = xs.find((v) => v.id === id)
  if (!x) throw new Error(`fixture ${id}`)
  return x
}
const model = find(models, 'meta-llama-llama-3.1-70b')
const gpu = find(gpus, 'nvidia-h100-80gb-sxm')
const base = { model, gpu, quantization: 'fp16' as const }
const ttft = (tokens: number) =>
  estimatePerformance({ ...base, batchSize: 1, sequenceLength: tokens }).expected.timeToFirstToken

describe('burstTTFT', () => {
  it('one class, burst 1: last equals first equals the engine batch-1 TTFT', () => {
    const b = burstTTFT(classPrefills(base, [{ tokens: 4096, share: 1 }]), 1)
    expect(b.lastExpected.eq(b.firstExpected)).toBe(true)
    expect(b.firstExpected.eq(ttft(4096))).toBe(true)
  })

  it('one class, burst 100: last is 100 x first', () => {
    const b = burstTTFT(classPrefills(base, [{ tokens: 4096, share: 1 }]), 100)
    expect(b.lastExpected.toNumber()).toBeCloseTo(b.firstExpected.times(100).toNumber(), 9)
  })

  it('burst below 1 clamps to 1', () => {
    expect(burstTTFT(classPrefills(base, [{ tokens: 4096, share: 1 }]), 0).burst).toBe(1)
  })

  it('mix 90/10, burst 10: last is the largest class + 9 x the share-weighted TTFT', () => {
    const b = burstTTFT(
      classPrefills(base, [
        { tokens: 4096, share: 0.9 },
        { tokens: 262144, share: 0.1 },
      ]),
      10,
    )
    const mean = ttft(4096).times(0.9).plus(ttft(262144).times(0.1))
    const want = ttft(262144).plus(mean.times(9))
    expect(b.lastExpected.toNumber()).toBeCloseTo(want.toNumber(), 9)
    expect(b.firstExpected.eq(ttft(262144))).toBe(true)
  })

  it('last >= first for a skewed mix at burst 1 and 2', () => {
    const classes = classPrefills(base, [
      { tokens: 4096, share: 0.99 },
      { tokens: 131072, share: 0.01 },
    ])
    for (const n of [1, 2]) {
      const b = burstTTFT(classes, n)
      expect(b.lastExpected.gte(b.firstExpected)).toBe(true)
      expect(b.lastCeiling.gte(b.firstCeiling)).toBe(true)
    }
    expect(burstTTFT(classes, 1).lastExpected.eq(ttft(131072))).toBe(true)
  })

  it('skewed mix: cap is 0 below the largest prompt, else (T - largest) / mean + 1', () => {
    const classes = classPrefills(base, [
      { tokens: 4096, share: 0.99 },
      { tokens: 131072, share: 0.01 },
    ])
    const big = ttft(131072).toNumber()
    const mean = ttft(4096).times(0.99).plus(ttft(131072).times(0.01)).toNumber()
    expect(maxUsersWithinTarget(classes, big * 0.99)).toBe(0)
    expect(maxUsersWithinTarget(classes, big + mean * 4.5)).toBe(5)
  })

  it('two-class multi-server: last is largest + (B - 1) x mean', () => {
    const m = find(models, 'meta-llama-llama-3.1-405b')
    const singleGPU = calculateInferenceVRAM({
      model: m,
      quantization: 'fp16',
      sequenceLength: 32768,
      batchSize: 1,
    })
    const multi = calculateMultiNodeVRAM({
      singleGPU,
      model: m,
      gpuVramGB: gpu.vram_gb,
      gpusPerNode: 8,
      numNodes: 2,
      intraNodeStrategy: 'tensor-parallel',
      gpu,
      fabric: FABRIC_SPECS['infiniband-ndr'],
      batchSize: 1,
      sequenceLength: 32768,
      quantization: 'fp16',
    })
    const cs = classPrefills({ model: m, gpu, quantization: 'fp16', multiGPUResult: multi }, [
      { tokens: 4096, share: 0.5 },
      { tokens: 32768, share: 0.5 },
    ])
    const [a, b] = cs
    const mean = (a?.expectedTTFT.plus(b?.expectedTTFT ?? 0) ?? new Decimal(0)).div(2)
    expect(burstTTFT(cs, 4).lastExpected.toNumber()).toBeCloseTo(
      (b?.expectedTTFT.plus(mean.times(3)) ?? new Decimal(0)).toNumber(),
      9,
    )
  })

  it('multi-server batch 1: classes equal the engine at batchSize 1, hop counted once per prompt', () => {
    const m = find(models, 'meta-llama-llama-3.1-405b')
    const singleGPU = calculateInferenceVRAM({
      model: m,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })
    const multi = calculateMultiNodeVRAM({
      singleGPU,
      model: m,
      gpuVramGB: gpu.vram_gb,
      gpusPerNode: 8,
      numNodes: 2,
      intraNodeStrategy: 'tensor-parallel',
      gpu,
      fabric: FABRIC_SPECS['infiniband-ndr'],
      batchSize: 1,
      sequenceLength: 4096,
      quantization: 'fp16',
    })
    const p = { model: m, gpu, quantization: 'fp16' as const, multiGPUResult: multi }
    const [c] = classPrefills(p, [{ tokens: 4096, share: 1 }])
    const direct = estimatePerformance({ ...p, batchSize: 1, sequenceLength: 4096 })
    expect(c?.expectedTTFT.eq(direct.expected.timeToFirstToken)).toBe(true)
    expect(c?.ceilingTTFT.eq(direct.timeToFirstToken)).toBe(true)
    const b = burstTTFT(c ? [c] : [], 3)
    expect(b.lastExpected.toNumber()).toBeCloseTo(c?.expectedTTFT.times(3).toNumber() ?? 0, 9)
  })

  it('M5 Max, Qwen3.6 35B A3B NVFP4, 262144 tokens: 29.0 min (ceiling 13.6 min); 10 s target misses', () => {
    const q = find(models, 'qwen-qwen3.6-35b-a3b')
    const m5 = find(gpus, 'apple-m5-max')
    const p = estimatePerformance({
      model: q,
      gpu: m5,
      quantization: 'nvfp4',
      batchSize: 1,
      sequenceLength: 262144,
    })
    expect(p.expected.timeToFirstToken.toNumber() / 60).toBeCloseTo(29.0, 1)
    expect(p.timeToFirstToken.toNumber() / 60).toBeCloseTo(13.6, 1)
    const classes = classPrefills({ model: q, gpu: m5, quantization: 'nvfp4' }, [
      { tokens: 262144, share: 1 },
    ])
    expect(maxUsersWithinTarget(classes, 10)).toBe(0)
    expect(burstTTFT(classes, 1).lastExpected.toNumber()).toBeGreaterThan(10)
    expect(meetsTarget(burstTTFT(classes, 100), 10)).toBe(false)
  })

  it('cap ties to the queue: last(cap) <= target < last(cap + 1), one class and skewed mix', () => {
    const t = ttft(4096).toNumber()
    const one = classPrefills(base, [{ tokens: 4096, share: 1 }])
    const skew = classPrefills(base, [
      { tokens: 4096, share: 0.99 },
      { tokens: 131072, share: 0.01 },
    ])
    const big = ttft(131072).toNumber()
    const mean = ttft(4096).times(0.99).plus(ttft(131072).times(0.01)).toNumber()
    const cases: [typeof one, number][] = [
      [one, t * 3], // exact multiple of one prompt
      [one, t * 3.5],
      [skew, big + mean * 4], // exact: largest + 4 means
      [skew, big + mean * 4.5],
    ]
    for (const [cs, target] of cases) {
      const cap = maxUsersWithinTarget(cs, target)
      expect(cap).toBeGreaterThan(0)
      expect(burstTTFT(cs, cap).lastExpected.toNumber()).toBeLessThanOrEqual(target + 1e-9)
      expect(burstTTFT(cs, cap + 1).lastExpected.toNumber()).toBeGreaterThan(target - 1e-9)
    }
    expect(maxUsersWithinTarget(one, t * 3)).toBe(3)
    expect(maxUsersWithinTarget(skew, big + mean * 4)).toBe(5)
  })

  it('meetsTarget is null without a target; maxUsersWithinTarget floors target / mean', () => {
    const classes = classPrefills(base, [{ tokens: 4096, share: 1 }])
    expect(meetsTarget(burstTTFT(classes, 5), null)).toBeNull()
    const t = ttft(4096).toNumber()
    expect(maxUsersWithinTarget(classes, t * 3.5)).toBe(3)
  })

  it.todo(
    'calibration: InferenceX rows with c > 1 are TP4/TP8 runs of "llama70b" (3.1 vs 3.3 unstated) with no recorded multiGPUResult, so they cannot be rebuilt from models.json and gpus.json alone',
  )
})
