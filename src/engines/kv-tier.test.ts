import type { GPU, Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { calculateInferenceVRAM } from './inference'
import {
  clampKVTier,
  DEFAULT_KV_TIER,
  hasGraceHost,
  type KVTierSettings,
  kvTierSummary,
  resetTierForGPU,
  resumeSeconds,
  sessionKVLayout,
  tierBandwidthGBps,
} from './kv-tier'
import { calculateMultiGPUVRAM } from './multi-gpu'
import { estimatePerformance } from './performance'

const network: KVTierSettings = { ...DEFAULT_KV_TIER, tier: 'network' }

// 4-GPU tensor parallel, one session: 2 GB on each GPU, 8 GB stored
const base = {
  settings: network,
  maxHotSessions: 100,
  perGPUKVGB: 2,
  totalKVGB: 8,
  concurrentUsers: 1,
  multi: { strategy: 'tensor-parallel' as const, gpusPerNode: 4, numNodes: 1, numGPUs: 4 },
  recomputeSeconds: 5,
}

describe('tierBandwidthGBps', () => {
  it('is null with no tier', () => {
    expect(tierBandwidthGBps(DEFAULT_KV_TIER)).toBeNull()
  })

  it('uses the preset, or a positive custom value', () => {
    expect(tierBandwidthGBps(network)).toBe(12.5)
    expect(tierBandwidthGBps({ ...network, customGBps: 40 })).toBe(40)
  })

  it('falls back to the preset when the custom value is 0 or empty', () => {
    expect(tierBandwidthGBps(clampKVTier({ ...network, customGBps: 0 }))).toBe(12.5)
    expect(tierBandwidthGBps({ ...network, customGBps: null })).toBe(12.5)
  })
})

describe('resumeSeconds', () => {
  it('reads GiB of KV over decimal GB/s', () => {
    // 1 GiB = 1.073741824 GB
    expect(resumeSeconds(1, 1)).toBeCloseTo(0.03 + 1.073741824, 9)
  })

  it('reproduces the Dell ObjectScale anchor within 10%', () => {
    // Dell: 43 GB KV at 235K tokens, TP4 on XE9680, 837 ms to first token.
    // Engine KV is in GiB: 43e9 bytes / 1024^3 per GPU share, at 12.8 GB/s per GPU.
    const s = resumeSeconds(43e9 / 1024 ** 3 / 4, 12.8)
    expect(s).toBeGreaterThan(0.837 * 0.9)
    expect(s).toBeLessThan(0.837 * 1.1)
  })
})

describe('hasGraceHost', () => {
  it('is true for both Grace-host GPU ids', () => {
    expect(hasGraceHost('nvidia-gb300-nvl72')).toBe(true)
    expect(hasGraceHost('nvidia-gb300-desktop-252gb')).toBe(true)
  })

  it('is false for the x86 HGX B300, GB10 unified memory, and non-Grace GPUs', () => {
    expect(hasGraceHost('nvidia-gb300-288gb')).toBe(false)
    expect(hasGraceHost('nvidia-gb10')).toBe(false)
    expect(hasGraceHost('nvidia-h100-80gb-sxm')).toBe(false)
  })
})

describe('resetTierForGPU', () => {
  const grace: KVTierSettings = { ...DEFAULT_KV_TIER, tier: 'host-grace' }

  it('falls back to none when the new GPU has no Grace host', () => {
    expect(resetTierForGPU(grace, 'nvidia-h100-80gb-sxm').tier).toBe('none')
    expect(resetTierForGPU(grace, null).tier).toBe('none')
  })

  it('keeps host-grace when the new GPU still has a Grace host', () => {
    expect(resetTierForGPU(grace, 'nvidia-gb300-nvl72').tier).toBe('host-grace')
  })

  it('leaves a non-host-grace tier untouched regardless of the GPU', () => {
    expect(resetTierForGPU(network, 'nvidia-h100-80gb-sxm')).toBe(network)
  })
})

describe('clampKVTier', () => {
  it('bounds active share to 1-100% and burst to at least 1 s', () => {
    expect(clampKVTier({ ...network, activeShare: 0 }).activeShare).toBe(0.01)
    expect(clampKVTier({ ...network, activeShare: 5 }).activeShare).toBe(1)
    expect(clampKVTier({ ...network, burstSeconds: 0 }).burstSeconds).toBe(1)
  })

  it('turns non-finite share or burst back into the defaults', () => {
    const c = clampKVTier({ ...network, activeShare: Number.NaN, burstSeconds: Infinity })
    expect(c.activeShare).toBe(DEFAULT_KV_TIER.activeShare)
    expect(c.burstSeconds).toBe(DEFAULT_KV_TIER.burstSeconds)
  })

  it('keeps bandwidth and capacity only when positive and finite', () => {
    const c = clampKVTier({ ...network, customGBps: 0, capacityTB: -1 })
    expect(c.customGBps).toBeNull()
    expect(c.capacityTB).toBeNull()
    expect(clampKVTier({ ...network, capacityTB: 0.5 }).capacityTB).toBe(0.5)
  })
})

describe('kvTierSummary', () => {
  it('is null with no tier', () => {
    expect(kvTierSummary({ ...base, settings: DEFAULT_KV_TIER })).toBeNull()
  })

  it('holds hot sessions divided by the active share', () => {
    expect(kvTierSummary(base)?.sessionsHeld).toBe(400)
  })

  it('caps the parked sessions by the tier capacity', () => {
    // 100 hot in HBM + 1 TB (1e12 bytes) / 8 GiB = 116 parked = 216 (below 400)
    const s = kvTierSummary({ ...base, settings: { ...network, capacityTB: 1 } })
    expect(s?.sessionsHeld).toBe(216)
  })

  it('never holds fewer than fit in HBM, even with a tiny tier', () => {
    const s = kvTierSummary({ ...base, settings: { ...network, capacityTB: 0.001 } })
    expect(s?.sessionsHeld).toBe(100)
  })

  it('holds 0 when nothing fits in HBM', () => {
    expect(kvTierSummary({ ...base, maxHotSessions: 0 })?.sessionsHeld).toBe(0)
  })

  it('flags churn the tier cannot sustain', () => {
    expect(kvTierSummary(base)?.overloaded).toBe(false)
    const busy = kvTierSummary({ ...base, settings: { ...network, burstSeconds: 1 } })
    expect(busy?.overloaded).toBe(true)
  })

  it('prices tier traffic as resumes per second times one session of KV', () => {
    // 400 held x 25% active / 30 s burst = 3.33 resumes/s x 8 GB = 26.7 GB/s
    const s = kvTierSummary(base)
    // GiB of KV per resume, reported in decimal GB/s like the tier bandwidth
    expect(s?.trafficGBps).toBeCloseTo(((400 * 0.25) / 30) * 2 * 4 * 1.073741824, 6)
    expect(s?.tierGBps).toBe(12.5 * 4)
  })

  it('reports recompute as unknown when prefill time is unknown', () => {
    const s = kvTierSummary({ ...base, recomputeSeconds: null })
    expect(s?.resumeFaster).toBeNull()
  })
})

describe('resume vs recompute, Dell crossover (8-16K tokens)', () => {
  // Qwen3-Coder-30B-A3B-Instruct config.json: 48 layers, 4 KV heads x head_dim 128,
  // 128 experts top-8; safetensors 30.5B. KV = 48 x 2 x 4 x 128 = 49152 per token.
  const coder: Model = {
    id: 'qwen3-coder-30b-a3b',
    name: 'Qwen3 Coder 30B A3B',
    architecture: 'moe',
    num_parameters_billion: 30.5,
    active_parameters_billion: 3.3,
    hidden_size: 2048,
    num_hidden_layers: 48,
    num_attention_heads: 32,
    num_kv_heads: 4,
    intermediate_size: 6144,
    num_experts: 128,
    num_experts_per_token: 8,
    kv_cache_elements_per_token: 49152,
  }
  const h100: GPU = {
    id: 'h100',
    name: 'H100',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 3350,
    memory_type: 'HBM3',
    bus_width: 5120,
    fp16_tflops: 989,
    interconnect: 'nvlink-4',
    tier: 'datacenter',
    max_gpus_per_node: 8,
  }
  const summaryAt = (sequenceLength: number) => {
    const single = calculateInferenceVRAM({
      model: coder,
      quantization: 'bf16',
      sequenceLength,
      batchSize: 1,
    })
    const multi = calculateMultiGPUVRAM(single, coder, 80, 4, 'tensor-parallel', h100)
    const perf = estimatePerformance({
      model: coder,
      gpu: h100,
      quantization: 'bf16',
      batchSize: 1,
      sequenceLength,
      multiGPUResult: multi,
    })
    return kvTierSummary({
      settings: network,
      maxHotSessions: 10,
      perGPUKVGB: multi.perGPU.kvCache.toNumber(),
      totalKVGB: single.kvCache.toNumber(),
      concurrentUsers: 1,
      multi,
      recomputeSeconds: perf.prefillSeconds?.toNumber() ?? null,
    })
  }

  it('resumes faster than recompute at 235K tokens', () => {
    expect(summaryAt(235000)?.resumeFaster).toBe(true)
  })

  it('recomputes faster than resuming at 4K tokens', () => {
    expect(summaryAt(4096)?.resumeFaster).toBe(false)
  })
})

describe('sessionKVLayout', () => {
  it('single GPU: the whole session on one GPU', () => {
    expect(sessionKVLayout({ perGPUKVGB: 10, concurrentUsers: 5, multi: null })).toEqual({
      kvPerSessionPerGPUGB: 2,
      gpusPerSession: 1,
    })
  })

  it('tensor parallel: every GPU holds its share (all of it for duplicated MLA)', () => {
    const multi = { strategy: 'tensor-parallel' as const, gpusPerNode: 8, numNodes: 1, numGPUs: 8 }
    expect(sessionKVLayout({ perGPUKVGB: 10, concurrentUsers: 5, multi })).toEqual({
      kvPerSessionPerGPUGB: 2,
      gpusPerSession: 8,
    })
  })

  it('expert parallel: a session lives on one rank per node and reloads through it', () => {
    // perGPU KV is the average over ranks (sessions spread 1/N); one session is N x that share
    const multi = {
      strategy: 'expert-parallel' as const,
      gpusPerNode: 72,
      numNodes: 1,
      numGPUs: 72,
    }
    expect(sessionKVLayout({ perGPUKVGB: 0.05, concurrentUsers: 1, multi })).toEqual({
      kvPerSessionPerGPUGB: 3.6,
      gpusPerSession: 1,
    })
  })

  it('expert parallel across nodes: one rank in each pipeline stage', () => {
    const multi = { strategy: 'expert-parallel' as const, gpusPerNode: 8, numNodes: 2, numGPUs: 16 }
    expect(sessionKVLayout({ perGPUKVGB: 1, concurrentUsers: 4, multi }).gpusPerSession).toBe(2)
  })
})
