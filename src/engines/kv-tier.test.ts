import type { GPU, Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { calculateInferenceVRAM } from './inference'
import {
  DEFAULT_KV_TIER,
  type KVTierSettings,
  kvTierSummary,
  resumeSeconds,
  sessionKVLayout,
  tierBandwidthGBps,
} from './kv-tier'
import { calculateMultiGPUVRAM } from './multi-gpu'
import { estimatePerformance } from './performance'

const network: KVTierSettings = { ...DEFAULT_KV_TIER, tier: 'network' }

const base = {
  settings: network,
  maxHotSessions: 100,
  kvPerSessionPerGPUGB: 2,
  gpusPerSession: 4,
  kvPerSessionGB: 8,
  totalGPUs: 4,
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
    expect(tierBandwidthGBps({ ...network, customGBps: 0 })).toBe(12.5)
    expect(tierBandwidthGBps({ ...network, customGBps: null })).toBe(12.5)
  })
})

describe('resumeSeconds', () => {
  it('reproduces the Dell ObjectScale anchor within 10%', () => {
    // Dell: 43 GB KV at 235K tokens, TP4 on XE9680, 837 ms to first token.
    // Per GPU 43 / 4 GB at 12.8 GB/s per GPU (>= 51 GB/s per server).
    const s = resumeSeconds(43 / 4, 12.8)
    expect(s).toBeGreaterThan(0.837 * 0.9)
    expect(s).toBeLessThan(0.837 * 1.1)
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
    // 100 hot in HBM + 1 TB / 8 GB = 125 parked = 225 (below 100 / 0.25 = 400)
    const s = kvTierSummary({ ...base, settings: { ...network, capacityTB: 1 } })
    expect(s?.sessionsHeld).toBe(225)
  })

  it('never holds fewer than fit in HBM, even with a tiny tier', () => {
    const s = kvTierSummary({ ...base, settings: { ...network, capacityTB: 0.001 } })
    expect(s?.sessionsHeld).toBe(100)
  })

  it('holds 0 when nothing fits in HBM', () => {
    expect(kvTierSummary({ ...base, maxHotSessions: 0 })?.sessionsHeld).toBe(0)
  })

  it('clamps an active share of 0 to 1%', () => {
    const s = kvTierSummary({ ...base, settings: { ...network, activeShare: 0 } })
    expect(s?.sessionsHeld).toBe(10000)
  })

  it('prices tier traffic as resumes per second times one session of KV', () => {
    // 400 held x 25% active / 30 s burst = 3.33 resumes/s x 8 GB = 26.7 GB/s
    const s = kvTierSummary(base)
    expect(s?.trafficGBps).toBeCloseTo(((400 * 0.25) / 30) * 2 * 4, 6)
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
      kvPerSessionPerGPUGB: multi.perGPU.kvCache.toNumber(),
      gpusPerSession: 4,
      kvPerSessionGB: single.kvCache.toNumber(),
      totalGPUs: 4,
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
