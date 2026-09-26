import {
  maxConcurrentSessions,
  perUserTimeToFirstToken,
  perUserTokensPerSecond,
} from '@engines/concurrency'
import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { calculateInferenceVRAM } from './inference'
import { calculateMultiGPUVRAM } from './multi-gpu'

describe('perUserTokensPerSecond', () => {
  it('never lets one user outrun the whole machine', () => {
    // The defect this replaces: batchSize was applied a second time, so at
    // batch 16 / 8 users the panel reported 17188.5 tok/s per user against an
    // aggregate of 8594.2 — each of eight users twice as fast as the machine.
    const aggregate = new Decimal(8594.2)
    const perUser = perUserTokensPerSecond(aggregate, 8)

    expect(perUser.toNumber()).toBeLessThanOrEqual(aggregate.toNumber())
    expect(perUser.toNumber()).toBeCloseTo(1074.275, 3)
  })

  it('splits the aggregate evenly, so the users sum back to it', () => {
    const aggregate = new Decimal(1000)
    for (const users of [1, 2, 8, 37]) {
      const perUser = perUserTokensPerSecond(aggregate, users)
      expect(perUser.mul(users).toNumber()).toBeCloseTo(aggregate.toNumber(), 6)
    }
  })

  it('returns the aggregate unchanged for a single user', () => {
    expect(perUserTokensPerSecond(new Decimal(500), 1).toNumber()).toBe(500)
  })

  it('does not divide by zero on a degenerate user count', () => {
    expect(perUserTokensPerSecond(new Decimal(500), 0).toNumber()).toBe(500)
  })
})

describe('perUserTimeToFirstToken', () => {
  it('is never faster than the idle single-request latency', () => {
    // The defect this replaces: `× users ÷ batch` reported 28.41s per user
    // against a 56.82s single-request TTFT — concurrency making the machine
    // faster than having it to yourself.
    const ttft = new Decimal(56.82)

    for (const [users, batch] of [
      [8, 16],
      [1, 64],
      [2, 2],
      [64, 64],
    ] as const) {
      expect(perUserTimeToFirstToken(ttft, users, batch).toNumber()).toBeGreaterThanOrEqual(
        ttft.toNumber(),
      )
    }
  })

  it('charges one wave when every user fits in a single batch', () => {
    const ttft = new Decimal(10)
    expect(perUserTimeToFirstToken(ttft, 8, 16).toNumber()).toBe(10)
    expect(perUserTimeToFirstToken(ttft, 16, 16).toNumber()).toBe(10)
  })

  it('charges a wave per batch once users exceed the batch', () => {
    const ttft = new Decimal(10)
    expect(perUserTimeToFirstToken(ttft, 32, 16).toNumber()).toBe(20)
    expect(perUserTimeToFirstToken(ttft, 33, 16).toNumber()).toBe(30)
  })

  it('does not divide by zero on a degenerate batch size', () => {
    expect(perUserTimeToFirstToken(new Decimal(10), 4, 0).toNumber()).toBe(40)
  })
})

// vLLM: "Maximum concurrency for N tokens per request" = free KV memory after
// weights, activations and overhead / one request's KV (kv_cache_utils), with
// gpu_memory_utilization 0.9 by default.
describe('maxConcurrentSessions', () => {
  it('divides the memory left after fixed costs by one session of KV', () => {
    // 80 GB x 0.9 = 72 usable; 40 GB fixed (50 total - 10 KV for 4 users); 2.5 GB/session
    expect(
      maxConcurrentSessions({
        totalPerGPUGB: 50,
        kvPerGPUGB: 10,
        concurrentUsers: 4,
        gpuVramGB: 80,
      }),
    ).toBe(12)
  })

  it('does not depend on how many users are configured', () => {
    const a = maxConcurrentSessions({
      totalPerGPUGB: 50,
      kvPerGPUGB: 10,
      concurrentUsers: 4,
      gpuVramGB: 80,
    })
    const b = maxConcurrentSessions({
      totalPerGPUGB: 60,
      kvPerGPUGB: 20,
      concurrentUsers: 8,
      gpuVramGB: 80,
    })
    expect(a).toBe(b)
  })

  it('returns 0 when the fixed costs alone overflow the usable memory', () => {
    expect(
      maxConcurrentSessions({
        totalPerGPUGB: 80,
        kvPerGPUGB: 5,
        concurrentUsers: 1,
        gpuVramGB: 80,
      }),
    ).toBe(0)
  })

  it('returns null when no KV sits on the GPU (offloaded)', () => {
    expect(
      maxConcurrentSessions({
        totalPerGPUGB: 40,
        kvPerGPUGB: 0,
        concurrentUsers: 4,
        gpuVramGB: 80,
      }),
    ).toBeNull()
  })
})

describe('maxConcurrentSessions through the engines', () => {
  const mlaMoE: Model = {
    id: 'mla-moe',
    name: 'MLA MoE',
    architecture: 'moe',
    num_parameters_billion: 671,
    active_parameters_billion: 37,
    hidden_size: 7168,
    num_hidden_layers: 61,
    num_attention_heads: 128,
    intermediate_size: 18432,
    num_experts: 256,
    num_experts_per_token: 8,
    kv_cache_elements_per_token: 35136,
    use_mla: true,
  }
  const gpu: GPU = {
    id: 'g',
    name: 'G',
    manufacturer: 'nvidia',
    vram_gb: 288,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3E',
    bus_width: 8192,
    interconnect: 'nvlink-5',
    tier: 'datacenter',
    max_gpus_per_node: 72,
  }
  const sessions = (strategy: 'tensor-parallel' | 'expert-parallel', sequenceLength: number) => {
    const single = calculateInferenceVRAM({
      model: mlaMoE,
      quantization: 'fp8',
      sequenceLength,
      batchSize: 1,
      concurrentUsers: 16,
    })
    const multi = calculateMultiGPUVRAM(single, mlaMoE, 288, 8, strategy, gpu)
    return maxConcurrentSessions({
      totalPerGPUGB: multi.totalPerGPU.toNumber(),
      kvPerGPUGB: multi.perGPU.kvCache.toNumber(),
      concurrentUsers: 16,
      gpuVramGB: 288,
    })
  }

  it('fits far more MLA sessions under expert parallelism than tensor parallelism', () => {
    expect(sessions('expert-parallel', 131072) ?? 0).toBeGreaterThan(
      4 * (sessions('tensor-parallel', 131072) ?? 0),
    )
  })

  it('fits more sessions at a shorter context', () => {
    expect(sessions('expert-parallel', 8192) ?? 0).toBeGreaterThan(
      sessions('expert-parallel', 131072) ?? 0,
    )
  })
})
