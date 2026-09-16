import type { GPU } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { clampGPUCount } from './gpuLimits'

function gpuWithBound(max: number): GPU {
  return {
    id: 'test-gpu',
    name: 'Test GPU',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 2000,
    memory_type: 'HBM3',
    bus_width: 5120,
    max_gpus_per_node: max,
    tier: 'datacenter',
  }
}

describe('clampGPUCount', () => {
  it('leaves a count inside the bound untouched', () => {
    expect(clampGPUCount(4, gpuWithBound(8))).toBe(4)
  })

  it('clamps a count above the bound down to it', () => {
    expect(clampGPUCount(8, gpuWithBound(4))).toBe(4)
  })

  it('clamps to 1 for a single-GPU part', () => {
    expect(clampGPUCount(8, gpuWithBound(1))).toBe(1)
  })

  it('allows 72 for an NVL72-class part', () => {
    expect(clampGPUCount(72, gpuWithBound(72))).toBe(72)
  })

  it('floors at 1 for zero and negative input', () => {
    expect(clampGPUCount(0, gpuWithBound(8))).toBe(1)
    expect(clampGPUCount(-3, gpuWithBound(8))).toBe(1)
  })

  it('truncates a fractional count', () => {
    expect(clampGPUCount(3.7, gpuWithBound(8))).toBe(3)
  })

  it('falls back to the engine sanity bound when no GPU is selected', () => {
    expect(clampGPUCount(72, null)).toBe(72)
    expect(clampGPUCount(999, null)).toBe(72)
  })
})
