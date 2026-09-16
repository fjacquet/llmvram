import type { InferenceVRAMBreakdown } from '@engines/types'
import { render, screen } from '@testing-library/react'
import type { GPU } from '@utils/schemas'
import Decimal from 'decimal.js'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the same shape instead —
// the established pattern in this repo (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  interface MockState {
    offloadingEnabled: boolean
  }

  const useUIStore = create<MockState>(() => ({
    offloadingEnabled: false,
  }))

  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

// Import after mock setup
import { Recommendations } from './Recommendations'

function gpu(overrides: Partial<GPU>): GPU {
  return {
    id: 'test-gpu',
    name: 'Test GPU',
    manufacturer: 'apple',
    vram_gb: 128,
    memory_bandwidth_gbps: 800,
    memory_type: 'Unified',
    bus_width: 0,
    max_gpus_per_node: 1,
    tier: 'apple-silicon',
    ...overrides,
  }
}

function breakdown(totalGB: number): InferenceVRAMBreakdown {
  return {
    modelWeights: new Decimal(totalGB * 0.9),
    kvCache: new Decimal(totalGB * 0.05),
    activations: new Decimal(totalGB * 0.03),
    frameworkOverhead: new Decimal(totalGB * 0.02),
    total: new Decimal(totalGB),
  }
}

describe('Recommendations', () => {
  beforeEach(() => {
    useUIStore.setState({ offloadingEnabled: false })
  })

  it('suppresses the multi-GPU recommendation for a part with max_gpus_per_node=1', () => {
    const appleGPU = gpu({ max_gpus_per_node: 1, vram_gb: 128 })
    render(
      <Recommendations
        gpu={appleGPU}
        breakdown={breakdown(900)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={1}
        multiGPUBreakdown={null}
      />,
    )

    expect(screen.queryByText(/with tensor parallelism/)).not.toBeInTheDocument()
  })

  it('clamps the suggested GPU count to max_gpus_per_node instead of an unbuildable count', () => {
    const gb10Like = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 2,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={gb10Like}
        breakdown={breakdown(100)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={1}
        multiGPUBreakdown={null}
      />,
    )

    // Unclamped math (100 / (16 * 0.85)) would ask for 8x — unbuildable on a 2-way part.
    expect(screen.queryByText(/8x Test GPU/)).not.toBeInTheDocument()
    expect(screen.getByText(/2x Test GPU/)).toBeInTheDocument()
  })

  it('clamps the "add more GPUs" recommendation when already multi-GPU', () => {
    const eightWay = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={eightWay}
        breakdown={breakdown(500)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={4}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(200),
            numGPUs: 4,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    // Unclamped math (500 / (16 * 0.85)) would ask for 37x — clamp to the 8-way bound.
    expect(screen.queryByText(/37x Test GPU/)).not.toBeInTheDocument()
    expect(screen.getByText(/8x Test GPU/)).toBeInTheDocument()
  })

  it('never advises fewer GPUs than are already configured across a cluster', () => {
    // 8 GPUs/node x 4 nodes = 32 total. numGPUs is the CLUSTER TOTAL (see
    // ResultsPanel), but the suggested count was clamped to the PER-NODE bound
    // of 8, so the panel advised cutting a 32-GPU cluster down to 8.
    const eightWay = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={eightWay}
        breakdown={breakdown(500)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={32}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(200),
            numGPUs: 32,
            numNodes: 4,
            gpusPerNode: 8,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    expect(screen.queryByText(/Try 8x Test GPU/)).not.toBeInTheDocument()
  })

  it('suppresses the no-op suggestion when a single node is already at its bound', () => {
    // 8 GPUs on an 8-way part: the clamp can only land back on 8, which used to
    // render as "Current 8x ... Try 8x" — advice to change nothing.
    const eightWay = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={eightWay}
        breakdown={breakdown(500)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={8}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(200),
            numGPUs: 8,
            numNodes: 1,
            gpusPerNode: 8,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    expect(screen.queryByText(/with tensor parallelism/)).not.toBeInTheDocument()
  })
})
