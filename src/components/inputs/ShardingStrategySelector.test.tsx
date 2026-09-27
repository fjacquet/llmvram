import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create(() => ({
    numGPUs: 8,
    numNodes: 1,
    mode: 'inference',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'none',
    interconnectOverride: null as string | null,
    setShardingStrategy: () => {},
    selectedGPU: null as unknown,
    selectedModel: null as unknown,
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ShardingStrategySelector } from './ShardingStrategySelector'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => models.find((m) => m.id === id) ?? null
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null

describe('ShardingStrategySelector', () => {
  beforeEach(() =>
    useUIStore.setState({
      selectedModel: null,
      selectedGPU: null,
      numGPUs: 8,
      interconnectOverride: null,
    }),
  )

  it('offers expert parallelism for a MoE model', () => {
    useUIStore.setState({ selectedModel: model('deepseek-r1') })
    render(<ShardingStrategySelector />)
    expect(screen.getByText(/Expert Parallel \+ DP attention/)).toBeInTheDocument()
  })

  it('hides expert parallelism for a dense model', () => {
    useUIStore.setState({ selectedModel: model('meta-llama-llama-3.1-70b') })
    render(<ShardingStrategySelector />)
    expect(screen.queryByText(/Expert Parallel/)).not.toBeInTheDocument()
  })

  it('shows the NVLink bridge at TP-2 and PCIe 5 at TP-4 on H100 PCIe (badge never contradicts the maths)', () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-h100-80gb-pcie'), numGPUs: 2 })
    const { unmount } = render(<ShardingStrategySelector />)
    expect(screen.getByText(/NVLink bridge: 600 GB\/s/)).toBeInTheDocument()
    unmount()
    useUIStore.setState({ numGPUs: 4 })
    render(<ShardingStrategySelector />)
    expect(screen.getByText(/PCIe 5: 128 GB\/s/)).toBeInTheDocument()
    expect(screen.queryByText(/NVLink bridge/)).not.toBeInTheDocument()
  })

  it('shows the override link, not the bridge, when interconnectOverride is set on a bridged GPU', () => {
    // At TP-2 this part's own bridge would otherwise carry the group (see the test
    // above) — an active override must win, matching what useInferenceCalculation
    // computes the actual numbers from (ADR: badge never contradicts the maths).
    useUIStore.setState({
      selectedGPU: gpu('nvidia-h100-80gb-pcie'),
      numGPUs: 2,
      interconnectOverride: 'pcie-5',
    })
    render(<ShardingStrategySelector />)
    expect(screen.getByText(/PCIe 5: 128 GB\/s/)).toBeInTheDocument()
    expect(screen.queryByText(/NVLink bridge/)).not.toBeInTheDocument()
  })

  it('stays visible at 1 GPU on a multi-GPU part, without an interconnect badge', () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-h100-80gb-sxm'), numGPUs: 1 })
    render(<ShardingStrategySelector />)
    expect(screen.getByText('Intra-server sharding strategy')).toBeInTheDocument()
    expect(screen.queryByText(/GB\/s ·/)).not.toBeInTheDocument()
  })
})
