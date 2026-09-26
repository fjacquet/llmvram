import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create(() => ({
    numGPUs: 8,
    shardingStrategy: 'tensor-parallel',
    setShardingStrategy: () => {},
    selectedGPU: null,
    selectedModel: null as { architecture: string } | null,
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ShardingStrategySelector } from './ShardingStrategySelector'

describe('ShardingStrategySelector', () => {
  beforeEach(() => useUIStore.setState({ selectedModel: null }))

  it('offers expert parallelism for a MoE model', () => {
    useUIStore.setState({ selectedModel: { architecture: 'moe' } })
    render(<ShardingStrategySelector />)
    expect(screen.getByText(/Expert Parallel \+ DP attention/)).toBeInTheDocument()
  })

  it('hides expert parallelism for a dense model', () => {
    useUIStore.setState({ selectedModel: { architecture: 'dense' } })
    render(<ShardingStrategySelector />)
    expect(screen.queryByText(/Expert Parallel/)).not.toBeInTheDocument()
  })
})
