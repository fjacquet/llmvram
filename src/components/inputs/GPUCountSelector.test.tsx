import { fireEvent, render, screen } from '@testing-library/react'
import type { GPU } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

function gpuWithBound(name: string, max: number): GPU {
  return {
    id: 'test-gpu',
    name,
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 2000,
    memory_type: 'HBM3',
    bus_width: 5120,
    max_gpus_per_node: max,
    tier: 'datacenter',
  }
}

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the same shape instead —
// the established pattern in this repo (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  interface MockState {
    numGPUs: number
    selectedGPU: unknown
    mode: string
    shardingStrategy: string
    setNumGPUs: (value: number) => void
  }

  const useUIStore = create<MockState>((set) => ({
    numGPUs: 1,
    selectedGPU: null,
    mode: 'inference',
    shardingStrategy: 'tensor-parallel',
    setNumGPUs: (value) => set({ numGPUs: value }),
  }))

  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

// Import after mock setup
import { GPUCountSelector } from './GPUCountSelector'

describe('GPUCountSelector', () => {
  beforeEach(() => {
    useUIStore.setState({
      numGPUs: 1,
      selectedGPU: gpuWithBound('Test 8-way', 8),
      mode: 'inference',
      shardingStrategy: 'tensor-parallel',
    })
  })

  it('caps the slider at the selected GPU max_gpus_per_node', () => {
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '8')
  })

  it('raises the cap to 72 for an NVL72-class part', () => {
    useUIStore.setState({ selectedGPU: gpuWithBound('Test NVL72', 72) })
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '72')
  })

  it('renders no slider for a single-GPU part', () => {
    useUIStore.setState({ selectedGPU: gpuWithBound('Test single', 1) })
    render(<GPUCountSelector />)
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    expect(screen.getByText(/Single GPU/)).toBeInTheDocument()
  })

  it('falls back to a cap of 8 when no GPU is selected', () => {
    useUIStore.setState({ selectedGPU: null })
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '8')
  })

  it('drops the false claim that 8 is the largest GPU domain in current hardware', () => {
    render(<GPUCountSelector />)
    // InfoTip only renders its `text` prop into the DOM once its trigger is opened.
    fireEvent.click(screen.getByRole('button', { name: /more info/i }))
    expect(screen.getByText(/Capped at 8/)).toBeInTheDocument()
    expect(screen.queryByText(/fully connected GPU domain/)).not.toBeInTheDocument()
  })
})
