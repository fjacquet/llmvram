import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the same shape instead —
// the established pattern in this repo (see SequenceLengthInput.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  interface MockState {
    numGPUs: number
    numNodes: number
    setNumNodes: (value: number) => void
  }

  const useUIStore = create<MockState>((set) => ({
    numGPUs: 1,
    numNodes: 1,
    setNumNodes: (value) => set({ numNodes: value }),
  }))

  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

// Import after mock setup
import { NodeCountSelector } from './NodeCountSelector'

describe('NodeCountSelector', () => {
  beforeEach(() => {
    useUIStore.setState({ numGPUs: 1, numNodes: 1 })
  })

  it('shows the per-node label and no pipeline-parallel note at 1 server', () => {
    render(<NodeCountSelector />)
    expect(screen.getByText('Number of servers')).toBeInTheDocument()
    expect(screen.getByText('1 GPUs total')).toBeInTheDocument()
    expect(screen.queryByText(/pipeline parallel across servers/)).not.toBeInTheDocument()
  })

  it('multiplies GPUs per server by server count for the total, and notes pipeline parallelism', () => {
    useUIStore.setState({ numGPUs: 8, numNodes: 4 })
    render(<NodeCountSelector />)
    expect(screen.getByText(/32 GPUs total/)).toBeInTheDocument()
    expect(screen.getByText(/pipeline parallel across servers/)).toBeInTheDocument()
  })

  it('updates the store when the slider changes', () => {
    render(<NodeCountSelector />)
    const slider = screen.getByRole('slider')
    fireEvent.change(slider, { target: { value: '3' } })
    expect(useUIStore.getState().numNodes).toBe(3)
  })
})
