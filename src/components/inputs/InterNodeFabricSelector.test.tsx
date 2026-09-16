import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the same shape instead —
// the established pattern in this repo (see SequenceLengthInput.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  interface MockCustomFabric {
    name: string
    port_gbps: number
  }

  interface MockState {
    numGPUs: number
    numNodes: number
    interNodeFabric: string
    setInterNodeFabric: (value: string) => void
    customFabric: MockCustomFabric | null
    setCustomFabric: (value: MockCustomFabric | null) => void
  }

  const useUIStore = create<MockState>((set) => ({
    numGPUs: 1,
    numNodes: 1,
    interNodeFabric: 'ethernet-800g',
    setInterNodeFabric: (value) => set({ interNodeFabric: value }),
    customFabric: null,
    setCustomFabric: (value) => set({ customFabric: value }),
  }))

  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

// Import after mock setup
import { InterNodeFabricSelector } from './InterNodeFabricSelector'

describe('InterNodeFabricSelector', () => {
  beforeEach(() => {
    useUIStore.setState({
      numGPUs: 1,
      numNodes: 1,
      interNodeFabric: 'ethernet-800g',
      customFabric: null,
    })
  })

  it('renders nothing when there is only one server', () => {
    const { container } = render(<InterNodeFabricSelector />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows per-server aggregate bandwidth, not port speed, once servers > 1', () => {
    useUIStore.setState({ numGPUs: 8, numNodes: 4 })
    render(<InterNodeFabricSelector />)
    // 800GbE is a 100 GB/s port; with 8 GPUs (8 NICs) per server that is 800 GB/s per
    // server, not 100 — the whole point of perNodeFabricGBps.
    expect(
      screen.getByText(
        (_, node) => node?.textContent === '800GbE (SONiC / RoCEv2) — 800 GB/s per server',
      ),
    ).toBeInTheDocument()
  })

  it('reveals the custom bandwidth field and computes per-server total across NICs', () => {
    useUIStore.setState({ numGPUs: 8, numNodes: 2, interNodeFabric: 'custom' })
    render(<InterNodeFabricSelector />)

    const input = screen.getByLabelText(/Port bandwidth/i)
    fireEvent.change(input, { target: { value: '25' } })

    expect(useUIStore.getState().customFabric).toEqual({ name: 'Custom fabric', port_gbps: 25 })
    expect(screen.getByText(/200 GB\/s per server across/)).toBeInTheDocument()
    expect(screen.getByText(/8 NICs/)).toBeInTheDocument()
  })

  it('clears custom fabric when the field is emptied', () => {
    useUIStore.setState({
      numGPUs: 8,
      numNodes: 2,
      interNodeFabric: 'custom',
      customFabric: { name: 'Custom fabric', port_gbps: 25 },
    })
    render(<InterNodeFabricSelector />)

    const input = screen.getByLabelText(/Port bandwidth/i)
    fireEvent.change(input, { target: { value: '' } })

    expect(useUIStore.getState().customFabric).toBeNull()
  })
})
