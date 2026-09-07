import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the same shape instead —
// the established pattern in this repo (see useDarkMode.test.ts) for this exact problem.
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  const model128k = {
    id: 'm-128k',
    name: 'Model 128K',
    architecture: 'dense' as const,
    num_parameters_billion: 7,
    hidden_size: 4096,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    intermediate_size: 11008,
    context_length: 131072,
  }

  interface MockState {
    selectedModel: typeof model128k | null
    setSelectedModel: (model: typeof model128k | null) => void
    sequenceLength: number
    setSequenceLength: (value: number) => void
  }

  const useUIStore = create<MockState>((set) => ({
    selectedModel: model128k,
    setSelectedModel: (model) => set({ selectedModel: model }),
    sequenceLength: 4096,
    setSequenceLength: (value) => set({ sequenceLength: value }),
  }))

  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

// Import after mock setup
import { SequenceLengthInput } from './SequenceLengthInput'

const model128k = {
  id: 'm-128k',
  name: 'Model 128K',
  architecture: 'dense' as const,
  num_parameters_billion: 7,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  intermediate_size: 11008,
  context_length: 131072,
}

describe('SequenceLengthInput', () => {
  beforeEach(() => {
    useUIStore.setState({ selectedModel: model128k, sequenceLength: 4096 })
  })

  it('formats 1M tokens as M, not as 1024K', () => {
    useUIStore.setState({ sequenceLength: 1048576 })
    render(<SequenceLengthInput />)
    expect(screen.getByText(/1M tokens/)).toBeInTheDocument()
    expect(screen.queryByText(/1024K/)).not.toBeInTheDocument()
  })

  it('offers a 1M preset', () => {
    render(<SequenceLengthInput />)
    expect(screen.getByRole('button', { name: '1M' })).toBeInTheDocument()
  })

  it('warns when the sequence exceeds the model native context', () => {
    useUIStore.setState({ sequenceLength: 262144 })
    render(<SequenceLengthInput />)
    expect(screen.getByText(/beyond native context/i)).toBeInTheDocument()
  })

  it('does not warn at or below the model native context', () => {
    useUIStore.setState({ sequenceLength: 131072 })
    render(<SequenceLengthInput />)
    expect(screen.queryByText(/beyond native context/i)).not.toBeInTheDocument()
  })

  it('extends its range for a model whose native context exceeds 1M', () => {
    useUIStore.setState({
      selectedModel: { ...model128k, id: 'm-10m', context_length: 10485760 },
    })
    render(<SequenceLengthInput />)
    const slider = screen.getByRole('slider', { name: /sequence length/i })
    expect(Number(slider.getAttribute('max'))).toBeCloseTo(Math.log2(10485760), 3)
  })
})
