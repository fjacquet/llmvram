import { fireEvent, render, screen } from '@testing-library/react'
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
    const max = Number(slider.getAttribute('max'))

    // The ceiling is rounded up to the step grid so the advertised maximum is actually
    // reachable: a range input only exposes min + n*step, so the exact log2(10,485,760)
    // of 23.3219 would top out at 23.3 — i.e. 10,301,796 tokens, never the full 10M.
    expect(max).toBeGreaterThanOrEqual(Math.log2(10485760))
    const step = Number(slider.getAttribute('step'))
    const min = Number(slider.getAttribute('min'))
    expect(Number.isInteger(Math.round((max - min) / step))).toBe(true)
    expect((max - min) / step - Math.round((max - min) / step)).toBeCloseTo(0, 6)
  })

  it('clamps a slider drag at the top of the track to the real maximum', () => {
    useUIStore.setState({
      selectedModel: { ...model128k, id: 'm-10m', context_length: 10485760 },
    })
    render(<SequenceLengthInput />)
    const slider = screen.getByRole('slider', { name: /sequence length/i })

    fireEvent.change(slider, { target: { value: slider.getAttribute('max') } })

    // 2^23.4 overshoots; the value lands exactly on the advertised maximum, not past it.
    expect(useUIStore.getState().sequenceLength).toBe(10485760)
  })

  it('says so when the stored value is above the slider range', () => {
    useUIStore.setState({
      selectedModel: { ...model128k, id: 'm-10m', context_length: 10485760 },
      sequenceLength: 4194304,
    })
    const { rerender } = render(<SequenceLengthInput />)

    // Switching to a model with a smaller context leaves the value outstanding: the thumb
    // pins to the right while the readout keeps 4M, so the mismatch must be stated.
    useUIStore.setState({ selectedModel: model128k })
    rerender(<SequenceLengthInput />)

    expect(screen.getByText(/above this slider's range/i)).toBeInTheDocument()
    expect(useUIStore.getState().sequenceLength).toBe(4194304)
  })
})
