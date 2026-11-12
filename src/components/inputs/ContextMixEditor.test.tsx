import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  type Mix = { tokens: number; share: number }[]
  const useUIStore = create<{
    contextMix: Mix
    sequenceLength: number
    setContextMix: (m: Mix) => void
  }>((set) => ({
    contextMix: [{ tokens: 4096, share: 1 }],
    sequenceLength: 4096,
    setContextMix: (contextMix) =>
      set({ contextMix, sequenceLength: Math.max(...contextMix.map((c) => c.tokens)) }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ContextMixEditor } from './ContextMixEditor'

const reset = () =>
  useUIStore.setState({ contextMix: [{ tokens: 4096, share: 1 }], sequenceLength: 4096 })

describe('ContextMixEditor', () => {
  beforeEach(reset)

  it('shows no memory label for the default mix', () => {
    render(<ContextMixEditor />)
    expect(screen.queryByText('memory sized on the mix average')).not.toBeInTheDocument()
  })

  it('adds a class that takes half the old share, and labels memory once the mix is not default', () => {
    render(<ContextMixEditor />)
    fireEvent.click(screen.getByText('Add class'))
    const mix = useUIStore.getState().contextMix
    expect(mix).toHaveLength(2)
    expect(mix.reduce((s, c) => s + c.share, 0)).toBeCloseTo(1)
    expect(screen.getByText('memory sized on the mix average')).toBeInTheDocument()
  })

  it('keeps the last share equal to the remainder and read-only', () => {
    useUIStore.setState({
      contextMix: [
        { tokens: 4096, share: 0.5 },
        { tokens: 262144, share: 0.5 },
      ],
      sequenceLength: 262144,
    })
    render(<ContextMixEditor />)
    const share = screen.getByLabelText('Class 1 share')
    fireEvent.change(share, { target: { value: '90' } })
    fireEvent.blur(share)
    const [a, b] = useUIStore.getState().contextMix
    expect(a?.share).toBeCloseTo(0.9)
    expect(b?.share).toBeCloseTo(0.1)
    expect(screen.getByLabelText('Class 2 share')).toHaveValue(10)
    expect(screen.getByLabelText('Class 2 share')).toHaveAttribute('readonly')
  })

  it('lets the share be cleared and retyped, committing on blur', () => {
    useUIStore.setState({
      contextMix: [
        { tokens: 4096, share: 0.5 },
        { tokens: 262144, share: 0.5 },
      ],
      sequenceLength: 262144,
    })
    render(<ContextMixEditor />)
    const field = screen.getByLabelText('Class 1 share')
    fireEvent.change(field, { target: { value: '' } })
    expect(useUIStore.getState().contextMix[0]?.share).toBe(0.5)
    fireEvent.change(field, { target: { value: '25' } })
    fireEvent.blur(field)
    const [a, b] = useUIStore.getState().contextMix
    expect(a?.share).toBeCloseTo(0.25)
    expect(b?.share).toBeCloseTo(0.75)
  })

  it('cannot add a fourth class', () => {
    useUIStore.setState({
      contextMix: [
        { tokens: 4096, share: 0.5 },
        { tokens: 32768, share: 0.3 },
        { tokens: 262144, share: 0.2 },
      ],
      sequenceLength: 262144,
    })
    render(<ContextMixEditor />)
    expect(screen.getByText('Add class')).toBeDisabled()
  })

  it('removes a class down to one, never below', () => {
    useUIStore.setState({
      contextMix: [
        { tokens: 4096, share: 0.9 },
        { tokens: 262144, share: 0.1 },
      ],
      sequenceLength: 262144,
    })
    render(<ContextMixEditor />)
    fireEvent.click(screen.getByLabelText('Remove class 2'))
    expect(useUIStore.getState().contextMix).toEqual([{ tokens: 4096, share: 1 }])
    expect(screen.queryByLabelText('Remove class 1')).not.toBeInTheDocument()
  })

  it('sets a class from a preset chip', () => {
    render(<ContextMixEditor />)
    fireEvent.click(screen.getByText('RAG'))
    expect(useUIStore.getState().contextMix[0]?.tokens).toBe(32768)
  })
})
