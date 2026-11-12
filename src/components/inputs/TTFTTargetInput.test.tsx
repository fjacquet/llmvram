import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<{
    ttftTargetSeconds: number | null
    setTTFTTarget: (s: number | null) => void
  }>((set) => ({
    ttftTargetSeconds: null,
    setTTFTTarget: (ttftTargetSeconds) => set({ ttftTargetSeconds }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { TTFTTargetInput } from './TTFTTargetInput'

describe('TTFTTargetInput', () => {
  beforeEach(() => useUIStore.setState({ ttftTargetSeconds: null }))

  it('sets a target from a preset and clears it', () => {
    render(<TTFTTargetInput />)
    fireEvent.click(screen.getByText('2 s'))
    expect(useUIStore.getState().ttftTargetSeconds).toBe(2)
    fireEvent.click(screen.getByText('Clear'))
    expect(useUIStore.getState().ttftTargetSeconds).toBeNull()
  })

  it('commits the typed value on blur, clamped to the allowed range; empty means no target', () => {
    render(<TTFTTargetInput />)
    const field = screen.getByLabelText('TTFT target (seconds)')
    fireEvent.change(field, { target: { value: '99999' } })
    fireEvent.blur(field)
    expect(useUIStore.getState().ttftTargetSeconds).toBe(3600)
    fireEvent.change(screen.getByLabelText('TTFT target (seconds)'), { target: { value: '' } })
    fireEvent.blur(screen.getByLabelText('TTFT target (seconds)'))
    expect(useUIStore.getState().ttftTargetSeconds).toBeNull()
  })
})
