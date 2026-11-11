import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore's `persist` middleware throws in jsdom: use a plain store.
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>((set) => ({
    mode: 'inference',
    setMode: (mode: string) => set({ mode }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ModeToggle } from './ModeToggle'

describe('ModeToggle (2026-11-11, sizing T5)', () => {
  beforeEach(() => useUIStore.setState({ mode: 'inference' }))

  it('renders Inference, Sizing and Training choices', () => {
    render(<ModeToggle />)
    expect(screen.getAllByRole('radio').map((r) => r.textContent)).toEqual([
      'Inference',
      'Sizing',
      'Training',
    ])
    expect(screen.getByRole('radio', { name: 'Inference' }).getAttribute('aria-checked')).toBe(
      'true',
    )
  })

  it.each([
    ['Sizing', 'sizing', 'Size NVL72 racks for N simultaneous users'],
    ['Training', 'training', 'Calculate VRAM for training/fine-tuning'],
  ])('choosing %s sets the mode and its description', (label, mode, description) => {
    render(<ModeToggle />)
    fireEvent.click(screen.getByRole('radio', { name: label }))
    expect(useUIStore.getState().mode).toBe(mode)
    expect(screen.getByText(description)).toBeTruthy()
  })
})
