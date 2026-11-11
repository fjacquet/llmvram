import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore's `persist` middleware throws in jsdom: use a plain store.
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>(() => ({
    mode: 'inference',
    selectedGPU: null,
    setSelectedGPU: () => {},
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { GPUSelector } from './GPUSelector'

function optionNames(): string[] {
  const toggle = screen.getAllByRole('button').find((b) => b.hasAttribute('aria-haspopup'))
  if (!toggle) throw new Error('combobox button not found')
  fireEvent.click(toggle)
  return screen.getAllByRole('option').map((o) => o.textContent ?? '')
}

describe('GPUSelector rack GPUs (2026-11-11, sizing T5)', () => {
  beforeEach(() => useUIStore.setState({ mode: 'inference' }))

  it.each(['inference', 'training'])('%s does not list the NVL72 racks', (mode) => {
    useUIStore.setState({ mode })
    render(<GPUSelector />)
    const names = optionNames()
    expect(names.length).toBeGreaterThan(10)
    expect(names.some((n) => n.includes('NVL72'))).toBe(false)
  })

  it('sizing lists every GPU, racks included', () => {
    useUIStore.setState({ mode: 'sizing' })
    render(<GPUSelector />)
    expect(optionNames().filter((n) => n.includes('NVL72'))).toHaveLength(3)
  })
})
