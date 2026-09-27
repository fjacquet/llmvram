import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { act, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { Header } from './Header'

vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})

describe('Header: Reset all (spec Section 4)', () => {
  beforeEach(() => {
    useUIStore.setState({ ...DEFAULT_UI_CONFIG, pendingNotice: null })
  })

  it('has a labelled reset-all button that returns the store to its initial state, with one notice', () => {
    useUIStore.setState({ batchSize: 8, numGPUs: 4 })
    render(<Header />)
    act(() => screen.getByRole('button', { name: 'Reset all settings to defaults' }).click())
    const state = useUIStore.getState()
    expect(state.batchSize).toBe(DEFAULT_UI_CONFIG.batchSize)
    expect(state.numGPUs).toBe(DEFAULT_UI_CONFIG.numGPUs)
    expect(state.pendingNotice?.title).toBe('Reset to defaults')
  })
})
