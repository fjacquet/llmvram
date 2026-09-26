import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<{ concurrentUsers: number; setConcurrentUsers: (n: number) => void }>(
    (set) => ({
      concurrentUsers: 1,
      setConcurrentUsers: (n) => set({ concurrentUsers: n }),
    }),
  )
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ConcurrentUsersInput, MAX_CONCURRENT_USERS } from './ConcurrentUsersInput'

describe('ConcurrentUsersInput', () => {
  beforeEach(() => useUIStore.setState({ concurrentUsers: 1 }))

  it('accepts rack-scale session counts beyond the old 256 cap', () => {
    render(<ConcurrentUsersInput />)
    fireEvent.change(screen.getByLabelText('Concurrent users (exact)'), {
      target: { value: '2500' },
    })
    expect(useUIStore.getState().concurrentUsers).toBe(2500)
  })

  it('clamps the exact field to 1..MAX_CONCURRENT_USERS', () => {
    render(<ConcurrentUsersInput />)
    const field = screen.getByLabelText('Concurrent users (exact)')
    fireEvent.change(field, { target: { value: '999999' } })
    expect(useUIStore.getState().concurrentUsers).toBe(MAX_CONCURRENT_USERS)
    fireEvent.change(field, { target: { value: '0' } })
    expect(useUIStore.getState().concurrentUsers).toBe(1)
  })

  it('moves the slider in powers of two', () => {
    render(<ConcurrentUsersInput />)
    fireEvent.change(screen.getByLabelText('Concurrent users'), { target: { value: '10' } })
    expect(useUIStore.getState().concurrentUsers).toBe(1024)
  })
})
