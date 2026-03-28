import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// vi.hoisted runs before vi.mock hoisting, so mockStore is available in the factory
const { mockStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  interface MockState {
    isDarkMode: boolean
    toggleDarkMode: () => void
    setIsDarkMode: (dark: boolean) => void
  }

  const mockStore = create<MockState>((set) => ({
    isDarkMode: false,
    toggleDarkMode: () => set((s) => ({ isDarkMode: !s.isDarkMode })),
    setIsDarkMode: (dark: boolean) => set({ isDarkMode: dark }),
  }))

  return { mockStore }
})

vi.mock('@store/uiStore', () => ({
  useUIStore: mockStore,
}))

// Import after mock setup
import { useDarkMode } from './useDarkMode'

describe('useDarkMode', () => {
  beforeEach(() => {
    mockStore.setState({ isDarkMode: false })
    document.documentElement.classList.remove('dark')
    // jsdom does not implement matchMedia — provide a minimal stub
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn((query: string) => ({
        matches: false,
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    })
  })

  it('should return isDarkMode and toggleDarkMode', () => {
    const { result } = renderHook(() => useDarkMode())
    expect(result.current).toHaveProperty('isDarkMode')
    expect(result.current).toHaveProperty('toggleDarkMode')
    expect(typeof result.current.toggleDarkMode).toBe('function')
  })

  it('should not have dark class when isDarkMode is false', () => {
    renderHook(() => useDarkMode())
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('should add dark class when isDarkMode is true', () => {
    mockStore.setState({ isDarkMode: true })
    renderHook(() => useDarkMode())
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('should toggle dark mode and update DOM class', () => {
    const { result } = renderHook(() => useDarkMode())
    expect(result.current.isDarkMode).toBe(false)

    act(() => {
      result.current.toggleDarkMode()
    })
    expect(result.current.isDarkMode).toBe(true)
    expect(document.documentElement.classList.contains('dark')).toBe(true)

    act(() => {
      result.current.toggleDarkMode()
    })
    expect(result.current.isDarkMode).toBe(false)
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })
})
