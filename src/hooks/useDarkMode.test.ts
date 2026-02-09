import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useDarkMode } from './useDarkMode'

// Mock the uiStore to avoid persist middleware localStorage issues in jsdom
vi.mock('@store/uiStore', () => {
  const { create } = require('zustand')
  const store = create(() => ({
    isDarkMode: false,
    toggleDarkMode: () =>
      store.setState((s: { isDarkMode: boolean }) => ({
        isDarkMode: !s.isDarkMode,
      })),
  }))
  return { useUIStore: store }
})

// Get the mocked store for test manipulation
import { useUIStore } from '@store/uiStore'

describe('useDarkMode', () => {
  beforeEach(() => {
    useUIStore.setState({ isDarkMode: false })
    document.documentElement.classList.remove('dark')
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
    useUIStore.setState({ isDarkMode: true })
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
