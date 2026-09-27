import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { act, renderHook } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

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
const { warning } = vi.hoisted(() => ({ warning: vi.fn() }))
vi.mock('sonner', () => ({ toast: { warning } }))

import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { useConfigNotices } from './useConfigNotices'

const L70 = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b') ?? null
const H100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm') ?? null

describe('useConfigNotices', () => {
  beforeEach(() => {
    warning.mockClear()
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: L70,
      selectedGPU: H100,
      pendingNotice: null,
    })
  })

  it('toasts one grouped notice per action and clears it', () => {
    renderHook(() => useConfigNotices())
    act(() => useUIStore.getState().setNumGPUs(6))
    expect(warning).toHaveBeenCalledTimes(1)
    expect(warning.mock.calls[0]?.[0]).toBe(`Adjusted for ${L70?.name}`)
    expect(useUIStore.getState().pendingNotice).toBeNull()
  })

  it('toasts an identical repeated correction again', () => {
    renderHook(() => useConfigNotices())
    act(() => useUIStore.getState().setNumGPUs(6))
    act(() => useUIStore.getState().setNumGPUs(6))
    expect(warning).toHaveBeenCalledTimes(2)
  })

  it('says nothing when nothing was corrected', () => {
    renderHook(() => useConfigNotices())
    act(() => useUIStore.getState().setNumGPUs(4))
    expect(warning).not.toHaveBeenCalled()
  })
})
