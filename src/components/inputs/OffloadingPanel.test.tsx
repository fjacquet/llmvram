import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
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

import { DEFAULT_UI_CONFIG, useUIStore } from '@store/uiStore'
import { OffloadingPanel } from './OffloadingPanel'

const gpus = validateGPUs(gpusData)
const L8 = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-8b') ?? null

describe('OffloadingPanel', () => {
  beforeEach(() =>
    useUIStore.setState({ ...DEFAULT_UI_CONFIG, selectedModel: L8, pendingNotice: null }),
  )

  it('offers only NVMe on unified memory (R6)', () => {
    useUIStore.setState({ selectedGPU: gpus.find((g) => g.id === 'apple-m3-ultra') ?? null })
    useUIStore.getState().setOffloadingEnabled(true)
    render(<OffloadingPanel />)
    expect(screen.queryByText('CPU/RAM')).not.toBeInTheDocument()
    expect(screen.getByText('NVMe SSD')).toBeInTheDocument()
  })

  it('offers CPU/RAM and NVMe on a discrete GPU', () => {
    useUIStore.setState({ selectedGPU: gpus.find((g) => g.id === 'nvidia-h100-80gb-sxm') ?? null })
    useUIStore.getState().setOffloadingEnabled(true)
    render(<OffloadingPanel />)
    expect(screen.getByText('CPU/RAM')).toBeInTheDocument()
    expect(screen.getByText('NVMe SSD')).toBeInTheDocument()
  })
})
