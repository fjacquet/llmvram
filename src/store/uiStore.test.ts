import gpusData from '@data/gpus.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import type { GPU } from '@utils/schemas'
import { validateGPUs } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The real uiStore wraps its state in zustand's `persist` middleware, which
 * needs localStorage — unavailable in this jsdom environment (see
 * KVTierPanel.test.tsx / NodeCountSelector.test.tsx for the same constraint).
 * Mocking `persist` as a pass-through (the real state creator, no storage)
 * lets this file exercise the ACTUAL setSelectedGPU/setKVTier from
 * uiStore.ts, not a hand-written reimplementation — a reimplementation can
 * silently drift from the real logic and pass even when the real store has
 * the bug (as happened here: an earlier version of this file mirrored the
 * store and never caught that setKVTier itself lacked the Grace guard).
 */
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return {
    ...actual,
    persist: (config: unknown) => config,
  }
})

// jsdom has no matchMedia; uiStore reads it once at module init for isDarkMode.
beforeEach(() => {
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

const gpus = validateGPUs(gpusData)

/** A real GPU row from the database, looked up by id (never hand-written). */
function findGPU(id: string): GPU {
  const gpu = gpus.find((g) => g.id === id)
  if (!gpu) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return gpu
}

describe('uiStore: host-grace KV tier guard', () => {
  beforeEach(async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.setState({ selectedGPU: null, kvTier: DEFAULT_KV_TIER })
  })

  it('URL restore order (GPU first, then tier): falls back to none for a non-Grace GPU', async () => {
    const { useUIStore } = await import('@store/uiStore')
    // useURLSync restores the GPU, then the tier, from the same hash.
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-h100-80gb-sxm'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().kvTier.tier).toBe('none')
  })

  it('URL restore order: falls back to none for the x86 HGX B300 too', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-gb300-288gb'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().kvTier.tier).toBe('none')
  })

  it('URL restore order: keeps host-grace for a Grace-host GPU', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-gb300-nvl72'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().kvTier.tier).toBe('host-grace')
  })

  it('GPU-change path (through the store, not just the helper): falls back to none', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-gb300-nvl72'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().kvTier.tier).toBe('host-grace')

    useUIStore.getState().setSelectedGPU(findGPU('nvidia-h100-80gb-sxm'))
    expect(useUIStore.getState().kvTier.tier).toBe('none')
  })
})
