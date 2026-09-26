import { DEFAULT_KV_TIER, type KVTierSettings } from '@engines/kv-tier'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<{
    kvTier: KVTierSettings
    setKVTier: (p: Partial<KVTierSettings>) => void
  }>((set) => ({
    kvTier: {
      tier: 'none',
      customGBps: null,
      activeShare: 0.25,
      burstSeconds: 30,
      capacityTB: null,
    },
    setKVTier: (p) => set((s) => ({ kvTier: { ...s.kvTier, ...p } })),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { KVTierPanel } from './KVTierPanel'

describe('KVTierPanel', () => {
  beforeEach(() => useUIStore.setState({ kvTier: DEFAULT_KV_TIER }))

  it('hides the tier settings when no tier is selected', () => {
    render(<KVTierPanel />)
    expect(screen.queryByLabelText('Active share (%)')).not.toBeInTheDocument()
  })

  it('selects network storage and shows its settings', () => {
    render(<KVTierPanel />)
    fireEvent.change(screen.getByLabelText('KV storage tier'), { target: { value: 'network' } })
    expect(useUIStore.getState().kvTier.tier).toBe('network')
    expect(screen.getByLabelText('Active share (%)')).toBeInTheDocument()
  })

  it('does not offer a Dell Lightning preset (cluster-scale storage, sized in raidy)', () => {
    render(<KVTierPanel />)
    expect(screen.queryByRole('option', { name: /Lightning/ })).not.toBeInTheDocument()
  })

  it('stores the active share as a fraction, clamped to 1-100%', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } })
    render(<KVTierPanel />)
    const share = screen.getByLabelText('Active share (%)')
    fireEvent.change(share, { target: { value: '10' } })
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.1)
    fireEvent.change(share, { target: { value: '0' } })
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.01)
  })

  it('treats a cleared capacity as unlimited', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network', capacityTB: 100 } })
    render(<KVTierPanel />)
    fireEvent.change(screen.getByLabelText('Tier capacity (TB)'), { target: { value: '' } })
    expect(useUIStore.getState().kvTier.capacityTB).toBeNull()
  })
})
