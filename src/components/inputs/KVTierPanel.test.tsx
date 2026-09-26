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
    fireEvent.blur(share)
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.1)
    fireEvent.change(share, { target: { value: '0' } })
    fireEvent.blur(share)
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.01)
  })

  it('treats a cleared capacity as unlimited', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network', capacityTB: 100 } })
    render(<KVTierPanel />)
    const capacity = screen.getByLabelText('Tier capacity (TB)')
    fireEvent.change(capacity, { target: { value: '' } })
    fireEvent.blur(capacity)
    expect(useUIStore.getState().kvTier.capacityTB).toBeNull()
  })

  it('lets a user clear a field and type a new value without it snapping', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } })
    render(<KVTierPanel />)
    const share = screen.getByLabelText('Active share (%)') as HTMLInputElement
    fireEvent.change(share, { target: { value: '' } })
    expect(share.value).toBe('')
    fireEvent.change(share, { target: { value: '5' } })
    fireEvent.change(share, { target: { value: '50' } })
    fireEvent.blur(share)
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.5)
  })

  it('accepts a fractional capacity typed digit by digit', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } })
    render(<KVTierPanel />)
    const capacity = screen.getByLabelText('Tier capacity (TB)') as HTMLInputElement
    fireEvent.change(capacity, { target: { value: '0' } })
    fireEvent.change(capacity, { target: { value: '0.' } })
    fireEvent.change(capacity, { target: { value: '0.5' } })
    expect(capacity.value).toBe('0.5')
    fireEvent.blur(capacity)
    expect(useUIStore.getState().kvTier.capacityTB).toBe(0.5)
  })
})
