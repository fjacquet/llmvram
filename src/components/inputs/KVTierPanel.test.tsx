import { DEFAULT_KV_TIER, type KVTierSettings } from '@engines/kv-tier'
import { fireEvent, render, screen } from '@testing-library/react'
import type { GPU } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx). setKVTier mirrors the real store: it clamps.
vi.mock('@store/uiStore', async () => {
  const { create } = await import('zustand')
  const { clampKVTier, DEFAULT_KV_TIER } = await import('@engines/kv-tier')
  const useUIStore = create<{
    kvTier: KVTierSettings
    selectedGPU: GPU | null
    setKVTier: (p: Partial<KVTierSettings>) => void
  }>((set) => ({
    kvTier: DEFAULT_KV_TIER,
    selectedGPU: null,
    setKVTier: (p) => set((s) => ({ kvTier: clampKVTier({ ...s.kvTier, ...p }) })),
  }))
  return { useUIStore }
})

import { useUIStore } from '@store/uiStore'
import { KVTierPanel } from './KVTierPanel'

function makeGPU(id: string): GPU {
  return {
    id,
    name: id,
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 3000,
    memory_type: 'HBM3',
    bus_width: 5120,
    tier: 'datacenter',
    interconnect: 'nvlink-4',
    max_gpus_per_node: 8,
  }
}

describe('KVTierPanel', () => {
  beforeEach(() => useUIStore.setState({ kvTier: DEFAULT_KV_TIER, selectedGPU: null }))

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

  it('hides the Grace host-memory option when no GPU or a non-Grace GPU is selected', () => {
    render(<KVTierPanel />)
    expect(screen.queryByRole('option', { name: /Grace/ })).not.toBeInTheDocument()

    useUIStore.setState({ selectedGPU: makeGPU('nvidia-h100-80gb-sxm') })
    render(<KVTierPanel />)
    expect(screen.queryByRole('option', { name: /Grace/ })).not.toBeInTheDocument()
  })

  it('shows the Grace host-memory option for a GB300 NVL72', () => {
    useUIStore.setState({ selectedGPU: makeGPU('nvidia-gb300-nvl72') })
    render(<KVTierPanel />)
    expect(screen.getByRole('option', { name: /Grace/ })).toBeInTheDocument()
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
