import gpusData from '@data/gpus.json'
import { INTERCONNECT_LABELS } from '@engines/constants'
import { fireEvent, render, screen } from '@testing-library/react'
import { type GPU, validateGPU, validateGPUs } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the fields the component
// and useAllowedOptions read (see GPUCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>((set) => ({
    selectedGPU: null,
    selectedModel: null,
    mode: 'inference',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'none',
    interconnectOverride: null,
    setInterconnectOverride: (value: string | null) => set({ interconnectOverride: value }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { InterconnectSelector } from './InterconnectSelector'

const gpus = validateGPUs(gpusData)
const H100 = gpus.find((g) => g.id === 'nvidia-h100-80gb-sxm') as GPU
// Derived row, validated by the real schema: after Task 3a no database GPU carries
// interconnect_options, so a custom-GPU-shaped row is the only way to reach a
// selector with two or more offered variants (mirrors config-rules.test.ts's WITH_OPTIONS).
const WITH_OPTIONS = validateGPU({ ...H100, interconnect_options: ['nvlink-4', 'pcie-5'] })

describe('InterconnectSelector', () => {
  beforeEach(() => {
    useUIStore.setState({
      selectedGPU: H100,
      interconnectOverride: null,
    })
  })

  it('renders nothing when allowedOptions offers no interconnect variants', () => {
    const { container } = render(<InterconnectSelector />)
    expect(container).toBeEmptyDOMElement()
  })

  it('shows the options from a custom GPU with interconnect_options', () => {
    useUIStore.setState({ selectedGPU: WITH_OPTIONS })
    render(<InterconnectSelector />)
    expect(screen.getByText('Interconnect Type')).toBeInTheDocument()
    for (const opt of WITH_OPTIONS.interconnect_options ?? []) {
      const label = INTERCONNECT_LABELS[opt] ?? opt
      expect(screen.getByRole('radio', { name: label })).toBeInTheDocument()
    }
  })

  it('calls the setter when a variant is selected', () => {
    useUIStore.setState({ selectedGPU: WITH_OPTIONS })
    render(<InterconnectSelector />)
    fireEvent.click(screen.getByRole('radio', { name: INTERCONNECT_LABELS['pcie-5'] }))
    expect(useUIStore.getState().interconnectOverride).toBe('pcie-5')
  })
})
