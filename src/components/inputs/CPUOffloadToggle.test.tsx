import gpusData from '@data/gpus.json'
import { fireEvent, render, screen } from '@testing-library/react'
import { validateGPUs } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the fields the component
// and useAllowedOptions read (see GPUCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>((set) => ({
    selectedGPU: null,
    selectedModel: null,
    mode: 'training',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'deepspeed-zero3',
    cpuOffloadOptimizer: false,
    setCpuOffloadOptimizer: (value: boolean) => set({ cpuOffloadOptimizer: value }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { CPUOffloadToggle } from './CPUOffloadToggle'

const gpus = validateGPUs(gpusData)
const H100 = gpus.find((g) => g.id === 'nvidia-h100-80gb-sxm') ?? null
const M3 = gpus.find((g) => g.id === 'apple-m3-ultra') ?? null

describe('CPUOffloadToggle', () => {
  beforeEach(() => {
    useUIStore.setState({
      selectedGPU: H100,
      frameworkPreset: 'deepspeed-zero3',
      cpuOffloadOptimizer: false,
    })
  })

  it('is visible for a ZeRO preset on a GPU with separate host memory', () => {
    render(<CPUOffloadToggle />)
    expect(screen.getByText('CPU Offload Optimizer')).toBeInTheDocument()
  })

  it('is hidden for a preset that does not support CPU offload (e.g. Unsloth)', () => {
    useUIStore.setState({ frameworkPreset: 'unsloth' })
    render(<CPUOffloadToggle />)
    expect(screen.queryByText('CPU Offload Optimizer')).not.toBeInTheDocument()
  })

  it('is hidden on a unified-memory GPU (R6: no separate host to offload to)', () => {
    useUIStore.setState({ selectedGPU: M3 })
    render(<CPUOffloadToggle />)
    expect(screen.queryByText('CPU Offload Optimizer')).not.toBeInTheDocument()
  })

  it('calls the setter when toggled', () => {
    render(<CPUOffloadToggle />)
    fireEvent.click(screen.getByRole('switch'))
    expect(useUIStore.getState().cpuOffloadOptimizer).toBe(true)
  })
})
