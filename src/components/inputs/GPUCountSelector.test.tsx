import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { MAX_GPUS_PER_NODE } from '@engines/constants'
import { fireEvent, render, screen } from '@testing-library/react'
import { maxGPUsFor } from '@utils/gpuLimits'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the fields the component
// and useAllowedOptions read.
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>((set) => ({
    numGPUs: 1,
    numNodes: 1,
    selectedGPU: null,
    selectedModel: null,
    mode: 'inference',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    kvCacheOffload: false,
    frameworkPreset: 'none',
    setNumGPUs: (value: number) => set({ numGPUs: value }),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { GPUCountSelector } from './GPUCountSelector'

const gpus = validateGPUs(gpusData)
const models = validateModels(modelsData)
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null
const model = (id: string) => models.find((m) => m.id === id) ?? null

describe('GPUCountSelector', () => {
  beforeEach(() => {
    useUIStore.setState({
      numGPUs: 1,
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      selectedModel: null,
      mode: 'inference',
      shardingStrategy: 'tensor-parallel',
    })
  })

  it('offers every count up to max_gpus_per_node with no model selected', () => {
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '7') // 8 stops: 1..8
  })

  it('raises the range to 72 stops for an NVL72 rack under pipeline parallel', () => {
    useUIStore.setState({
      selectedGPU: gpu('nvidia-gb300-nvl72'),
      shardingStrategy: 'pipeline-parallel',
    })
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', '71')
  })

  it('offers only valid tensor-parallel degrees (R14): Llama 3.1 70B on 8 GPUs = 1, 2, 4, 8', () => {
    useUIStore.setState({ selectedModel: model('meta-llama-llama-3.1-70b') })
    render(<GPUCountSelector />)
    const slider = screen.getByRole('slider')
    expect(slider).toHaveAttribute('max', '3')
    fireEvent.change(slider, { target: { value: '2' } })
    expect(useUIStore.getState().numGPUs).toBe(4)
  })

  it('renders no slider for a single-GPU part', () => {
    useUIStore.setState({ selectedGPU: gpu('apple-m3-ultra') })
    render(<GPUCountSelector />)
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    expect(screen.getByText(/Single GPU/)).toBeInTheDocument()
  })

  it('falls back to the shared bound when no GPU is selected', () => {
    useUIStore.setState({ selectedGPU: null })
    render(<GPUCountSelector />)
    expect(screen.getByRole('slider')).toHaveAttribute('max', String(maxGPUsFor(null) - 1))
    expect(maxGPUsFor(null)).toBe(MAX_GPUS_PER_NODE)
  })

  it('states the cap in its tooltip', () => {
    render(<GPUCountSelector />)
    // InfoTip only renders its `text` prop into the DOM once its trigger is opened.
    fireEvent.click(screen.getByRole('button', { name: /more info/i }))
    expect(screen.getByText(/Capped at 8/)).toBeInTheDocument()
  })
})
