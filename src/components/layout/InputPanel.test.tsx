import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real store reads matchMedia at module load: stub it before any import runs.
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
import { InputPanel } from './InputPanel'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => models.find((m) => m.id === id) ?? null
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null

function field(container: HTMLElement, id: string): HTMLElement | null {
  return container.querySelector(`#${id}`)
}

describe('InputPanel composition', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      pendingNotice: null,
    })
  })

  it('shows the essential inputs in inference', () => {
    const { container } = render(<InputPanel />)
    expect(screen.getByText('Inference Mode')).toBeVisible()
    for (const id of [
      'model-selector',
      'quantization-picker',
      'gpu-selector',
      'gpu-count',
      'node-count',
      'sequence-length',
      'concurrent-users',
    ]) {
      expect(field(container, id), id).toBeVisible()
    }
  })

  it('keeps every advanced input reachable', () => {
    useUIStore.setState({ numGPUs: 2 })
    const { container } = render(<InputPanel />)
    for (const id of ['batch-size', 'kv-quantization-picker', 'kv-tier']) {
      expect(field(container, id), id).toBeInTheDocument()
    }
    expect(screen.getByText('Intra-server sharding strategy')).toBeInTheDocument()
    expect(screen.getByText('Offloading Configuration')).toBeInTheDocument()
  })

  it('shows the fabric selector once the cluster spans servers', () => {
    useUIStore.setState({ numNodes: 2 })
    const { container } = render(<InputPanel />)
    expect(field(container, 'inter-node-fabric')).toBeInTheDocument()
  })

  it('hides servers in training mode', () => {
    useUIStore.setState({ mode: 'training' })
    const { container } = render(<InputPanel />)
    expect(field(container, 'node-count')).toBeNull()
    expect(screen.getByText('Training Configuration')).toBeVisible()
  })
})

describe('InputPanel layout (ADR 0005)', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      pendingNotice: null,
    })
  })

  it('collapses Advanced at the defaults', () => {
    const { container } = render(<InputPanel />)
    const advanced = screen.getByTestId('advanced-settings')
    expect(advanced).not.toHaveAttribute('open')
    expect(within(advanced).getByText('Advanced')).toBeVisible()
    expect(field(container, 'batch-size')).not.toBeVisible()
  })

  it('opens Advanced with "N settings changed" when a value differs from its default', () => {
    useUIStore.setState({ batchSize: 8, kvQuantization: 'fp8' })
    render(<InputPanel />)
    const advanced = screen.getByTestId('advanced-settings')
    expect(advanced).toHaveAttribute('open')
    expect(within(advanced).getByText(/2 settings changed/)).toBeVisible()
  })

  it('opens Advanced when a setting changes after render', () => {
    render(<InputPanel />)
    act(() => useUIStore.getState().setBatchSize(8))
    expect(screen.getByTestId('advanced-settings')).toHaveAttribute('open')
    expect(screen.getByText(/1 setting changed/)).toBeVisible()
  })

  it('does not reopen after the user closes it, when the changed count rises from 1 to 2', () => {
    render(<InputPanel />)
    act(() => useUIStore.getState().setBatchSize(8))
    const advanced = screen.getByTestId('advanced-settings')
    expect(advanced).toHaveAttribute('open')
    act(() => {
      ;(advanced as HTMLDetailsElement).open = false
      fireEvent(advanced, new Event('toggle'))
    })
    expect(advanced).not.toHaveAttribute('open')
    act(() => useUIStore.getState().setKVQuantization('fp8'))
    expect(advanced).not.toHaveAttribute('open')
  })

  it('keeps the strategy reachable at 1 GPU on a multi-GPU part, so pipeline parallel is choosable before R14 snaps', () => {
    render(<InputPanel />)
    expect(screen.getByText('Intra-server sharding strategy')).toBeInTheDocument()
  })

  it('offers no strategy on a single-GPU part', () => {
    useUIStore.setState({ selectedGPU: gpu('apple-m3-ultra') })
    render(<InputPanel />)
    expect(screen.queryByText('Intra-server sharding strategy')).not.toBeInTheDocument()
  })

  it('labels the GPU count as GPUs per replica', () => {
    render(<InputPanel />)
    expect(screen.getByText('GPUs per replica (in one server)')).toBeVisible()
  })

  it('hides inputs that are inert in training', () => {
    useUIStore.setState({ mode: 'training' })
    const { container } = render(<InputPanel />)
    for (const id of [
      'quantization-picker',
      'kv-quantization-picker',
      'concurrent-users',
      'node-count',
      'kv-tier',
      'gpu-count',
    ]) {
      expect(field(container, id), id).toBeNull()
    }
    expect(screen.queryByText('Offloading Configuration')).not.toBeInTheDocument()
    expect(screen.queryByText('Intra-server sharding strategy')).not.toBeInTheDocument()
  })

  it('shows the GPU count in training once a ZeRO preset makes it matter', () => {
    useUIStore.setState({ mode: 'training', frameworkPreset: 'deepspeed-zero3' })
    const { container } = render(<InputPanel />)
    expect(field(container, 'gpu-count')).toBeVisible()
  })

  it('the Advanced reset button is visible inside the section, restores the defaults, and collapses "N settings changed" to 0', () => {
    useUIStore.setState({ batchSize: 8, kvQuantization: 'fp8' })
    render(<InputPanel />)
    const advanced = screen.getByTestId('advanced-settings')
    const resetButton = within(advanced).getByRole('button', { name: 'Reset advanced settings' })
    expect(resetButton).toBeVisible()
    expect(within(advanced).getByText(/2 settings changed/)).toBeVisible()
    act(() => resetButton.click())
    expect(within(advanced).queryByText(/settings? changed/)).not.toBeInTheDocument()
    expect(useUIStore.getState().batchSize).toBe(DEFAULT_UI_CONFIG.batchSize)
    expect(useUIStore.getState().kvQuantization).toBe(DEFAULT_UI_CONFIG.kvQuantization)
  })
})
