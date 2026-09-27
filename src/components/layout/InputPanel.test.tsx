import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { render, screen } from '@testing-library/react'
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
