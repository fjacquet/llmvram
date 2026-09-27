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
import { ResultsPanel } from './ResultsPanel'

const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)
const model = (id: string) => models.find((m) => m.id === id) ?? null
const gpu = (id: string) => gpus.find((g) => g.id === id) ?? null

/** At least one element matching `text` is visible (a copy may sit in a collapsed section) */
function expectSomeVisible(text: RegExp) {
  const visible = screen.getAllByText(text).some((el) => {
    try {
      expect(el).toBeVisible()
      return true
    } catch {
      return false
    }
  })
  expect(visible, String(text)).toBe(true)
}

describe('ResultsPanel composition', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      quantization: 'fp8',
      numGPUs: 2,
      pendingNotice: null,
    })
  })

  it('shows fit, decode speed, the first-token figure and max sessions', async () => {
    render(<ResultsPanel />)
    await screen.findAllByText(/tokens\/sec/)
    expectSomeVisible(/Fits Comfortably|Tight Fit|Does Not Fit/)
    expectSomeVisible(/tokens\/sec/)
    expectSomeVisible(/time to first token/i)
    expectSomeVisible(/sessions at 4,096 tokens/)
  })

  it('keeps the interconnect warning visible (W2 at TP-8 over PCIe 5)', async () => {
    useUIStore.setState({ selectedGPU: gpu('nvidia-h100-80gb-pcie'), numGPUs: 8 })
    render(<ResultsPanel />)
    await screen.findAllByText(/may have significant communication overhead/)
    expectSomeVisible(/may have significant communication overhead/)
  })

  it('keeps soft warnings visible (W8: 256 experts over 6 GPUs)', async () => {
    useUIStore.setState({
      selectedModel: model('deepseek-r1'),
      shardingStrategy: 'expert-parallel',
      numGPUs: 6,
    })
    render(<ResultsPanel />)
    expect(await screen.findByTestId('soft-warning-W8')).toBeVisible()
  })
})
