import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DECODE_EFFICIENCY, PREFILL_EFFICIENCY } from '@engines/constants'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
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

import { useComparisonStore } from '@store/comparisonStore'
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

  it('reads "Does Not Fit" above vLLM 0.9 utilisation (audit M4)', async () => {
    useUIStore.setState({
      numGPUs: 1,
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      sequenceLength: 16384,
    })
    render(<ResultsPanel />)
    await screen.findAllByText(/\d+%/)
    expectSomeVisible(/Does Not Fit/)
    // the fit gauge (an <output>): 90-95% fit under the old 100% rule, not vLLM's 90%
    expect(screen.getAllByRole('status')[0]?.textContent).toMatch(/Does Not Fit.*9[0-5]%/)
  })

  it('applies the 90% rule to the on-device share when offloading', async () => {
    useUIStore.setState({
      numGPUs: 1,
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      sequenceLength: 16384,
      offloadingEnabled: true,
      offloadTarget: 'cpu-ram',
      offloadMode: 'percentage',
      offloadPercentage: 4,
    })
    render(<ResultsPanel />)
    await screen.findAllByText(/\d+%/)
    // 72.7 of 80 GB on device (91%): over 90%, with the host far from full
    expect(screen.getAllByRole('status')[0]?.textContent).toMatch(/Does Not Fit.*91%/)
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

describe('ResultsPanel layout (ADR 0005)', () => {
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

  it('leads with a visible verdict: fit, decode, first token, sessions', async () => {
    render(<ResultsPanel />)
    const verdict = await screen.findByTestId('verdict')
    expect(verdict).toBeVisible()
    expect(within(verdict).getByText(/Fits Comfortably|Tight Fit|Does Not Fit/)).toBeVisible()
    // 2026-11-11: the verdict now shows expected tok/s plus a "Theoretical ceiling" line (spec P3)
    expect(within(verdict).getByText(/^[\d.]+ tokens\/sec$/)).toBeVisible()
    expect(within(verdict).getByText(/^Theoretical ceiling: [\d.]+ tokens\/sec$/)).toBeVisible()
    expect(within(verdict).getByText('Time to first token')).toBeVisible()
    expect(within(verdict).getByText('Max sessions at 4,096 tokens')).toBeVisible()
  })

  it('collapses the details by default', async () => {
    render(<ResultsPanel />)
    const details = await screen.findByTestId('result-details')
    expect(details).not.toHaveAttribute('open')
    expect(within(details).getByText(/Weights (measured from|estimated)/)).not.toBeVisible()
  })

  it('keeps Details open across a recalculation', async () => {
    render(<ResultsPanel />)
    const details = (await screen.findByTestId('result-details')) as HTMLDetailsElement
    act(() => {
      details.open = true
      details.dispatchEvent(new Event('toggle'))
    })
    act(() => useUIStore.getState().setBatchSize(2))
    await screen.findAllByText(/tokens\/sec/)
    expect(await screen.findByTestId('result-details')).toHaveAttribute('open')
  })
})

describe('ResultsPanel expected vs ceiling (spec P3, P4b)', () => {
  beforeEach(() => {
    useUIStore.setState({
      ...DEFAULT_UI_CONFIG,
      selectedModel: model('meta-llama-llama-3.1-70b'),
      selectedGPU: gpu('nvidia-h100-80gb-sxm'),
      quantization: 'fp8',
      numGPUs: 2,
      pendingNotice: null,
    })
    useComparisonStore.setState({ snapshots: [] })
  })

  it('saves expected as the comparison headline and keeps the ceiling beside it', async () => {
    render(<ResultsPanel />)
    await screen.findAllByText(/tokens\/sec/)
    fireEvent.click(screen.getByRole('button', { name: /Save to Compare/ }))
    const results = useComparisonStore.getState().snapshots[0]?.results
    if (!results) throw new Error('no snapshot saved')
    expect(results.tokensPerSecond).toBeCloseTo(
      results.ceilingTokensPerSecond * DECODE_EFFICIENCY.toNumber(),
      6,
    )
    expect(results.timeToFirstToken).toBeCloseTo(
      results.ceilingTimeToFirstToken / PREFILL_EFFICIENCY.toNumber(),
      6,
    )
  })

  it('compares the KV tier resume against the CEILING prefill (P4b)', async () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_UI_CONFIG.kvTier, tier: 'host-pcie' } })
    render(<ResultsPanel />)
    const line = await screen.findByText(/vs recompute/)
    const recompute = line.textContent?.match(/vs recompute (\S+ \S+) \(/)?.[1]
    const promptLabel = screen.getByText('Prompt Processing')
    const headline = promptLabel.nextElementSibling?.textContent
    const ceiling = promptLabel.parentElement?.textContent?.match(
      /Theoretical ceiling: ([\d.]+ (?:ms|s|min|h))/,
    )?.[1]
    expect(recompute).toBeDefined()
    expect(recompute).toBe(ceiling)
    expect(recompute).not.toBe(headline)
  })
})
