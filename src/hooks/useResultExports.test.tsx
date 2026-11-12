import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { defaultMix } from '@engines/context-mix'
import { resolveFabricSpec } from '@engines/fabric'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { estimatePerformance } from '@engines/performance'
import { sizeDeployment } from '@engines/sizing'
import { useUIStore } from '@store/uiStore'
import { act, renderHook } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import Decimal from 'decimal.js'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { capture } = vi.hoisted(() => ({ capture: vi.fn() }))
vi.mock('html2canvas-pro', () => ({ default: capture }))
vi.mock('jspdf', () => ({
  jsPDF: class {
    internal = { pageSize: { getWidth: () => 595, getHeight: () => 842 } }
    addImage = vi.fn()
    addPage = vi.fn()
    save = vi.fn()
    setFont = vi.fn()
    setFontSize = vi.fn()
    setTextColor = vi.fn()
    setFillColor = vi.fn()
    text = vi.fn()
    rect = vi.fn()
    setPage = vi.fn()
    getNumberOfPages = () => 1
  },
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

// PPTX interconnect-override test below only needs to see WHICH gpu reaches
// exportPptx, not a rendered deck — mock it the way the PDF path mocks its libs.
const { exportPptx, exportSizingPptx } = vi.hoisted(() => ({
  exportPptx: vi.fn(async () => undefined),
  exportSizingPptx: vi.fn(async () => undefined),
}))
vi.mock('@utils/exportPptx', () => ({ exportPptx, exportSizingPptx }))

import { useResultExports, useSizingExports } from './useResultExports'

const params = {
  selectedModel: null,
  selectedGPU: null,
  quantization: 'fp16' as const,
  sequenceLength: 4096,
  batchSize: 1,
  numGPUs: 1,
  numNodes: 1,
  interconnectOverride: null,
  result: null,
}

function mountSection() {
  document.body.insertAdjacentHTML(
    'beforeend',
    '<div id="grid"><div id="inputs"></div><div id="results-column"><button>Export PDF</button><details id="advanced"><summary>A</summary></details><details id="details" open><summary>D</summary></details></div></div>',
  )
}
const isOpen = (id: string) => (document.getElementById(id) as HTMLDetailsElement).open
const fakeCanvas = { width: 100, height: 100 }

describe('useResultExports: PDF', () => {
  beforeEach(() => {
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,')
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
  })
  afterEach(() => {
    document.getElementById('grid')?.remove()
    capture.mockReset()
    vi.restoreAllMocks()
    useUIStore.setState({ isDarkMode: false })
  })

  it('captures the results column, not the input form', async () => {
    mountSection()
    capture.mockResolvedValue(fakeCanvas)
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(capture.mock.calls[0]?.[0]).toBe(document.getElementById('results-column'))
  })

  it('restyles the clone only: no buttons, details open, inputs gone, live page untouched', async () => {
    mountSection()
    capture.mockImplementation(
      async (el: HTMLElement, opts: { onclone: (d: Document, e: HTMLElement) => void }) => {
        const clone = el.cloneNode(true) as HTMLElement
        const grid = document.createElement('div')
        grid.append(document.createElement('div'), clone)
        opts.onclone(document, clone)
        expect(clone.querySelector('button')).toBeNull()
        expect(Array.from(clone.querySelectorAll('details')).every((d) => d.open)).toBe(true)
        expect(grid.children).toHaveLength(1)
        return fakeCanvas
      },
    )
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(document.querySelector('#results-column button')).not.toBeNull()
    expect(isOpen('advanced')).toBe(false)
  })

  it('renders in the light theme and puts dark mode back, even when the capture fails', async () => {
    mountSection()
    useUIStore.setState({ isDarkMode: true })
    let darkAtCapture: boolean | null = null
    capture.mockImplementation(async () => {
      darkAtCapture = useUIStore.getState().isDarkMode
      throw new Error('canvas failed')
    })
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(darkAtCapture).toBe(false)
    expect(useUIStore.getState().isDarkMode).toBe(true)
  })
})

describe('useResultExports: PPTX interconnect override', () => {
  afterEach(() => {
    exportPptx.mockClear()
  })

  const bridgedGpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-pcie')
  const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
  if (!bridgedGpu || !model) throw new Error('fixture not found')

  const one = new Decimal(1)
  const fakeResult = {
    vram: { modelWeights: one, kvCache: one, activations: one, frameworkOverhead: one, total: one },
    // 2026-11-11: built from the real engine (Task 6 added `expected`; the hand literal went stale)
    performance: estimatePerformance({
      model,
      gpu: bridgedGpu,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    }),
    offloading: null,
    multiGPU: null,
    interconnectWarning: null,
  }
  const capacity = {
    maxSessions: null,
    tierSessionsHeld: null,
    weightSource: null,
    concurrentUsers: 1,
    offload: null,
  }

  it('forwards the effective GPU (override applied, bridge dropped) to exportPptx, not the raw selection', async () => {
    // nvidia-h100-80gb-pcie has a 2-GPU NVLink bridge; the user overrode the link to PCIe.
    expect(bridgedGpu.nvlink_bridge).toBeDefined()

    const { result } = renderHook(() =>
      useResultExports({
        ...params,
        selectedModel: model,
        selectedGPU: bridgedGpu,
        interconnectOverride: 'pcie-5',
        result: fakeResult,
      }),
    )
    await act(() => result.current.handleExportPptx(capacity))

    expect(exportPptx).toHaveBeenCalledTimes(1)
    const passedGpu = exportPptx.mock.calls[0]?.[0]?.gpu
    expect(passedGpu.interconnect).toBe('pcie-5')
    expect(passedGpu.nvlink_bridge).toBeUndefined()
    // The raw store selection (still bridged) must not be what was passed through.
    expect(passedGpu).not.toBe(bridgedGpu)
  })

  it('passes the raw GPU unchanged when no override is set', async () => {
    const { result } = renderHook(() =>
      useResultExports({
        ...params,
        selectedModel: model,
        selectedGPU: bridgedGpu,
        interconnectOverride: null,
        result: fakeResult,
      }),
    )
    await act(() => result.current.handleExportPptx(capacity))

    expect(exportPptx.mock.calls[0]?.[0]?.gpu).toBe(bridgedGpu)
  })
})

describe('useSizingExports', () => {
  const rack = validateGPUs(gpusData).find((g) => g.id === 'nvidia-gb300-nvl72')
  const llama = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
  if (!rack || !llama) throw new Error('fixture not found')
  const sizing = sizeDeployment({
    model: llama,
    gpu: rack,
    quantization: 'fp8',
    kvQuantization: 'fp16',
    sequenceLength: 8192,
    users: 500,
    fabric: resolveFabricSpec('ethernet-800g', null),
    kvTier: DEFAULT_KV_TIER,
    contextMix: defaultMix(8192),
    ttftTargetSeconds: null,
  })
  const sizingParams = {
    selectedModel: llama,
    selectedGPU: rack,
    quantization: 'fp8' as const,
    sequenceLength: 8192,
    users: 500,
    kvQuantization: 'fp16' as const,
    fabricLabel: '800GbE (SONiC / RoCEv2)',
    contextMix: [{ tokens: 8192, share: 1 }],
    ttftTargetSeconds: 2,
    result: sizing,
  }

  afterEach(() => {
    document.getElementById('grid')?.remove()
    capture.mockReset()
    vi.restoreAllMocks()
    exportSizingPptx.mockClear()
  })

  it('PDF captures the results column, which holds the Sizing card', async () => {
    mountSection()
    vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/jpeg;base64,')
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    capture.mockResolvedValue(fakeCanvas)
    const { result } = renderHook(() => useSizingExports(sizingParams))
    await act(() => result.current.handleExportPDF())
    expect(capture.mock.calls[0]?.[0]).toBe(document.getElementById('results-column'))
  })

  it('PPTX hands the sizing result, KV precision and fabric to exportSizingPptx', async () => {
    const { result } = renderHook(() => useSizingExports(sizingParams))
    await act(() => result.current.handleExportPptx())
    expect(exportSizingPptx).toHaveBeenCalledWith(
      expect.objectContaining({
        model: llama,
        gpu: rack,
        result: sizing,
        kvQuantization: 'fp16',
        fabricLabel: '800GbE (SONiC / RoCEv2)',
        users: 500,
        ttftTargetSeconds: 2,
        contextMix: [{ tokens: 8192, share: 1 }],
      }),
    )
  })
})
