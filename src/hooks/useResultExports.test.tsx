import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { act, renderHook } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import Decimal from 'decimal.js'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { capture } = vi.hoisted(() => ({ capture: vi.fn() }))
vi.mock('html2canvas-pro', () => ({ default: capture }))
vi.mock('jspdf', () => ({
  jsPDF: class {
    internal = { pageSize: { getWidth: () => 595, getHeight: () => 842 } }
    addImage = vi.fn()
    addPage = vi.fn()
    save = vi.fn()
  },
}))
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn() } }))

// PPTX interconnect-override test below only needs to see WHICH gpu reaches
// exportPptx, not a rendered deck — mock it the way the PDF path mocks its libs.
const { exportPptx } = vi.hoisted(() => ({ exportPptx: vi.fn(async () => undefined) }))
vi.mock('@utils/exportPptx', () => ({ exportPptx }))

import { useResultExports } from './useResultExports'

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
    '<div id="calculator-section"><details id="advanced"><summary>A</summary></details><details id="details" open><summary>D</summary></details></div>',
  )
}
const isOpen = (id: string) => (document.getElementById(id) as HTMLDetailsElement).open

describe('useResultExports: PDF', () => {
  afterEach(() => {
    document.getElementById('calculator-section')?.remove()
    capture.mockReset()
  })

  it('expands every <details> for the capture, then restores each one exactly', async () => {
    mountSection()
    let openAtCapture: boolean[] = []
    capture.mockImplementation(async (el: HTMLElement) => {
      openAtCapture = Array.from(el.querySelectorAll('details')).map((d) => d.open)
      return { width: 100, height: 100, toDataURL: () => 'data:image/jpeg;base64,' }
    })
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(openAtCapture).toEqual([true, true])
    expect(isOpen('advanced')).toBe(false)
    expect(isOpen('details')).toBe(true)
  })

  it('restores the details when the capture fails', async () => {
    mountSection()
    capture.mockRejectedValue(new Error('canvas failed'))
    const { result } = renderHook(() => useResultExports(params))
    await act(() => result.current.handleExportPDF())
    expect(isOpen('advanced')).toBe(false)
    expect(isOpen('details')).toBe(true)
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
    performance: {
      tokensPerSecond: one,
      timeToFirstToken: one,
      prefillSeconds: null,
      prefillBottleneck: 'linear' as const,
      prefillEstimateDegraded: false,
      isComputeBound: false,
      isMemoryBound: true,
      bottleneck: 'memory' as const,
      offloadSlowdown: null,
    },
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
