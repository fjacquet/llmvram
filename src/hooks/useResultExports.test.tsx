import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { resolveFabricSpec } from '@engines/fabric'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { estimatePerformance } from '@engines/performance'
import { sizeDeployment } from '@engines/sizing'
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
  })
  const sizingParams = {
    selectedModel: llama,
    selectedGPU: rack,
    quantization: 'fp8' as const,
    sequenceLength: 8192,
    users: 500,
    kvQuantization: 'fp16' as const,
    fabricLabel: '800GbE (SONiC / RoCEv2)',
    result: sizing,
  }

  afterEach(() => {
    document.getElementById('calculator-section')?.remove()
    capture.mockReset()
    exportSizingPptx.mockClear()
  })

  it('PDF captures the calculator section, which holds the Sizing card', async () => {
    mountSection()
    let captured: HTMLElement | null = null
    capture.mockImplementation(async (el: HTMLElement) => {
      captured = el
      return { width: 100, height: 100, toDataURL: () => 'data:image/jpeg;base64,' }
    })
    const { result } = renderHook(() => useSizingExports(sizingParams))
    await act(() => result.current.handleExportPDF())
    expect(captured).toBe(document.getElementById('calculator-section'))
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
      }),
    )
  })
})
