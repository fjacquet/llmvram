import { act, renderHook } from '@testing-library/react'
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

import { useResultExports } from './useResultExports'

const params = {
  selectedModel: null,
  selectedGPU: null,
  quantization: 'fp16' as const,
  sequenceLength: 4096,
  batchSize: 1,
  numGPUs: 1,
  numNodes: 1,
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
