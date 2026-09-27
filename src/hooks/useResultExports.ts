import type { QuantizationFormat } from '@engines/types'
import type { UseInferenceCalculationResult } from '@hooks/useInferenceCalculation'
import { type ExportPptxParams, exportPptx } from '@utils/exportPptx'
import type { GPU, Model } from '@utils/schemas'
import { toast } from 'sonner'

/**
 * Capacity and offload figures computed in ResultsPanel after `result` is
 * known — max sessions, the KV tier, and offloading depend on the displayed
 * breakdown, which isn't available yet when useResultExports itself is
 * called, so they are passed in at export time instead of at
 * hook-construction time.
 */
type ExportPptxCapacity = Pick<
  ExportPptxParams,
  'maxSessions' | 'tierSessionsHeld' | 'weightSource' | 'concurrentUsers' | 'offload'
>

interface UseResultExportsParams {
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  numGPUs: number
  numNodes: number
  result: UseInferenceCalculationResult['result']
}

/**
 * PDF and PPTX export handlers for the results panel.
 *
 * PDF captures the whole calculator section (InputPanel + ResultsPanel) as a
 * screenshot; PPTX rebuilds the numbers into a native deck via exportPptx.
 */
export function useResultExports({
  selectedModel,
  selectedGPU,
  quantization,
  sequenceLength,
  batchSize,
  numGPUs,
  numNodes,
  result,
}: UseResultExportsParams) {
  const handleExportPDF = async () => {
    // Capture the full calculator section (InputPanel + ResultsPanel) so the PDF
    // includes both the assumptions and the results without duplicating content in the UI.
    const captureEl = document.getElementById('calculator-section')
    if (!captureEl) return
    try {
      // html2canvas-pro supports oklch colors (Tailwind v4)
      const [{ default: html2canvasPro }, { jsPDF }] = await Promise.all([
        import('html2canvas-pro'),
        import('jspdf'),
      ])

      // Force light mode so the capture uses correct contrast
      const root = document.documentElement
      const wasDark = root.classList.contains('dark')
      if (wasDark) root.classList.remove('dark')

      let canvas: HTMLCanvasElement
      try {
        canvas = await html2canvasPro(captureEl, {
          scale: 2,
          useCORS: true,
          backgroundColor: '#f9fafb', // gray-50 — matches page background
          logging: false,
          // windowWidth: 800 tells the renderer to use the mobile breakpoint:
          // lg:grid-cols-12 collapses to grid-cols-1, panels stack vertically.
          // This is a render-time option — the live app and mobile users are unaffected.
          windowWidth: 800,
        })
      } finally {
        if (wasDark) root.classList.add('dark')
      }

      const imgData = canvas.toDataURL('image/jpeg', 0.97)
      const pdf = new jsPDF({ unit: 'pt', format: 'a4', orientation: 'portrait' })

      const margin = 28.8 // 0.4 in × 72 pt/in
      const pageW = pdf.internal.pageSize.getWidth()
      const pageH = pdf.internal.pageSize.getHeight()
      const imgW = pageW - margin * 2
      const imgH = (canvas.height / canvas.width) * imgW

      // Slice the full-height image across multiple pages
      let heightLeft = imgH - (pageH - margin * 2)
      pdf.addImage(imgData, 'JPEG', margin, margin, imgW, imgH)

      while (heightLeft > 0) {
        pdf.addPage()
        pdf.addImage(imgData, 'JPEG', margin, margin - (imgH - heightLeft), imgW, imgH)
        heightLeft -= pageH - margin * 2
      }

      pdf.save(`llmvram-${selectedModel?.name ?? 'estimate'}.pdf`)
      toast.success('PDF exported')
    } catch (err) {
      toast.error(`PDF export failed: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  const handleExportPptx = async (capacity: ExportPptxCapacity) => {
    if (!result || !selectedModel || !selectedGPU) return
    try {
      await exportPptx({
        model: selectedModel,
        gpu: selectedGPU,
        quantization,
        // numGPUs from the store is per-node; the deck must report the true
        // cluster total (see IMPORTANT-1 in the multi-node fix wave).
        numGPUs: result.multiGPU?.numGPUs ?? numGPUs,
        numNodes: result.multiGPU?.numNodes ?? numNodes,
        sequenceLength,
        batchSize,
        vram: result.vram,
        performance: result.performance,
        multiGPU: result.multiGPU,
        ...capacity,
      })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'PPTX export failed')
    }
  }

  return { handleExportPDF, handleExportPptx }
}
