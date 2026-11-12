import { applyInterconnectOverride } from '@engines/multi-gpu'
import type { ContextMix, QuantizationFormat, SizingResult } from '@engines/types'
import type { UseInferenceCalculationResult } from '@hooks/useInferenceCalculation'
import { exportCalculatorPdf } from '@utils/exportPdf'
import { type ExportPptxParams, exportPptx, exportSizingPptx } from '@utils/exportPptx'
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
  | 'maxSessions'
  | 'tierSessionsHeld'
  | 'weightSource'
  | 'concurrentUsers'
  | 'offload'
  | 'contextMix'
  | 'ttftTargetSeconds'
  | 'ttftBurst'
>

interface UseResultExportsParams {
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  numGPUs: number
  numNodes: number
  /** The variant picked in InterconnectSelector; applied the same way useInferenceCalculation
   *  does, so the export names the link its numbers actually came from. */
  interconnectOverride: string | null
  result: UseInferenceCalculationResult['result']
}

/**
 * PDF and PPTX export handlers for the results panel.
 *
 * PDF captures the results column behind an assumptions table (utils/exportPdf); PPTX rebuilds the numbers into a native deck via exportPptx.
 */
export function useResultExports({
  selectedModel,
  selectedGPU,
  quantization,
  sequenceLength,
  batchSize,
  numGPUs,
  numNodes,
  interconnectOverride,
  result,
}: UseResultExportsParams) {
  const handleExportPDF = () => exportCalculatorPdf()

  const handleExportPptx = async (capacity: ExportPptxCapacity) => {
    if (!result || !selectedModel || !selectedGPU) return
    try {
      // Same effective GPU useInferenceCalculation computed result.multiGPU from
      // (interconnect override applied, bridge dropped) — otherwise the deck can
      // name a link (e.g. "NVLink bridge") the numbers were never computed from.
      const effectiveGPU = applyInterconnectOverride(selectedGPU, interconnectOverride)
      await exportPptx({
        model: selectedModel,
        gpu: effectiveGPU,
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

interface UseSizingExportsParams {
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  sequenceLength: number
  users: number
  kvQuantization: QuantizationFormat
  fabricLabel: string
  contextMix: ContextMix
  ttftTargetSeconds: number | null
  result: SizingResult | null
}

/** Sizing mode: the same PDF export, and a one-slide sizing deck */
export function useSizingExports(p: UseSizingExportsParams) {
  const handleExportPDF = () => exportCalculatorPdf()
  const handleExportPptx = async () => {
    if (!p.result || !p.selectedModel || !p.selectedGPU) return
    try {
      await exportSizingPptx({
        model: p.selectedModel,
        gpu: p.selectedGPU,
        quantization: p.quantization,
        sequenceLength: p.sequenceLength,
        users: p.users,
        kvQuantization: p.kvQuantization,
        fabricLabel: p.fabricLabel,
        contextMix: p.contextMix,
        ttftTargetSeconds: p.ttftTargetSeconds,
        result: p.result,
      })
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'PPTX export failed')
    }
  }
  return { handleExportPDF, handleExportPptx }
}
