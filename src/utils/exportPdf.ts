import { isRackGPU } from '@engines/config-rules'
import { isDefaultMix } from '@engines/context-mix'
import { resolveFabricSpec } from '@engines/fabric'
import { FRAMEWORK_PRESETS } from '@engines/frameworks'
import { KV_TIER_PRESETS } from '@engines/kv-tier'
import { useUIStore } from '@store/uiStore'
import { formatDuration } from '@utils/formatDuration'
import { mixLabel } from '@utils/formatTTFT'
import Decimal from 'decimal.js'
import { toast } from 'sonner'

/** The element the PDF captures: the results column, never the input form */
export const PDF_CAPTURE_ID = 'results-column'

const MODE_LABEL = { inference: 'Inference', sizing: 'Sizing', training: 'Training' } as const
const FOOTER = 'llmvram - Reference engine: vLLM; figures are estimates'
const MARGIN = 36
const GRAY = [107, 114, 128] as const

type Row = [label: string, value: string]
type UIState = ReturnType<typeof useUIStore.getState>

/** The facts the numbers rest on, read from the store so all three modes share one source */
export function assumptionRows(s: UIState): Row[] {
  const { selectedModel: model, selectedGPU: gpu } = s
  if (!model || !gpu) return []
  const gpus = s.numGPUs * s.numNodes
  const tier = s.kvTier.tier === 'none' ? null : KV_TIER_PRESETS[s.kvTier.tier].label
  const fabric = resolveFabricSpec(s.interNodeFabric, s.customFabric).label
  const rows: Row[] = [['Model', model.name]]
  if (s.mode === 'training') {
    const preset = FRAMEWORK_PRESETS[s.frameworkPreset]
    rows.push(
      ['Method', s.trainingMethod.toUpperCase()],
      ['Optimizer', s.optimizer],
      ['Precision', s.trainingPrecision.toUpperCase()],
      ['Framework', preset.zeroStage ? preset.name : 'Standard training'],
    )
  } else {
    rows.push(
      ['Weights', s.quantization.toUpperCase()],
      ['KV precision', s.kvQuantization.toUpperCase()],
    )
  }
  rows.push(['GPU', gpu.name], ['Context', `${s.sequenceLength.toLocaleString('en-US')} tokens`])
  if (s.mode === 'sizing') rows.push(['Users', s.concurrentUsers.toLocaleString('en-US')])
  else {
    rows.push([
      s.mode === 'training' ? 'Batch size' : 'Batch size (sequences)',
      String(s.batchSize),
    ])
    if (s.mode === 'inference') rows.push(['Concurrent users', String(s.concurrentUsers)])
    rows.push([
      'GPUs',
      s.numNodes > 1 ? `${gpus} (${s.numGPUs} per server x ${s.numNodes} servers)` : String(gpus),
    ])
  }
  if ((s.mode === 'sizing' && !isRackGPU(gpu)) || (s.mode !== 'sizing' && s.numNodes > 1))
    rows.push(['Network between servers', fabric])
  if (s.mode !== 'training') {
    if (!isDefaultMix(s.contextMix, s.sequenceLength))
      rows.push(['Context mix', mixLabel(s.contextMix).replace('mix: ', '')])
    if (s.ttftTargetSeconds !== null)
      rows.push(['TTFT target', formatDuration(new Decimal(s.ttftTargetSeconds))])
  }
  if (tier && s.mode !== 'training') rows.push(['KV tier', tier])
  return rows
}

/** Y offsets (CSS px from the element top) of block edges, where a page may break (never right after a heading) */
function blockBreaks(root: HTMLElement, maxDepth = 3): number[] {
  const top = root.getBoundingClientRect().top
  const out = new Set<number>()
  const walk = (el: Element, depth: number) => {
    for (const child of Array.from(el.children)) {
      const r = child.getBoundingClientRect()
      if (r.height > 0 && child.tagName !== 'SUMMARY') out.add(Math.round(r.bottom - top))
      if (depth < maxDepth) walk(child, depth + 1)
    }
  }
  walk(root, 1)
  return [...out].sort((a, b) => a - b)
}

/**
 * Captures the results column and lays it out on A4 behind a title and an
 * assumptions table. The capture is restyled in the clone only (light theme,
 * no buttons or tooltips, details open), so the live page never changes.
 */
export async function exportCalculatorPdf(): Promise<void> {
  const captureEl = document.getElementById(PDF_CAPTURE_ID)
  if (!captureEl) return
  const state = useUIStore.getState()
  try {
    // html2canvas-pro supports oklch colors (Tailwind v4)
    const [{ default: html2canvasPro }, { jsPDF }] = await Promise.all([
      import('html2canvas-pro'),
      import('jspdf'),
    ])

    // Charts pick their colours from the store, so the capture needs the light theme for real
    const wasDark = state.isDarkMode
    if (wasDark) {
      useUIStore.setState({ isDarkMode: false })
      await new Promise((r) => setTimeout(r, 150))
    }
    let breaks: number[] = []
    let canvas: HTMLCanvasElement
    try {
      canvas = await html2canvasPro(captureEl, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#ffffff',
        logging: false,
        windowWidth: 1280,
        onclone: (_doc, el) => {
          for (const x of Array.from(el.querySelectorAll('button, [data-html2canvas-ignore]')))
            x.remove()
          for (const d of Array.from(el.querySelectorAll('details'))) d.open = true
          for (const m of Array.from(el.querySelectorAll('summary'))) m.style.display = 'block'
          // The grid gives the results column 7/12 of the page; the PDF wants a fixed readable width
          const grid = el.parentElement
          if (grid) {
            grid.style.display = 'block'
            for (const sib of Array.from(grid.children)) if (sib !== el) sib.remove()
          }
          el.style.width = '860px'
          breaks = blockBreaks(el)
        },
      })
    } finally {
      if (wasDark) useUIStore.setState({ isDarkMode: true })
    }

    const pdf = new jsPDF({ unit: 'pt', format: 'a4', orientation: 'portrait' })
    const pageW = pdf.internal.pageSize.getWidth()
    const pageH = pdf.internal.pageSize.getHeight()
    const contentW = pageW - MARGIN * 2
    const date = new Date().toISOString().slice(0, 10)
    const model = state.selectedModel?.name ?? 'estimate'

    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(18)
    pdf.text(`${model} - ${MODE_LABEL[state.mode]}`, MARGIN, MARGIN + 14)
    pdf.setFont('helvetica', 'normal')
    pdf.setFontSize(10)
    pdf.setTextColor(...GRAY)
    pdf.text(`LLM VRAM estimate - ${date}`, MARGIN, MARGIN + 30)
    pdf.setTextColor(17, 24, 39)

    let y = MARGIN + 52
    pdf.setFont('helvetica', 'bold')
    pdf.setFontSize(11)
    pdf.text('Assumptions', MARGIN, y)
    y += 8
    pdf.setFontSize(9)
    assumptionRows(state).forEach(([label, value], i) => {
      if (i % 2 === 0) {
        pdf.setFillColor(243, 244, 246)
        pdf.rect(MARGIN, y, contentW, 16, 'F')
      }
      pdf.setFont('helvetica', 'normal')
      pdf.setTextColor(...GRAY)
      pdf.text(label, MARGIN + 6, y + 11)
      pdf.setTextColor(17, 24, 39)
      pdf.text(value, MARGIN + 170, y + 11)
      y += 16
    })
    y += 14

    // Slice the capture into pages, preferring to cut between blocks rather than through one
    const pxPerPt = canvas.width / contentW
    const pxBreaks = breaks.map((b) => (b * canvas.width) / 860)
    const footerRoom = 44
    let start = 0
    let first = true
    while (start < canvas.height) {
      if (!first) {
        pdf.addPage()
        y = MARGIN
      }
      first = false
      const pageRoomPx = (pageH - footerRoom - y) * pxPerPt
      const limit = start + pageRoomPx
      const fitsAll = limit >= canvas.height
      const cut = fitsAll
        ? canvas.height
        : (pxBreaks.filter((b) => b > start + pageRoomPx * 0.4 && b <= limit).pop() ?? limit)
      const slice = document.createElement('canvas')
      slice.width = canvas.width
      slice.height = Math.ceil(cut - start)
      slice
        .getContext('2d')
        ?.drawImage(canvas, 0, start, canvas.width, slice.height, 0, 0, canvas.width, slice.height)
      pdf.addImage(
        slice.toDataURL('image/jpeg', 0.95),
        'JPEG',
        MARGIN,
        y,
        contentW,
        slice.height / pxPerPt,
      )
      start = cut
    }

    const pages = pdf.getNumberOfPages()
    pdf.setFontSize(8)
    pdf.setTextColor(...GRAY)
    for (let p = 1; p <= pages; p++) {
      pdf.setPage(p)
      pdf.text(`${FOOTER} - ${date}`, MARGIN, pageH - 20)
      pdf.text(`${p} / ${pages}`, pageW - MARGIN, pageH - 20, { align: 'right' })
    }

    pdf.save(`llmvram-${state.selectedModel?.name ?? 'estimate'}.pdf`)
    toast.success('PDF exported')
  } catch (err) {
    toast.error(`PDF export failed: ${err instanceof Error ? err.message : String(err)}`)
  }
}
