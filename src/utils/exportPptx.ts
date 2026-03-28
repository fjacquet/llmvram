import type {
  InferenceVRAMBreakdown,
  MultiGPUVRAMBreakdown,
  PerformanceEstimate,
} from '@engines/types'
import type { GPU, Model } from '@utils/schemas'
import type Decimal from 'decimal.js'

interface ExportPptxParams {
  model: Model
  gpu: GPU
  quantization: string
  numGPUs: number
  sequenceLength: number
  batchSize: number
  vram: InferenceVRAMBreakdown
  performance: PerformanceEstimate
  multiGPU: MultiGPUVRAMBreakdown | null
}

function gbStr(val: Decimal): string {
  return `${val.toFixed(2)} GB`
}

function pctStr(num: Decimal, denom: Decimal): string {
  if (denom.isZero()) return '0.0%'
  return `${num.div(denom).mul(100).toFixed(1)}%`
}

// Color palette — no # prefix for pptxgenjs
const C = {
  headerFill: { color: '1E3A5F' },
  altRowFill: { color: 'F1F5F9' },
  whiteFill: { color: 'FFFFFF' },
  modelWeights: '6366F1',
  kvCache: '10B981',
  activations: 'F59E0B',
  framework: '8B5CF6',
  communication: 'EF4444',
  bodyText: '374151',
  tableBorder: 'E5E7EB',
  metricBoxFill: { color: 'E0E7FF' },
  darkBlue: '1E3A5F',
} as const

export async function exportPptx(params: ExportPptxParams): Promise<void> {
  const {
    model,
    gpu,
    quantization,
    numGPUs,
    sequenceLength,
    batchSize,
    vram,
    performance,
    multiGPU,
  } = params

  const PptxGenJS = (await import('pptxgenjs')).default
  const pptx = new PptxGenJS()

  pptx.layout = 'LAYOUT_WIDE'
  pptx.title = `LLM VRAM Estimate — ${model.name}`

  // Define slide master — applied to all slides
  pptx.defineSlideMaster({
    title: 'LLMVRAM',
    background: { color: 'F8FAFC' },
    objects: [
      // Full-width dark header bar
      { rect: { x: 0, y: 0, w: '100%', h: 0.6, fill: { color: '1E3A5F' } } },
      // App name in header
      {
        text: {
          text: 'LLM VRAM Calculator',
          options: {
            x: 0.4,
            y: 0.12,
            w: 9,
            h: 0.36,
            color: 'FFFFFF',
            fontSize: 13,
            bold: true,
            valign: 'middle',
          },
        },
      },
    ],
  })

  // ─── Slide 1: Configuration Summary ─────────────────────────────────────────
  const slide1 = pptx.addSlide({ masterName: 'LLMVRAM' })

  // Title (model + GPU name)
  slide1.addText(`${model.name} — ${gpu.name}`, {
    x: 0.4,
    y: 0.75,
    w: 12.5,
    h: 0.5,
    fontSize: 20,
    bold: true,
    color: C.darkBlue,
  })

  // Subtitle
  slide1.addText('LLM VRAM Estimation Report', {
    x: 0.4,
    y: 1.2,
    w: 12.5,
    h: 0.35,
    fontSize: 14,
    color: C.bodyText,
  })

  const contextK =
    model.context_length != null ? `${((model.context_length ?? 0) / 1000).toFixed(0)}K` : 'N/A'

  const configRows: [string, string][] = [
    ['Model', model.name],
    ['Architecture', model.architecture.toUpperCase()],
    ['Parameters', `${model.num_parameters_billion}B`],
    ['Context Length', `${contextK} tokens`],
    ['GPU', gpu.name],
    ['GPU VRAM', `${gpu.vram_gb} GB`],
    ['Number of GPUs', String(numGPUs)],
    ['Quantization', quantization.toUpperCase()],
    ['Sequence Length', `${sequenceLength.toLocaleString()} tokens`],
    ['Batch Size', String(batchSize)],
  ]

  slide1.addTable(
    [
      [
        { text: 'Parameter', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
        { text: 'Value', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
      ],
      ...configRows.map(([k, v], i) => [
        { text: k, options: { fill: i % 2 === 0 ? C.whiteFill : C.altRowFill } },
        { text: v, options: { fill: i % 2 === 0 ? C.whiteFill : C.altRowFill } },
      ]),
    ],
    {
      x: 0.4,
      y: 1.6,
      w: 12.5,
      colW: [4.5, 8],
      border: { pt: 1, color: C.tableBorder },
      fontSize: 13,
    },
  )

  // ─── Slide 2: VRAM Breakdown ─────────────────────────────────────────────────
  const slide2 = pptx.addSlide({ masterName: 'LLMVRAM' })

  slide2.addText('VRAM Requirements', {
    x: 0.4,
    y: 0.7,
    w: 12.5,
    h: 0.4,
    fontSize: 18,
    bold: true,
    color: C.darkBlue,
  })

  // Donut chart (LEFT side)
  slide2.addChart(
    pptx.ChartType.doughnut,
    [
      {
        name: 'VRAM',
        labels: ['Model Weights', 'KV Cache', 'Activations', 'Framework Overhead'],
        values: [
          vram.modelWeights.toNumber(),
          vram.kvCache.toNumber(),
          vram.activations.toNumber(),
          vram.frameworkOverhead.toNumber(),
        ],
      },
    ],
    {
      x: 0.4,
      y: 0.75,
      w: 6.2,
      h: 5.0,
      chartColors: [C.modelWeights, C.kvCache, C.activations, C.framework],
      holeSize: 55,
      showLegend: true,
      legendPos: 'b',
      showPercent: false,
      dataLabelFontSize: 11,
      dataLabelColor: 'FFFFFF',
    },
  )

  // Breakdown table (RIGHT side)
  slide2.addTable(
    [
      [
        { text: 'Component', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
        { text: 'Size (GB)', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
        { text: '% of Total', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
      ],
      [
        { text: 'Model Weights', options: { fill: C.whiteFill } },
        { text: gbStr(vram.modelWeights), options: { fill: C.whiteFill } },
        { text: pctStr(vram.modelWeights, vram.total), options: { fill: C.whiteFill } },
      ],
      [
        { text: 'KV Cache', options: { fill: C.altRowFill } },
        { text: gbStr(vram.kvCache), options: { fill: C.altRowFill } },
        { text: pctStr(vram.kvCache, vram.total), options: { fill: C.altRowFill } },
      ],
      [
        { text: 'Activations', options: { fill: C.whiteFill } },
        { text: gbStr(vram.activations), options: { fill: C.whiteFill } },
        { text: pctStr(vram.activations, vram.total), options: { fill: C.whiteFill } },
      ],
      [
        { text: 'Framework Overhead', options: { fill: C.altRowFill } },
        { text: gbStr(vram.frameworkOverhead), options: { fill: C.altRowFill } },
        { text: pctStr(vram.frameworkOverhead, vram.total), options: { fill: C.altRowFill } },
      ],
      [
        { text: 'Total', options: { bold: true, fill: C.whiteFill } },
        { text: gbStr(vram.total), options: { bold: true, fill: C.whiteFill } },
        { text: '100.0%', options: { bold: true, fill: C.whiteFill } },
      ],
    ],
    {
      x: 6.8,
      y: 0.75,
      w: 6.4,
      colW: [3.0, 1.7, 1.7],
      border: { pt: 1, color: C.tableBorder },
      fontSize: 12,
    },
  )

  // ─── Slide 3: Multi-GPU Distribution (conditional) ───────────────────────────
  if (multiGPU && numGPUs > 1) {
    const slide3 = pptx.addSlide({ masterName: 'LLMVRAM' })

    slide3.addText('Multi-GPU Memory Distribution', {
      x: 0.4,
      y: 0.7,
      w: 12.5,
      h: 0.4,
      fontSize: 18,
      bold: true,
      color: C.darkBlue,
    })

    const gpuLabels = Array.from({ length: numGPUs }, (_, i) => `GPU ${i + 1}`)

    slide3.addChart(
      pptx.ChartType.bar,
      [
        {
          name: 'Model Weights',
          labels: gpuLabels,
          values: Array(numGPUs).fill(multiGPU.perGPU.modelWeights.toNumber()) as number[],
        },
        {
          name: 'KV Cache',
          labels: gpuLabels,
          values: Array(numGPUs).fill(multiGPU.perGPU.kvCache.toNumber()) as number[],
        },
        {
          name: 'Activations',
          labels: gpuLabels,
          values: Array(numGPUs).fill(multiGPU.perGPU.activations.toNumber()) as number[],
        },
        {
          name: 'Framework & NCCL',
          labels: gpuLabels,
          values: Array(numGPUs).fill(multiGPU.perGPU.frameworkOverhead.toNumber()) as number[],
        },
        {
          name: 'Communication',
          labels: gpuLabels,
          values: Array(numGPUs).fill(multiGPU.perGPU.communicationOverhead.toNumber()) as number[],
        },
      ],
      {
        x: 0.4,
        y: 0.75,
        w: 12.5,
        h: 4.2,
        barDir: 'col',
        barGrouping: 'stacked',
        chartColors: [C.modelWeights, C.kvCache, C.activations, C.framework, C.communication],
        showLegend: true,
        legendPos: 'r',
        valAxisMinVal: 0,
      },
    )

    // Stats summary below chart
    const bandwidth =
      multiGPU.interconnectBandwidthGBps > 0 ? `${multiGPU.interconnectBandwidthGBps} GB/s` : 'N/A'

    slide3.addText(
      [
        { text: 'Strategy: ', options: { bold: true } },
        { text: `${multiGPU.strategy}  ` },
        { text: 'Per-GPU Total: ', options: { bold: true } },
        { text: `${gbStr(multiGPU.totalPerGPU)}  ` },
        { text: 'GPU Capacity: ', options: { bold: true } },
        { text: `${gpu.vram_gb} GB  ` },
        { text: 'Utilization: ', options: { bold: true } },
        { text: `${multiGPU.utilizationPercent.toFixed(1)}%  ` },
        { text: 'Interconnect BW: ', options: { bold: true } },
        { text: bandwidth },
      ],
      {
        x: 0.4,
        y: 5.05,
        w: 12.5,
        h: 0.45,
        fontSize: 12,
        color: C.bodyText,
      },
    )
  }

  // ─── Slide 4: Performance Estimate ──────────────────────────────────────────
  const slide4 = pptx.addSlide({ masterName: 'LLMVRAM' })

  slide4.addText('Performance Estimate', {
    x: 0.4,
    y: 0.7,
    w: 12.5,
    h: 0.4,
    fontSize: 18,
    bold: true,
    color: C.darkBlue,
  })

  const ttftMs = performance.timeToFirstToken.mul(1000).toFixed(1)
  const bottleneckLabel =
    performance.bottleneck === 'memory'
      ? 'Memory Bandwidth'
      : performance.bottleneck === 'compute'
        ? 'Compute'
        : 'Balanced'

  // Three metric boxes
  const metricBoxes: { label: string; value: string }[] = [
    { label: 'Decode Speed', value: `${performance.tokensPerSecond.toFixed(1)} tok/s` },
    { label: 'Time to First Token', value: `${ttftMs} ms` },
    { label: 'Bottleneck', value: bottleneckLabel },
  ]
  const boxXPositions = [0.4, 4.6, 8.8] as const

  metricBoxes.forEach(({ label, value }, idx) => {
    const xPos = boxXPositions[idx] ?? 0.4
    // Background rounded rectangle
    slide4.addShape('roundRect', {
      x: xPos,
      y: 0.9,
      w: 3.8,
      h: 2.2,
      fill: C.metricBoxFill,
      line: { color: 'BFC9FF', pt: 1 },
      rectRadius: 0.05,
    })
    // Label text
    slide4.addText(label, {
      x: xPos + 0.15,
      y: 1.0,
      w: 3.5,
      h: 0.4,
      fontSize: 12,
      color: C.darkBlue,
      bold: true,
      align: 'center',
    })
    // Value text (large)
    slide4.addText(value, {
      x: xPos + 0.15,
      y: 1.5,
      w: 3.5,
      h: 1.0,
      fontSize: 18,
      color: C.darkBlue,
      bold: true,
      align: 'center',
      valign: 'middle',
    })
  })

  // Performance details table below metric boxes
  slide4.addTable(
    [
      [
        { text: 'Metric', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
        { text: 'Value', options: { bold: true, fill: C.headerFill, color: 'FFFFFF' } },
      ],
      [
        { text: 'Decode Speed', options: { fill: C.whiteFill } },
        {
          text: `${performance.tokensPerSecond.toFixed(1)} tokens/sec`,
          options: { fill: C.whiteFill },
        },
      ],
      [
        { text: 'Time to First Token', options: { fill: C.altRowFill } },
        { text: `${ttftMs} ms`, options: { fill: C.altRowFill } },
      ],
      [
        { text: 'Bottleneck', options: { fill: C.whiteFill } },
        { text: bottleneckLabel, options: { fill: C.whiteFill } },
      ],
      [
        { text: 'GPU Memory Bandwidth', options: { fill: C.altRowFill } },
        { text: `${gpu.memory_bandwidth_gbps} GB/s`, options: { fill: C.altRowFill } },
      ],
    ],
    {
      x: 0.4,
      y: 3.3,
      w: 12.5,
      colW: [6, 6.5],
      border: { pt: 1, color: C.tableBorder },
      fontSize: 13,
    },
  )

  await pptx.writeFile({ fileName: `llmvram-${model.name.replace(/\s+/g, '-')}.pptx` })
}
