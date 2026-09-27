import { DECIMAL_GB_PER_GIB } from '@engines/kv-tier'
import { interconnectLabel } from '@engines/multi-gpu'
import { roundOffloadSlowdown } from '@engines/offloading'
import type { weightSource } from '@engines/quantization'
import type {
  InferenceVRAMBreakdown,
  MultiGPUVRAMBreakdown,
  PerformanceEstimate,
} from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import { firstTokenLabel } from '@utils/perfLabels'
import type { GPU, Model } from '@utils/schemas'
import type Decimal from 'decimal.js'

export interface ExportPptxOffload {
  target: 'cpu-ram' | 'nvme'
  mode: 'percentage' | 'layers'
  percentage: number
  layers: number
  kvOffloaded: boolean
  /** Total offloaded memory, GiB */
  offloadedGiB: number
  /** Decode offload slowdown ratio (see PerformanceEstimate.offloadSlowdown); null = negligible */
  slowdown: number | null
  /** Host capacity actually checked against (per-server capacity x server count), decimal GB */
  hostCapacityGB: number
  /** Whether the offloaded memory exceeds hostCapacityGB */
  exceedsHost: boolean
}

export interface ExportPptxParams {
  model: Model
  /** The effective GPU multiGPU/performance were computed from — with any
   *  interconnectOverride already applied (see useResultExports), not the raw store
   *  selection, or the Interconnect BW row can name a link the numbers weren't. */
  gpu: GPU
  quantization: string
  /** Total GPUs across the whole cluster (gpusPerNode × numNodes), not per-server */
  numGPUs: number
  /** Number of servers in the cluster; 1 for a single-node configuration */
  numNodes: number
  sequenceLength: number
  batchSize: number
  /** Concurrent users/sessions sized for; shown as a config row when it differs from batchSize */
  concurrentUsers: number
  vram: InferenceVRAMBreakdown
  performance: PerformanceEstimate
  multiGPU: MultiGPUVRAMBreakdown | null
  /** Sessions that fit at this context (vLLM's "Maximum concurrency"); null when unknown */
  maxSessions: number | null
  /** Sessions held with the KV storage tier active; null when no tier is set */
  tierSessionsHeld: number | null
  /** The repo weights were measured from, or null when estimated — same shape as weightSource() */
  weightSource: ReturnType<typeof weightSource>
  /** CPU/RAM or NVMe offloading state; null when offloading is disabled */
  offload: ExportPptxOffload | null
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
    numNodes,
    sequenceLength,
    batchSize,
    vram,
    performance,
    multiGPU,
    maxSessions,
    tierSessionsHeld,
    weightSource,
    concurrentUsers,
    offload,
  } = params

  const PptxGenJS = (await import('pptxgenjs-plus')).default
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

  // Offloading rows: what's offloaded, how much slower decode is, and — when the
  // offloaded memory doesn't fit the host(s) — that it doesn't. See
  // hostLinkGBps/defaultHostCapacityGB (engines/offloading.ts) for how the caller
  // (ResultsPanel) derives offload.slowdown and offload.hostCapacityGB.
  const offloadTargetLabel = offload?.target === 'nvme' ? 'NVMe' : 'CPU RAM'
  const offloadAmountLabel = offload
    ? offload.mode === 'percentage'
      ? `${offload.percentage}% of weights`
      : `${offload.layers} layers`
    : ''
  const offloadDesc = offload
    ? `${offloadTargetLabel}: ${offloadAmountLabel}${offload.kvOffloaded ? ' + KV cache' : ''}, ${offload.offloadedGiB.toLocaleString('en-US', { maximumFractionDigits: 0 })} GiB`
    : ''
  const slowdownRounded = offload ? roundOffloadSlowdown(offload.slowdown) : null
  const slowdownLabel =
    slowdownRounded === null ? 'no measurable slowdown' : `≈ ${slowdownRounded}× slower decode`
  const neededHostGB = offload ? offload.offloadedGiB * DECIMAL_GB_PER_GIB : 0

  const configRows: [string, string][] = [
    ['Model', model.name],
    ['Architecture', model.architecture.toUpperCase()],
    ['Parameters', `${model.num_parameters_billion}B`],
    ['Context Length', `${contextK} tokens`],
    ['GPU', gpu.name],
    ['GPU VRAM', `${gpu.vram_gb} GB`],
    [
      'GPUs per replica',
      numNodes > 1
        ? `${numGPUs / numNodes} per server × ${numNodes} servers (${numGPUs} total)`
        : `${numGPUs} (in one server)`,
    ],
    ...(numNodes > 1 ? ([['Servers', String(numNodes)]] as [string, string][]) : []),
    ['Quantization', quantization.toUpperCase()],
    ['Sequence Length', `${sequenceLength.toLocaleString('en-US')} tokens`],
    ['Batch Size', String(batchSize)],
    ...(concurrentUsers !== batchSize
      ? ([['Concurrent users', concurrentUsers.toLocaleString('en-US')]] as [string, string][])
      : []),
    ...(offload
      ? ([
          ['Offloading', offloadDesc],
          ['Offload slowdown', slowdownLabel],
          ...(offload.exceedsHost
            ? [
                [
                  'Host capacity',
                  `exceeded: ${neededHostGB.toLocaleString('en-US', { maximumFractionDigits: 0 })} GB needed vs ${offload.hostCapacityGB.toLocaleString('en-US', { maximumFractionDigits: 0 })} GB`,
                ],
              ]
            : []),
        ] as [string, string][])
      : []),
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

  slide1.addText('GB here means GiB (1024³ bytes), as nvidia-smi reports.', {
    x: 0.4,
    y: 7.05,
    w: 12.5,
    h: 0.3,
    fontSize: 9,
    color: C.bodyText,
    italic: true,
  })

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
        labels: [['Model Weights', 'KV Cache', 'Activations', 'Framework Overhead']],
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

    // One stacked bar, not one per GPU: every GPU in the configuration is
    // identical (all filled from breakdown.perGPU.*), so N bars just repeat
    // the same number N times. The category label carries the cluster total
    // instead, so that information survives even though the mock only
    // records (type, data) and not the chart options/title.
    const categoryLabel = [[`Per GPU (${numGPUs} GPU${numGPUs === 1 ? '' : 's'} total)`]]

    slide3.addChart(
      pptx.ChartType.bar,
      [
        {
          name: 'Model Weights',
          labels: categoryLabel,
          values: [multiGPU.perGPU.modelWeights.toNumber()],
        },
        {
          name: 'KV Cache',
          labels: categoryLabel,
          values: [multiGPU.perGPU.kvCache.toNumber()],
        },
        {
          name: 'Activations',
          labels: categoryLabel,
          values: [multiGPU.perGPU.activations.toNumber()],
        },
        {
          name: 'Framework & NCCL',
          labels: categoryLabel,
          values: [multiGPU.perGPU.frameworkOverhead.toNumber()],
        },
        {
          name: 'Communication',
          labels: categoryLabel,
          values: [multiGPU.perGPU.communicationOverhead.toNumber()],
        },
      ],
      {
        x: 0.4,
        // Clears the slide heading above (y 0.7, h 0.4). PowerPoint draws the
        // chart title inside the top of this frame, so starting at 0.75 would
        // overprint the two strings.
        y: 1.2,
        w: 12.5,
        // One category, so the frame height is the bar's thickness. 4.2 (the
        // height that suited N stacked bars) renders a single slab.
        h: 2.2,
        barDir: 'bar',
        barGrouping: 'stacked',
        chartColors: [C.modelWeights, C.kvCache, C.activations, C.framework, C.communication],
        showLegend: true,
        legendPos: 'r',
        valAxisMinVal: 0,
        // Pin the axis to the GPU's capacity so the bar renders as a fill
        // against its limit, the way the in-app meter does. Without it
        // PowerPoint auto-scales to the bar's own total and every export
        // looks full regardless of headroom. Over capacity, the total is the
        // larger of the two and the bar spans the axis.
        valAxisMaxVal: Math.max(gpu.vram_gb, multiGPU.totalPerGPU.toNumber()),
        showTitle: true,
        title: `Per GPU — ${gbStr(multiGPU.totalPerGPU)} / ${gpu.vram_gb} GB capacity`,
      },
    )

    // Stats summary below chart. interconnectLabel already renders "{name} — {n} GB/s"
    // (or "NVLink bridge — {n} GB/s") from the same (gpu, groupSize) pair multi-gpu.ts
    // used to compute interconnectBandwidthGBps, so this always names the link the
    // numbers actually came from — no re-assembly from a split label needed. `gpu`
    // must be the caller's effective GPU (interconnect override already applied, see
    // useResultExports), not the raw store selection, or the two can disagree.
    const bandwidth =
      multiGPU.interconnectBandwidthGBps > 0 ? interconnectLabel(gpu, multiGPU.gpusPerNode) : 'N/A'

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
        y: 3.6,
        w: 12.5,
        h: 0.45,
        fontSize: 12,
        color: C.bodyText,
      },
    )

    // The per-GPU chart and stats above show only what's actually on the GPU —
    // offloaded memory lives on the host, outside that figure.
    if (offload) {
      slide3.addText(
        `Offloaded to host: ${offload.offloadedGiB.toLocaleString('en-US', { maximumFractionDigits: 0 })} GiB (not included per GPU)`,
        {
          x: 0.4,
          y: 4.05,
          w: 12.5,
          h: 0.3,
          fontSize: 11,
          italic: true,
          color: C.bodyText,
        },
      )
    }
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

  const ttftLabel = formatDuration(performance.timeToFirstToken)
  const bottleneckLabel =
    performance.bottleneck === 'memory'
      ? 'Memory Bandwidth'
      : performance.bottleneck === 'compute'
        ? 'Compute'
        : 'Balanced'
  const prefillRegimeLabel =
    performance.prefillBottleneck === 'attention'
      ? 'attention-dominated (quadratic)'
      : 'weight-dominated (linear)'
  const prefillLabel = performance.prefillSeconds
    ? `${formatDuration(performance.prefillSeconds)} — ${prefillRegimeLabel}`
    : 'n/a'
  const prefillTableLabel = performance.prefillEstimateDegraded
    ? `${prefillLabel} (rough estimate — no FLOPS data for this GPU)`
    : prefillLabel

  // Four metric boxes
  const metricBoxes: { label: string; value: string }[] = [
    { label: 'Decode Speed', value: `${performance.tokensPerSecond.toFixed(1)} tok/s` },
    { label: firstTokenLabel(batchSize), value: ttftLabel },
    { label: 'Bottleneck', value: bottleneckLabel },
    {
      label: 'Prompt Processing',
      value: performance.prefillSeconds ? formatDuration(performance.prefillSeconds) : 'n/a',
    },
  ]
  const boxXPositions = [0.4, 3.6, 6.8, 10.0] as const
  const boxWidth = 2.9

  metricBoxes.forEach(({ label, value }, idx) => {
    const xPos = boxXPositions[idx] ?? 0.4
    // Background rounded rectangle
    slide4.addShape('roundRect', {
      x: xPos,
      // Clears the slide heading above (y 0.7, h 0.4), which the cards used to
      // overlap by 0.2" and clip. Height drops to keep the table below at 3.3.
      y: 1.2,
      w: boxWidth,
      h: 1.9,
      fill: C.metricBoxFill,
      line: { color: 'BFC9FF', pt: 1 },
      rectRadius: 0.05,
    })
    // Label text
    slide4.addText(label, {
      x: xPos + 0.15,
      y: 1.35,
      w: boxWidth - 0.3,
      h: 0.4,
      fontSize: 12,
      color: C.darkBlue,
      bold: true,
      align: 'center',
    })
    // Value text (large)
    slide4.addText(value, {
      x: xPos + 0.15,
      y: 1.85,
      w: boxWidth - 0.3,
      h: 1.0,
      fontSize: 14,
      color: C.darkBlue,
      bold: true,
      align: 'center',
      valign: 'middle',
    })
  })

  // Capacity rows: sessions that fit, sessions held with the KV tier, and
  // whether the weight sizes above are measured or estimated. Optional/absent
  // fields are omitted rather than shown as "N/A" (see ResultsPanel: not every
  // configuration has a KV tier, and a GPU without FLOPS data can still lack
  // a max-sessions figure).
  const capacityRows: [string, string][] = []
  if (maxSessions != null) {
    capacityRows.push([
      `Max concurrent sessions (at ${sequenceLength.toLocaleString('en-US')} tokens)`,
      maxSessions.toLocaleString('en-US'),
    ])
  }
  if (tierSessionsHeld != null) {
    capacityRows.push(['Sessions held with KV tier', tierSessionsHeld.toLocaleString('en-US')])
  }
  // Unlike the two checks above, `null` is a meaningful value here (estimated,
  // no reference checkpoint) distinct from the field being absent (test call
  // sites that predate this field) — so this checks undefined specifically.
  if (weightSource !== undefined) {
    capacityRows.push([
      'Weights',
      weightSource ? `measured from ${weightSource}` : 'estimated (no reference checkpoint)',
    ])
  }

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
        { text: firstTokenLabel(batchSize), options: { fill: C.altRowFill } },
        { text: ttftLabel, options: { fill: C.altRowFill } },
      ],
      [
        { text: 'Bottleneck', options: { fill: C.whiteFill } },
        { text: bottleneckLabel, options: { fill: C.whiteFill } },
      ],
      [
        { text: 'Prompt Processing', options: { fill: C.altRowFill } },
        { text: prefillTableLabel, options: { fill: C.altRowFill } },
      ],
      [
        { text: 'GPU Memory Bandwidth', options: { fill: C.whiteFill } },
        { text: `${gpu.memory_bandwidth_gbps} GB/s`, options: { fill: C.whiteFill } },
      ],
      ...capacityRows.map(([k, v], i) => [
        { text: k, options: { fill: (5 + i) % 2 === 0 ? C.whiteFill : C.altRowFill } },
        { text: v, options: { fill: (5 + i) % 2 === 0 ? C.whiteFill : C.altRowFill } },
      ]),
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

  const safeName =
    model.name
      .replace(/\s+/g, '-')
      .replace(/[/\\:*?"<>|]+/g, '_')
      .replace(/^[-_]+|[-_]+$/g, '') || 'model'
  await pptx.writeFile({ fileName: `llmvram-${safeName}.pptx` })
}
