import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { FABRIC_SPECS, resolveFabricSpec } from '@engines/fabric'
import { calculateInferenceVRAM } from '@engines/inference'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { applyInterconnectOverride } from '@engines/multi-gpu'
import { calculateMultiNodeVRAM } from '@engines/multi-node'
import { estimatePerformance } from '@engines/performance'
import { sizeDeployment } from '@engines/sizing'
import { formatDuration } from '@utils/formatDuration'
import { type GPU, type Model, validateGPUs, validateModels } from '@utils/schemas'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// pptxgenjs drives real browser/Node file I/O (pptx.writeFile) and canvas-ish
// chart rendering — none of which is relevant to IMPORTANT-1 (the deck must
// report the cluster TOTAL GPU count, not the per-node count). We mock the
// library and assert on the arguments exportPptx passes it instead.
interface RecordedTable {
  rows: unknown[]
}
interface Box {
  x?: number
  y?: number
  w?: number
  h?: number
}
interface RecordedChart {
  type: string
  data: Array<{ name: string; labels: string[][]; values: number[] }>
  // Chart options were originally discarded. They carry the geometry and the
  // axis bound, neither of which is visible in (type, data).
  opts: Box & { valAxisMaxVal?: number; showTitle?: boolean; title?: string }
}
interface RecordedText {
  text: string
  opts: Box
}
interface RecordedShape {
  kind: string
  opts: Box
}

const tables: RecordedTable[] = []
const charts: RecordedChart[] = []
const texts: RecordedText[] = []
const shapes: RecordedShape[] = []

class MockSlide {
  addText = vi.fn((text: unknown, opts?: Box) => {
    if (typeof text === 'string') {
      texts.push({ text, opts: opts ?? {} })
    } else if (Array.isArray(text)) {
      // Rich-text runs (e.g. the slide 3 stats line: alternating bold labels and
      // plain values) — concatenate so a test can still search the joined string.
      const joined = text
        .map((run) => (run && typeof run === 'object' && 'text' in run ? String(run.text) : ''))
        .join('')
      texts.push({ text: joined, opts: opts ?? {} })
    }
  })
  addShape = vi.fn((kind: string, opts?: Box) => {
    shapes.push({ kind, opts: opts ?? {} })
  })
  addTable = vi.fn((rows: unknown[]) => {
    tables.push({ rows })
  })
  addChart = vi.fn(
    (
      type: string,
      data: Array<{ name: string; labels: string[][]; values: number[] }>,
      opts?: RecordedChart['opts'],
    ) => {
      charts.push({ type, data, opts: opts ?? {} })
    },
  )
}

class MockPptxGenJS {
  layout = ''
  title = ''
  ChartType = { doughnut: 'doughnut', bar: 'bar' }
  defineSlideMaster = vi.fn()
  addSlide = vi.fn(() => new MockSlide())
  writeFile = vi.fn(async () => undefined)
}

vi.mock('pptxgenjs-plus', () => ({ default: MockPptxGenJS }))

const { exportPptx, exportSizingPptx } = await import('./exportPptx')

const model: Model = {
  id: 'test-llama-3-70b',
  name: 'Test Llama 3 70B',
  architecture: 'dense',
  num_parameters_billion: 70,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,
  intermediate_size: 28672,
}

const gpu: GPU = {
  id: 'nvidia-h100-80gb-sxm',
  name: 'NVIDIA H100 80GB SXM',
  manufacturer: 'nvidia',
  vram_gb: 80,
  memory_bandwidth_gbps: 3352,
  memory_type: 'HBM3',
  bus_width: 5120,
  fp16_tflops: 1979,
  fp32_tflops: 989,
  tier: 'datacenter',
  interconnect: 'nvlink-4',
  max_gpus_per_node: 8,
}

const realModel = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
const realGpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
if (!realModel || !realGpu) throw new Error('fixture not found')
// 2026-11-11: built from the real engine (Task 6 added `expected`; the hand literal went stale)
const performance = estimatePerformance({
  model: realModel,
  gpu: realGpu,
  quantization: 'fp16',
  sequenceLength: 4096,
  batchSize: 1,
})

function tableRows(index: number): [string, string][] {
  const call = tables[index]?.rows as Array<Array<{ text?: string; options?: unknown }>> | undefined
  if (!call) throw new Error(`no table recorded at index ${index}`)
  // Skip header row; flatten [label, value] text pairs.
  return call.slice(1).map((row) => [row[0]?.text ?? '', row[1]?.text ?? ''] as [string, string])
}

describe('exportPptx', () => {
  beforeEach(() => {
    tables.length = 0
    charts.length = 0
    texts.length = 0
    shapes.length = 0
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('reports the cluster TOTAL GPU count and a Servers row, not the per-node count (IMPORTANT-1)', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    // 4 servers x 8 GPUs/server = 32 GPUs total. The store's numGPUs is the
    // PER-NODE value (8); exportPptx must be handed the true total (32).
    const multiGPU = calculateMultiNodeVRAM({
      singleGPU,
      model,
      gpuVramGB: 80,
      gpusPerNode: 8,
      numNodes: 4,
      intraNodeStrategy: 'tensor-parallel',
      gpu,
      fabric: FABRIC_SPECS['ethernet-800g'],
      batchSize: 1,
      sequenceLength: 4096,
      quantization: 'fp16',
    })

    expect(multiGPU.numGPUs).toBe(32)

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: multiGPU.numGPUs, // caller passes the resolved total, per ResultsPanel
      numNodes: multiGPU.numNodes,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    // Slide 1 (config summary) is the first addTable call.
    const configRows = tableRows(0)
    expect(configRows).toContainEqual(['GPUs per replica', '8 per server × 4 servers (32 total)'])
    expect(configRows.some(([label]) => label === 'Number of GPUs')).toBe(false)
    expect(configRows).toContainEqual(['Servers', '4'])
    expect(configRows).toContainEqual([
      'GPU VRAM',
      `${gpu.vram_gb} GB per GPU, ${gpu.vram_gb * 32 >= 1000 ? `${((gpu.vram_gb * 32) / 1000).toFixed(1)} TB` : `${gpu.vram_gb * 32} GB`} across 32 GPUs`,
    ])

    // Slide 3's bar chart is a single stacked bar (one category, five series) —
    // every GPU is identical, so N repeated bars carried no information. The
    // TOTAL (32), not the per-node count (8), must appear in the category label,
    // since the mock only records (type, data) and not the chart title/options.
    const barChart = charts.find((c) => c.type === 'bar')
    expect(barChart).toBeDefined()
    expect(barChart?.data[0]?.values).toHaveLength(1)
    expect(barChart?.data[0]?.labels[0]?.[0]).toContain('32 GPUs total')
    expect(barChart?.data[0]?.labels[0]?.[0]).not.toContain('8 GPUs total')

    // The value axis must be pinned to the GPU's capacity. Without a max,
    // PowerPoint auto-scales to the bar's own total and the exported bar looks
    // full at any utilization — losing the headroom reading the in-app meter
    // exists to give.
    const totalPerGPU = multiGPU.totalPerGPU.toNumber()
    expect(barChart?.opts.valAxisMaxVal).toBeCloseTo(Math.max(gpu.vram_gb, totalPerGPU), 5)
    expect(barChart?.opts.valAxisMaxVal).toBeGreaterThanOrEqual(totalPerGPU)

    // The chart frame must clear the slide heading above it. PowerPoint draws
    // the chart title inside the top of the frame, so an overlapping frame
    // overprints the two strings.
    const heading = texts.find((x) => x.text === 'Multi-GPU Memory Distribution')
    expect(heading).toBeDefined()
    const headingBottom = (heading?.opts.y ?? 0) + (heading?.opts.h ?? 0)
    expect(barChart?.opts.y).toBeGreaterThanOrEqual(headingBottom)
  })

  it('shows the resolved interconnect link name next to the bandwidth', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })
    const multiGPU = calculateMultiNodeVRAM({
      singleGPU,
      model,
      gpuVramGB: 80,
      gpusPerNode: 8,
      numNodes: 1,
      intraNodeStrategy: 'tensor-parallel',
      gpu,
      fabric: FABRIC_SPECS['ethernet-800g'],
      batchSize: 1,
      sequenceLength: 4096,
      quantization: 'fp16',
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: multiGPU.numGPUs,
      numNodes: multiGPU.numNodes,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    // gpu.interconnect is nvlink-4 with no bridge, so the resolved link is "NVLink 4".
    const stats = texts.find((t) => t.text.includes('Interconnect BW:'))
    expect(stats?.text).toContain(`NVLink 4 — ${multiGPU.interconnectBandwidthGBps} GB/s`)
  })

  it('names the override link, not the bridge, when the caller passes the effective GPU', async () => {
    // nvidia-h100-80gb-pcie carries a 2-GPU NVLink bridge natively; a user who picked
    // an interconnect override in InterconnectSelector gets that link's numbers
    // instead (useResultExports applies it before calling exportPptx).
    const bridgedGpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-pcie')
    if (!bridgedGpu) throw new Error('fixture GPU not found')
    const effectiveGpu = applyInterconnectOverride(bridgedGpu, 'pcie-5')

    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })
    const multiGPU = calculateMultiNodeVRAM({
      singleGPU,
      model,
      gpuVramGB: effectiveGpu.vram_gb,
      gpusPerNode: 2,
      numNodes: 1,
      intraNodeStrategy: 'tensor-parallel',
      gpu: effectiveGpu,
      fabric: FABRIC_SPECS['ethernet-800g'],
      batchSize: 1,
      sequenceLength: 4096,
      quantization: 'fp16',
    })

    await exportPptx({
      model,
      gpu: effectiveGpu,
      quantization: 'fp16',
      numGPUs: multiGPU.numGPUs,
      numNodes: multiGPU.numNodes,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    const stats = texts.find((t) => t.text.includes('Interconnect BW:'))
    expect(stats?.text).not.toContain('NVLink bridge')
    expect(stats?.text).toContain(`PCIe 5 — ${multiGPU.interconnectBandwidthGBps} GB/s`)
    expect(multiGPU.interconnectBandwidthGBps).toBe(128)
  })

  it('omits the Servers row for a single-node configuration', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    const configRows = tableRows(0)
    expect(configRows).toContainEqual(['GPUs per replica', '1 (in one server)'])
    expect(configRows.some(([label]) => label === 'Servers')).toBe(false)
  })

  it('notes that GB means GiB on the summary slide', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    expect(
      texts.some((t) => t.text === 'GB here means GiB (1024³ bytes), as nvidia-smi reports.'),
    ).toBe(true)
  })

  it('adds capacity rows to the performance table when provided', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: 42,
      tierSessionsHeld: 100,
      weightSource: 'meta-llama/Llama-3.1-70B',
      concurrentUsers: 1,
      offload: null,
    })

    // Slide 4's table is the third addTable call (slide1 config, slide2
    // breakdown, slide4 performance — slide3 has no table).
    const perfRows = tableRows(2)
    expect(perfRows).toContainEqual(['Max concurrent sessions (at 4,096 tokens)', '42'])
    expect(perfRows).toContainEqual(['Sessions held with KV tier', '100'])
    expect(perfRows).toContainEqual(['Weights', 'measured from meta-llama/Llama-3.1-70B'])
  })

  it('reports expected decode, first-token and prefill with the ceiling beside each (spec P3)', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      concurrentUsers: 1,
      offload: null,
    })

    const { expected } = performance
    if (!expected.prefillSeconds || !performance.prefillSeconds) throw new Error('no prefill')
    const decode = `${expected.tokensPerSecond.toFixed(1)} tokens/sec (ceiling ${performance.tokensPerSecond.toFixed(1)})`
    const ttft = `${formatDuration(expected.timeToFirstToken)} (ceiling ${formatDuration(performance.timeToFirstToken)})`
    const prefill = `${formatDuration(expected.prefillSeconds)} (ceiling ${formatDuration(performance.prefillSeconds)})`
    const perfRows = tableRows(2)
    expect(perfRows).toContainEqual(['Decode Speed', decode])
    expect(perfRows).toContainEqual(['Time to first token', ttft])
    expect(perfRows).toContainEqual(['Prompt Processing', `${prefill} — weight-dominated (linear)`])
    // the metric boxes headline the same figures
    const boxText = texts.map((t) => t.text)
    expect(boxText).toContain(
      `${expected.tokensPerSecond.toFixed(1)} tok/s (ceiling ${performance.tokensPerSecond.toFixed(1)})`,
    )
    expect(boxText).toContain(ttft)
    expect(boxText).toContain(prefill)
  })

  it('omits capacity rows when not provided, and labels weights estimated when null', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    const perfRows = tableRows(2)
    expect(perfRows.some(([label]) => label.startsWith('Max concurrent sessions'))).toBe(false)
    expect(perfRows.some(([label]) => label === 'Sessions held with KV tier')).toBe(false)
    expect(perfRows).toContainEqual(['Weights', 'estimated (no reference checkpoint)'])
  })

  it('keeps every slide heading clear of the content below it', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      gpu,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    // Slide 4's metric cards used to start at y 0.9 while the heading occupied
    // 0.7-1.1, clipping it. The card shapes are the only roundRects in the deck.
    const heading = texts.find((x) => x.text === 'Performance Estimate')
    expect(heading).toBeDefined()
    const headingBottom = (heading?.opts.y ?? 0) + (heading?.opts.h ?? 0)

    const cards = shapes.filter((s) => s.kind === 'roundRect')
    expect(cards.length).toBeGreaterThan(0)
    for (const card of cards) {
      expect(card.opts.y ?? 0).toBeGreaterThanOrEqual(headingBottom)
    }
  })

  it('formats Sequence Length with en-US regardless of the host locale', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: null,
    })

    const configRows = tableRows(0)
    expect(configRows).toContainEqual(['Sequence Length', '4,096 tokens'])
  })

  it('shows a Concurrent users row only when it differs from batchSize', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 4096,
      offload: null,
    })

    const configRows = tableRows(0)
    expect(configRows).toContainEqual(['Concurrent users', '4,096'])
  })

  it('adds Offloading and Offload slowdown rows, and a Host capacity row when exceeded', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: {
        target: 'cpu-ram',
        mode: 'percentage',
        percentage: 100,
        layers: 0,
        kvOffloaded: true,
        offloadedGiB: 92278,
        slowdown: 31.4,
        hostCapacityGB: 2048,
        exceedsHost: true,
      },
    })

    const configRows = tableRows(0)
    expect(configRows).toContainEqual([
      'Offloading',
      'CPU RAM: 100% of weights + KV cache, 92,278 GiB',
    ])
    expect(configRows).toContainEqual(['Offload slowdown', '≈ 31× slower decode'])
    expect(
      configRows.some(([label, value]) => label === 'Host capacity' && value.includes('exceeded')),
    ).toBe(true)
  })

  it('omits the Host capacity row when offload fits, and reports no measurable slowdown below 1.05x', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: {
        target: 'cpu-ram',
        mode: 'percentage',
        percentage: 5,
        layers: 0,
        kvOffloaded: false,
        offloadedGiB: 2,
        slowdown: 1.01,
        hostCapacityGB: 2048,
        exceedsHost: false,
      },
    })

    const configRows = tableRows(0)
    expect(configRows.some(([label]) => label === 'Host capacity')).toBe(false)
    expect(configRows).toContainEqual(['Offload slowdown', 'no measurable slowdown'])
  })

  it('notes offloaded memory on the multi-GPU slide when offload is set', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const multiGPU = calculateMultiNodeVRAM({
      singleGPU,
      model,
      gpuVramGB: 80,
      gpusPerNode: 8,
      numNodes: 1,
      intraNodeStrategy: 'tensor-parallel',
      gpu,
      fabric: FABRIC_SPECS['ethernet-800g'],
      batchSize: 1,
      sequenceLength: 4096,
      quantization: 'fp16',
    })

    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: multiGPU.numGPUs,
      numNodes: multiGPU.numNodes,
      sequenceLength: 4096,
      batchSize: 1,
      vram: singleGPU,
      performance,
      multiGPU,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 1,
      offload: {
        target: 'cpu-ram',
        mode: 'percentage',
        percentage: 50,
        layers: 0,
        kvOffloaded: false,
        offloadedGiB: 500,
        slowdown: 3,
        hostCapacityGB: 2048,
        exceedsHost: false,
      },
    })

    expect(
      texts.some((t) => t.text.includes('Offloaded to host: 500 GiB (not included per GPU)')),
    ).toBe(true)
  })

  it('labels the first-token row as amortized over the batch when B > 1 (ADR 0007)', async () => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 8,
    })
    await exportPptx({
      model,
      gpu,
      quantization: 'fp16',
      numGPUs: 1,
      numNodes: 1,
      sequenceLength: 4096,
      batchSize: 8,
      vram: singleGPU,
      performance,
      multiGPU: null,
      maxSessions: null,
      tierSessionsHeld: null,
      weightSource: null,
      concurrentUsers: 8,
      offload: null,
    })
    const perfRows = tableRows(2)
    expect(
      perfRows.some(([label]) => label === 'Prefill per request (amortized over batch 8)'),
    ).toBe(true)
    expect(perfRows.some(([label]) => label === 'Time to First Token')).toBe(false)
  })
})

describe('exportSizingPptx', () => {
  beforeEach(() => {
    tables.length = 0
    charts.length = 0
    texts.length = 0
  })

  const sizingGpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-gb300-nvl72')
  if (!sizingGpu) throw new Error('fixture not found')
  const size = (kvTier = DEFAULT_KV_TIER) =>
    sizeDeployment({
      model: realModel,
      gpu: sizingGpu,
      quantization: 'fp8',
      kvQuantization: 'fp16',
      sequenceLength: 32768,
      users: 2000,
      fabric: resolveFabricSpec('ethernet-800g', null),
      kvTier,
    })
  const run = (result: ReturnType<typeof size>) =>
    exportSizingPptx({
      model: realModel,
      gpu: sizingGpu,
      quantization: 'fp8',
      kvQuantization: 'fp16',
      fabricLabel: resolveFabricSpec('ethernet-800g', null).label,
      sequenceLength: 32768,
      users: 2000,
      result,
    })

  it('writes one summary slide with headline, replica, replicas, sessions and speed with ceiling, and no memory charts', async () => {
    const r = size()
    const c = r.chosen
    if (!c) throw new Error('expected a fit')
    await run(r)
    expect(charts).toHaveLength(0)
    expect(tables).toHaveLength(1)
    expect(texts[0]?.text.startsWith(`${r.units} rack`)).toBe(true)
    expect(texts[0]?.text).toContain(sizingGpu.name)
    const rows = tableRows(0)
    expect(rows).toContainEqual(['Replica', expect.stringContaining(`TP ${c.tp} × ${c.nodes}`)])
    expect(rows).toContainEqual(['Replicas', String(r.replicas)])
    expect(rows).toContainEqual(['Sessions per replica', String(c.sessionsPerReplica)])
    const speed = rows.find(([k]) => k.startsWith('Per-user decode speed'))
    expect(speed?.[1]).toBe(
      `${c.perUserTokensPerSecond.toFixed(1)} tok/s (ceiling ${c.perUserCeilingTokensPerSecond.toFixed(1)})`,
    )
    // No tier: no activity assumption to state
    expect(rows.some(([k]) => k === 'Assumption')).toBe(false)
  })

  it('states the tier and the active-share assumption when a KV tier raised the count', async () => {
    const r = size({ ...DEFAULT_KV_TIER, tier: 'host-grace' })
    const c = r.chosen
    if (!c?.tierUsed || c.tierUsed === 'none') throw new Error('expected a tier')
    await run(r)
    const rows = tableRows(0)
    expect(rows).toContainEqual(['Assumption', '25% of sessions decode at once'])
    expect(rows.find(([k]) => k === 'Open sessions per replica')?.[1]).toMatch(/parked in/)
  })

  // 2026-11-11, sizing final review M3/I1/I5: the slide gains KV precision, packing rows (rack
  // replicas of one unit), the decode-batch note and the corrected footer; no Fabric row for a rack.
  it('adds KV precision and packing rows, no Fabric row for a rack', async () => {
    const r = size()
    await run(r)
    const rows = tableRows(0)
    expect(rows).toContainEqual(['KV precision', 'FP16'])
    expect(rows).toContainEqual([`Replicas per rack`, String(r.replicasPerUnit)])
    expect(rows).toContainEqual([`Idle GPUs per rack`, String(r.idleGPUsPerUnit)])
    expect(rows.some(([k]) => k === 'Fabric')).toBe(false)
    expect(texts.some((t) => t.text.startsWith('Counts are set by memory (90% vLLM limit)'))).toBe(
      true,
    )
    expect(texts.some((t) => t.text.includes('Unit counts use expected speed'))).toBe(false)
  })

  it('shows the Fabric row for server GPUs and omits packing rows for multi-server replicas', async () => {
    const h100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
    if (!h100) throw new Error('fixture not found')
    const r = sizeDeployment({
      model: realModel,
      gpu: h100,
      quantization: 'fp8',
      kvQuantization: 'fp16',
      sequenceLength: 32768,
      users: 50,
      fabric: resolveFabricSpec('ethernet-800g', null),
      kvTier: DEFAULT_KV_TIER,
    })
    await exportSizingPptx({
      model: realModel,
      gpu: h100,
      quantization: 'fp8',
      kvQuantization: 'fp16',
      fabricLabel: '800GbE (SONiC / RoCEv2)',
      sequenceLength: 32768,
      users: 50,
      result: r,
    })
    const rows = tableRows(0)
    expect(rows).toContainEqual(['Fabric', '800GbE (SONiC / RoCEv2)'])
    expect(rows.some(([k]) => k.startsWith('Replicas per'))).toBe(r.chosen?.nodes === 1)
  })

  it('adds the max-num-seqs note when more than 256 sequences decode at once', async () => {
    const small = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-8b')
    if (!small) throw new Error('fixture not found')
    const r = sizeDeployment({
      model: small,
      gpu: sizingGpu,
      quantization: 'fp8',
      kvQuantization: 'fp16',
      sequenceLength: 512,
      users: 20000,
      fabric: resolveFabricSpec('ethernet-800g', null),
      kvTier: DEFAULT_KV_TIER,
    })
    expect(r.chosen?.decodeBatch).toBeGreaterThan(256)
    await exportSizingPptx({
      model: small,
      gpu: sizingGpu,
      quantization: 'fp8',
      kvQuantization: 'fp16',
      fabricLabel: 'x',
      sequenceLength: 512,
      users: 20000,
      result: r,
    })
    expect(tableRows(0).find(([k]) => k === 'Note')?.[1]).toContain('--max-num-seqs')
  })
})
