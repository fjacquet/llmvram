import { FABRIC_SPECS } from '@engines/fabric'
import { calculateInferenceVRAM } from '@engines/inference'
import { calculateMultiNodeVRAM } from '@engines/multi-node'
import type { PerformanceEstimate } from '@engines/types'
import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// pptxgenjs drives real browser/Node file I/O (pptx.writeFile) and canvas-ish
// chart rendering — none of which is relevant to IMPORTANT-1 (the deck must
// report the cluster TOTAL GPU count, not the per-node count). We mock the
// library and assert on the arguments exportPptx passes it instead.
interface RecordedTable {
  rows: unknown[]
}
interface RecordedChart {
  type: string
  data: Array<{ name: string; labels: string[]; values: number[] }>
}

const tables: RecordedTable[] = []
const charts: RecordedChart[] = []

class MockSlide {
  addText = vi.fn()
  addShape = vi.fn()
  addTable = vi.fn((rows: unknown[]) => {
    tables.push({ rows })
  })
  addChart = vi.fn(
    (type: string, data: Array<{ name: string; labels: string[]; values: number[] }>) => {
      charts.push({ type, data })
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

vi.mock('pptxgenjs', () => ({ default: MockPptxGenJS }))

const { exportPptx } = await import('./exportPptx')

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

const performance: PerformanceEstimate = {
  tokensPerSecond: new Decimal(42),
  timeToFirstToken: new Decimal(0.5),
  prefillSeconds: new Decimal(0.2),
  prefillBottleneck: 'linear',
  prefillEstimateDegraded: false,
  isComputeBound: false,
  isMemoryBound: true,
  bottleneck: 'memory',
}

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
    })

    // Slide 1 (config summary) is the first addTable call.
    const configRows = tableRows(0)
    expect(configRows).toContainEqual(['Number of GPUs', '32'])
    expect(configRows).not.toContainEqual(['Number of GPUs', '8'])
    expect(configRows).toContainEqual(['Servers', '4'])

    // Slide 3's bar chart must have one bar per TOTAL GPU (32), not per-node (8).
    const barChart = charts.find((c) => c.type === 'bar')
    expect(barChart).toBeDefined()
    expect(barChart?.data[0]?.values).toHaveLength(32)
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
    })

    const configRows = tableRows(0)
    expect(configRows).toContainEqual(['Number of GPUs', '1'])
    expect(configRows.some(([label]) => label === 'Servers')).toBe(false)
  })
})
