import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { GPU_MEMORY_UTILIZATION } from '@engines/constants'
import { FABRIC_SPECS } from '@engines/fabric'
import { calculateInferenceVRAM } from '@engines/inference'
import { calculateMultiGPUVRAM } from '@engines/multi-gpu'
import { calculateMultiNodeVRAM } from '@engines/multi-node'
import { estimatePerformance } from '@engines/performance'
import { render, screen } from '@testing-library/react'
import { type GPU, type Model, validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'

import { MultiGPUBreakdownChart } from './MultiGPUBreakdownChart'

// Real factories, not hand-written Decimal literals — see CLAUDE.md
// "Test files are typechecked by nothing"; fixtures built by calling the
// engine avoid going stale silently.
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

const singleGPU = calculateInferenceVRAM({
  model,
  quantization: 'fp16',
  sequenceLength: 4096,
  batchSize: 1,
})

describe('MultiGPUBreakdownChart', () => {
  it('sums segment widths to the utilization percent when under capacity', () => {
    // 4-way tensor parallel on an 80GB H100 — comfortably under capacity.
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const gpuVRAM = 80
    const totalPerGPU = breakdown.totalPerGPU.toNumber()
    const expectedUtilization = (totalPerGPU / gpuVRAM) * 100
    expect(expectedUtilization).toBeLessThan(100)

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    const segmentValues = [
      breakdown.perGPU.modelWeights.toNumber(),
      breakdown.perGPU.kvCache.toNumber(),
      breakdown.perGPU.activations.toNumber(),
      breakdown.perGPU.frameworkOverhead.toNumber(),
      breakdown.perGPU.communicationOverhead.toNumber(),
    ]
    const segmentTitles = [
      new RegExp(`^Model Weights: ${segmentValues[0]?.toFixed(2)} GB$`),
      new RegExp(`^KV Cache: ${segmentValues[1]?.toFixed(2)} GB$`),
      new RegExp(`^Activations: ${segmentValues[2]?.toFixed(2)} GB$`),
      new RegExp(`^Framework & NCCL: ${segmentValues[3]?.toFixed(2)} GB$`),
      new RegExp(`^Communication: ${segmentValues[4]?.toFixed(2)} GB$`),
    ]

    let widthSum = 0
    for (const titleRe of segmentTitles) {
      const el = screen.getByTitle(titleRe)
      const width = Number.parseFloat(el.style.width)
      expect(Number.isNaN(width)).toBe(false)
      widthSum += width
    }

    expect(widthSum).toBeCloseTo(expectedUtilization, 5)

    // Per-segment values must be reachable via the accessibility tree, not just
    // the `title` attribute — role="img" on an ancestor makes children
    // presentational and hides them from assistive tech, which `getByTitle`
    // (DOM attribute, not a11y tree) would not catch.
    const segmentImages = screen.getAllByRole('img')
    expect(segmentImages).toHaveLength(5)
    const accessibleNames = segmentImages.map((el) => el.getAttribute('aria-label'))
    expect(accessibleNames).toEqual([
      `Model Weights: ${segmentValues[0]?.toFixed(2)} GB`,
      `KV Cache: ${segmentValues[1]?.toFixed(2)} GB`,
      `Activations: ${segmentValues[2]?.toFixed(2)} GB`,
      `Framework & NCCL: ${segmentValues[3]?.toFixed(2)} GB`,
      `Communication: ${segmentValues[4]?.toFixed(2)} GB`,
    ])
  })

  it('shows the headroom figure under capacity', () => {
    // 2026-11-11, final review I2: headroom is to vLLM's 90% limit, not to the full card
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const gpuVRAM = 80
    const headroom = gpuVRAM * GPU_MEMORY_UTILIZATION - breakdown.totalPerGPU.toNumber()
    expect(headroom).toBeGreaterThan(0)

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    expect(
      screen.getByText(`${headroom.toFixed(1)} GB headroom to the 90% limit`),
    ).toBeInTheDocument()
    expect(screen.queryByText(/GB over/)).not.toBeInTheDocument()
  })

  it('saturates the meter at 100% and shows the over-capacity amount when over capacity', () => {
    // Force over-capacity deterministically: compute a real breakdown, then
    // hand the component a smaller gpuVRAM than its totalPerGPU (e.g. the
    // user picked a smaller GPU after the breakdown was computed).
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const totalPerGPU = breakdown.totalPerGPU.toNumber()
    const gpuVRAM = totalPerGPU / 2
    // 2026-11-11, final review I2: over the 90% limit, was over the full card
    const overAmount = totalPerGPU - gpuVRAM * GPU_MEMORY_UTILIZATION

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    expect(screen.getByText(`${overAmount.toFixed(1)} GB over the 90% limit`)).toBeInTheDocument()
    expect(screen.queryByText(/GB headroom/)).not.toBeInTheDocument()

    // Segment widths still sum to 100% (saturated), not the (>100%) raw utilization.
    const segmentValues = [
      breakdown.perGPU.modelWeights.toNumber(),
      breakdown.perGPU.kvCache.toNumber(),
      breakdown.perGPU.activations.toNumber(),
      breakdown.perGPU.frameworkOverhead.toNumber(),
      breakdown.perGPU.communicationOverhead.toNumber(),
    ]
    const segmentTitles = [
      new RegExp(`^Model Weights: ${segmentValues[0]?.toFixed(2)} GB$`),
      new RegExp(`^KV Cache: ${segmentValues[1]?.toFixed(2)} GB$`),
      new RegExp(`^Activations: ${segmentValues[2]?.toFixed(2)} GB$`),
      new RegExp(`^Framework & NCCL: ${segmentValues[3]?.toFixed(2)} GB$`),
      new RegExp(`^Communication: ${segmentValues[4]?.toFixed(2)} GB$`),
    ]

    let widthSum = 0
    const overCapacityColor = 'rgb(220, 38, 38)'
    for (const titleRe of segmentTitles) {
      const el = screen.getByTitle(titleRe)
      widthSum += Number.parseFloat(el.style.width)
      expect(el.style.backgroundColor).toBe(overCapacityColor)
    }
    expect(widthSum).toBeCloseTo(100, 5)

    // The legend must describe the bar it sits under. With every segment painted
    // one color, a five-color legend would name swatches that appear nowhere.
    const swatches = document.querySelectorAll('span.inline-block.rounded-sm')
    expect(swatches).toHaveLength(5)
    for (const swatch of swatches) {
      expect((swatch as HTMLElement).style.backgroundColor).toBe(overCapacityColor)
    }
  })

  it('suppresses the per-node multiplier on a single node, whatever the GPU count', () => {
    const singleGpuBreakdown = calculateMultiGPUVRAM(
      singleGPU,
      model,
      80,
      1,
      'tensor-parallel',
      gpu,
    )
    const { unmount } = render(
      <MultiGPUBreakdownChart breakdown={singleGpuBreakdown} gpuVRAM={80} />,
    )
    expect(screen.getByText(/identical across all 1 GPU/)).toBeInTheDocument()
    expect(screen.queryByText(/per node/)).not.toBeInTheDocument()
    unmount()

    // 4 GPUs in one node: the count is already stated, so "4 per node x 1 node"
    // would only restate it. The split appears once numNodes > 1 (see below).
    const multiGpuBreakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    render(<MultiGPUBreakdownChart breakdown={multiGpuBreakdown} gpuVRAM={80} />)
    expect(screen.getByText('identical across all 4 GPUs')).toBeInTheDocument()
    expect(screen.queryByText(/per node/)).not.toBeInTheDocument()
  })

  it('reports the per-node split (not just the total) across multiple servers', () => {
    // 4 servers x 8 GPUs/server = 32 GPUs total. numGPUs on the breakdown is
    // the TOTAL (per CLAUDE.md); gpusPerNode/numNodes must come through
    // separately so the footer can show "8 per node x 4 nodes", not e.g.
    // "32 per node x 1 node" from a swapped field.
    const multiNodeBreakdown = calculateMultiNodeVRAM({
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
    expect(multiNodeBreakdown.numGPUs).toBe(32)

    render(<MultiGPUBreakdownChart breakdown={multiNodeBreakdown} gpuVRAM={80} />)

    expect(
      screen.getByText('identical across all 32 GPUs (8 per node × 4 nodes)'),
    ).toBeInTheDocument()

    // The footer above owns the counts; the strategy line must not restate
    // them (it used to read "4 servers x 8 GPUs = 32 GPUs - ...").
    expect(
      screen.getByText('Tensor parallel within each server, pipeline parallel across them.'),
    ).toBeInTheDocument()
    // 100 GB/s per port x 8 GPUs x eta 0.8 (HGX assumption, fabric.ts)
    expect(screen.getByText(/640 GB\/s effective between servers/)).toBeInTheDocument()
    expect(screen.queryByText(/4 servers/)).not.toBeInTheDocument()
  })

  it('states capacity, usage and percent in the header line', () => {
    // The header is the design's headline deliverable ("Per GPU - 156.3 / 180
    // GB - 87% used"). Without this, a completely mangled header ships green.
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const gpuVRAM = 80
    const totalPerGPU = breakdown.totalPerGPU.toNumber()
    const percentUsed = Math.round((totalPerGPU / gpuVRAM) * 100)

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    expect(
      screen.getByText(
        `Per GPU \u2014 ${totalPerGPU.toFixed(1)} / ${gpuVRAM} GB \u2014 ${percentUsed}% used`,
      ),
    ).toBeInTheDocument()
  })

  it('reads as over the limit at a full card, without renormalizing the widths', () => {
    // 2026-11-11, final review I2: was "0.0 GB headroom" at usage = capacity; the limit is
    // now vLLM's 90%, so a full card is 10% over it. Widths stay a share of the card (100%).
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const gpuVRAM = breakdown.totalPerGPU.toNumber()

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    const over = gpuVRAM - gpuVRAM * GPU_MEMORY_UTILIZATION
    expect(screen.getByText(`${over.toFixed(1)} GB over the 90% limit`)).toBeInTheDocument()
    expect(screen.queryByText(/GB headroom/)).not.toBeInTheDocument()
  })

  it('between 90% and 100% of the card: over-limit colours, no headroom, 90% marker (I2)', () => {
    // 2026-11-11, final review I2: 76 GB per GPU on an 80 GB card read as normal colours with
    // 4.0 GB headroom while the verdict said "Does Not Fit". Same 95% ratio, real breakdown.
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const totalPerGPU = breakdown.totalPerGPU.toNumber()
    const gpuVRAM = totalPerGPU / 0.95

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    const over = totalPerGPU - gpuVRAM * GPU_MEMORY_UTILIZATION
    expect(screen.getByText(`${over.toFixed(1)} GB over the 90% limit`)).toBeInTheDocument()
    expect(screen.queryByText(/GB headroom/)).not.toBeInTheDocument()
    let widthSum = 0
    for (const el of screen.getAllByRole('img')) {
      expect(el.style.backgroundColor).toBe('rgb(220, 38, 38)')
      widthSum += Number.parseFloat(el.style.width)
    }
    // Under the card the widths stay a share of it, so the marker sits at 90% of the bar
    expect(widthSum).toBeCloseTo(95, 5)
    expect(Number.parseFloat(screen.getByTestId('limit-marker').style.left)).toBeCloseTo(90, 9)
  })

  it('marks the 90% limit relative to the bar when over the card', () => {
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const gpuVRAM = breakdown.totalPerGPU.toNumber() / 2
    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)
    // Widths renormalize to the total (2x the card), so 90% of the card is 45% of the bar
    expect(Number.parseFloat(screen.getByTestId('limit-marker').style.left)).toBeCloseTo(45, 5)
  })

  it('shows expected throughput with the theoretical ceiling beside it (spec P3)', () => {
    const realModel = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
    const realGpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
    if (!realModel || !realGpu) throw new Error('fixture not found')
    const vram = calculateInferenceVRAM({
      model: realModel,
      quantization: 'fp8',
      sequenceLength: 4096,
      batchSize: 1,
    })
    const breakdown = calculateMultiGPUVRAM(vram, realModel, 80, 2, 'tensor-parallel', realGpu)
    const perf = estimatePerformance({
      model: realModel,
      gpu: realGpu,
      quantization: 'fp8',
      sequenceLength: 4096,
      batchSize: 1,
      multiGPUResult: breakdown,
    })
    const expected = perf.expected.tokensPerSecond.toFixed(0)
    const ceiling = perf.tokensPerSecond.toFixed(0)
    expect(expected).not.toBe(ceiling)

    render(
      <MultiGPUBreakdownChart
        breakdown={breakdown}
        gpuVRAM={80}
        effectiveTokPerSec={perf.expected.tokensPerSecond.toNumber()}
        ceilingTokPerSec={perf.tokensPerSecond.toNumber()}
      />,
    )
    expect(screen.getByText(`~${expected} tok/s`, { exact: false })).toBeInTheDocument()
    expect(
      screen.getByText(`Theoretical ceiling: ~${ceiling} tok/s`, { exact: false }),
    ).toBeInTheDocument()
  })
})
