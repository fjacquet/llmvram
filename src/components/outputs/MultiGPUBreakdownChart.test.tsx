import { calculateInferenceVRAM } from '@engines/inference'
import { calculateMultiGPUVRAM } from '@engines/multi-gpu'
import { render, screen } from '@testing-library/react'
import type { GPU, Model } from '@utils/schemas'
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
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const gpuVRAM = 80
    const headroom = gpuVRAM - breakdown.totalPerGPU.toNumber()
    expect(headroom).toBeGreaterThan(0)

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    expect(screen.getByText(`${headroom.toFixed(1)} GB headroom`)).toBeInTheDocument()
    expect(screen.queryByText(/GB over/)).not.toBeInTheDocument()
  })

  it('saturates the meter at 100% and shows the over-capacity amount when over capacity', () => {
    // Force over-capacity deterministically: compute a real breakdown, then
    // hand the component a smaller gpuVRAM than its totalPerGPU (e.g. the
    // user picked a smaller GPU after the breakdown was computed).
    const breakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    const totalPerGPU = breakdown.totalPerGPU.toNumber()
    const gpuVRAM = totalPerGPU / 2
    const overAmount = totalPerGPU - gpuVRAM

    render(<MultiGPUBreakdownChart breakdown={breakdown} gpuVRAM={gpuVRAM} />)

    expect(screen.getByText(`${overAmount.toFixed(1)} GB over`)).toBeInTheDocument()
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
  })

  it('suppresses the per-node multiplier at N=1 but shows it for N>1', () => {
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

    const multiGpuBreakdown = calculateMultiGPUVRAM(singleGPU, model, 80, 4, 'tensor-parallel', gpu)
    render(<MultiGPUBreakdownChart breakdown={multiGpuBreakdown} gpuVRAM={80} />)
    expect(screen.getByText(/identical across all 4 GPUs/)).toBeInTheDocument()
    expect(screen.getByText(/4 per node × 1 node/)).toBeInTheDocument()
  })
})
