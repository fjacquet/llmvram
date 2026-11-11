import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { fitsInference } from '@engines/concurrency'
import { isValidTPDegree } from '@engines/config-rules'
import { calculateInferenceVRAM } from '@engines/inference'
import { calculateMultiGPUVRAM } from '@engines/multi-gpu'
import type { InferenceVRAMBreakdown } from '@engines/types'
import { render, screen } from '@testing-library/react'
import { type GPU, validateGPUs, validateModels } from '@utils/schemas'
import Decimal from 'decimal.js'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// The real uiStore wraps its state in zustand's `persist` middleware, which throws in
// jsdom (no localStorage backing). Build a plain store with the same shape instead —
// the established pattern in this repo (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')

  interface MockState {
    offloadingEnabled: boolean
  }

  const useUIStore = create<MockState>(() => ({
    offloadingEnabled: false,
  }))

  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

// Import after mock setup
import { Recommendations } from './Recommendations'

const anyModel = validateModels(modelsData)[0]
if (!anyModel) throw new Error('models.json is empty')

function gpu(overrides: Partial<GPU>): GPU {
  return {
    id: 'test-gpu',
    name: 'Test GPU',
    manufacturer: 'apple',
    vram_gb: 128,
    memory_bandwidth_gbps: 800,
    memory_type: 'Unified',
    bus_width: 0,
    max_gpus_per_node: 1,
    tier: 'apple-silicon',
    ...overrides,
  }
}

function breakdown(totalGB: number): InferenceVRAMBreakdown {
  return {
    modelWeights: new Decimal(totalGB * 0.9),
    kvCache: new Decimal(totalGB * 0.05),
    activations: new Decimal(totalGB * 0.03),
    frameworkOverhead: new Decimal(totalGB * 0.02),
    total: new Decimal(totalGB),
  }
}

describe('Recommendations', () => {
  beforeEach(() => {
    useUIStore.setState({ offloadingEnabled: false })
  })

  it('suppresses the multi-GPU recommendation for a part with max_gpus_per_node=1', () => {
    const appleGPU = gpu({ max_gpus_per_node: 1, vram_gb: 128 })
    render(
      <Recommendations
        gpu={appleGPU}
        model={anyModel}
        breakdown={breakdown(900)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={1}
        multiGPUBreakdown={null}
      />,
    )

    expect(screen.queryByText(/with tensor parallelism/)).not.toBeInTheDocument()
  })

  it('clamps the suggested GPU count to max_gpus_per_node instead of an unbuildable count', () => {
    const gb10Like = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 2,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={gb10Like}
        model={anyModel}
        breakdown={breakdown(100)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={1}
        multiGPUBreakdown={null}
      />,
    )

    // Unclamped math ((100 - 2) / (16 x 0.9 - 2 - 0.25)) would ask for 9x — unbuildable on a
    // 2-way part. 2026-11-11, final review I1: was 8x at 100% of the card.
    expect(screen.queryByText(/9x Test GPU/)).not.toBeInTheDocument()
    expect(screen.getByText(/2x Test GPU/)).toBeInTheDocument()
  })

  it('clamps the "add more GPUs" recommendation when already multi-GPU', () => {
    const eightWay = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={eightWay}
        model={anyModel}
        breakdown={breakdown(500)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={4}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(200),
            numGPUs: 4,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    // Unclamped math ((500 - 10) / (16 x 0.9 - 10 - 0.25)) would ask for 119x — clamp to the
    // 8-way bound. 2026-11-11, final review I1: was 86x at 100% of the card.
    expect(screen.queryByText(/119x Test GPU/)).not.toBeInTheDocument()
    expect(screen.getByText(/8x Test GPU/)).toBeInTheDocument()
  })

  it('never advises fewer GPUs than are already configured across a cluster', () => {
    // 8 GPUs/node x 4 nodes = 32 total. numGPUs is the CLUSTER TOTAL (see
    // ResultsPanel), but the suggested count was clamped to the PER-NODE bound
    // of 8, so the panel advised cutting a 32-GPU cluster down to 8.
    const eightWay = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={eightWay}
        model={anyModel}
        breakdown={breakdown(500)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={32}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(200),
            numGPUs: 32,
            numNodes: 4,
            gpusPerNode: 8,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    expect(screen.queryByText(/Try 8x Test GPU/)).not.toBeInTheDocument()
  })

  it('suppresses the no-op suggestion when a single node is already at its bound', () => {
    // 8 GPUs on an 8-way part: the clamp can only land back on 8, which used to
    // render as "Current 8x ... Try 8x" — advice to change nothing.
    const eightWay = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 16,
    })
    render(
      <Recommendations
        gpu={eightWay}
        model={anyModel}
        breakdown={breakdown(500)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={8}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(200),
            numGPUs: 8,
            numNodes: 1,
            gpusPerNode: 8,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    expect(screen.queryByText(/with tensor parallelism/)).not.toBeInTheDocument()
  })

  it('advises more servers when every configured node is already full', () => {
    // 8-way part, 8 GPUs/node x 2 nodes = 16 total, model needs far more. GPUs
    // per node is maxed, so node count is the only free variable. Previously
    // this rendered "Need N GB more VRAM. Try:" above an empty list, because the
    // "upgrade GPU" fallback only fires below 80 GB.
    const bigPart = gpu({
      manufacturer: 'nvidia',
      tier: 'datacenter',
      max_gpus_per_node: 8,
      vram_gb: 288,
      name: 'Test GB300',
    })
    render(
      <Recommendations
        gpu={bigPart}
        model={anyModel}
        breakdown={breakdown(12000)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={16}
        multiGPUBreakdown={
          {
            totalPerGPU: new Decimal(750),
            numGPUs: 16,
            numNodes: 2,
            gpusPerNode: 8,
            scalingEfficiency: 0.85,
          } as never
        }
      />,
    )

    expect(screen.getByText(/Add more servers/)).toBeInTheDocument()
    // and still never advises shrinking
    expect(screen.queryByText(/Try 16x Test GB300/)).not.toBeInTheDocument()
  })

  it('sizes the GPU count by memory alone, not by interconnect efficiency', () => {
    // 2026-11-11, final review I1: 150 GB -> 130 GB, capacity is now 90% of the card, and at
    // 150 GB memory alone needs 3x. 130 GB (2.6 GB framework) on 80 GB GPUs: each GPU holds its
    // own framework context and 0.25 GB of NCCL buffers, so (130 - 2.6) / (72 - 2.6 - 0.25) =
    // 1.84, i.e. 2x. Derating by a scaling efficiency (130 / (72 * 0.85) = 2.1) asks for 3x.
    const pcie = gpu({
      manufacturer: 'nvidia',
      tier: 'consumer',
      max_gpus_per_node: 8,
      vram_gb: 80,
    })
    render(
      <Recommendations
        gpu={pcie}
        model={anyModel}
        breakdown={breakdown(130)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
        numGPUs={1}
        multiGPUBreakdown={null}
      />,
    )
    expect(screen.getByText(/2x Test GPU/)).toBeInTheDocument()
    expect(screen.queryByText(/3x Test GPU/)).not.toBeInTheDocument()
  })

  it('advises for a config at 95% of the card: deficit is measured against the 90% limit', () => {
    // 2026-11-11: fit at vLLM 0.9 (audit M4)
    const { container } = render(
      <Recommendations
        gpu={gpu({ vram_gb: 80, max_gpus_per_node: 8 })}
        model={anyModel}
        breakdown={breakdown(76)}
        currentQuantization="fp16"
        currentSequenceLength={4096}
      />,
    )
    expect(container).not.toBeEmptyDOMElement()
    // 2026-11-11, final review I1: exact text; the old regex also matched "24 GB"
    expect(screen.getByText(/Need 4\.00 GB more VRAM/)).toBeInTheDocument()
  })

  it('advises an upgrade only to a card the total fits at 90% of', () => {
    // 2026-11-11, final review I1: 30 GB fits a 32 GB card at 100% (old rule) but not at 90%
    render(
      <Recommendations
        gpu={gpu({ vram_gb: 16, max_gpus_per_node: 1 })}
        model={anyModel}
        breakdown={breakdown(30)}
        currentQuantization="fp16"
        currentSequenceLength={2048}
      />,
    )
    expect(screen.queryByText(/RTX 5090/)).not.toBeInTheDocument()
    expect(screen.getByText(/A6000 \(48 GB\)/)).toBeInTheDocument()
  })

  describe('advised GPU count passes the inference fit rule (final review I1)', () => {
    // 2026-11-11: sized at 100% of the card, the advice landed between 90% and 100% and the
    // verdict then read "Does Not Fit" on the very configuration it advised
    const h100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
    if (!h100) throw new Error('nvidia-h100-80gb-sxm missing from gpus.json')
    const advise = (modelId: string, sequenceLength: number) => {
      const model = validateModels(modelsData).find((m) => m.id === modelId)
      if (!model) throw new Error(`${modelId} missing from models.json`)
      const vram = calculateInferenceVRAM({
        model,
        quantization: 'fp16',
        sequenceLength,
        batchSize: 1,
      })
      render(
        <Recommendations
          gpu={h100}
          model={model}
          breakdown={vram}
          currentQuantization="fp16"
          currentSequenceLength={sequenceLength}
          numGPUs={1}
          multiGPUBreakdown={null}
        />,
      )
      const text = screen.getByText(/with tensor parallelism/).textContent ?? ''
      const n = Number(text.match(/Use (\d+)x/)?.[1])
      const multi = calculateMultiGPUVRAM(vram, model, 80, n, 'tensor-parallel', h100, 'fp16')
      expect(isValidTPDegree(model, n)).toBe(true)
      return { n, perGPU: multi.totalPerGPU.toNumber() }
    }

    it('DeepSeek R1 Distill 70B fp16, 32k, H100 80 GB', () => {
      const { n, perGPU } = advise('deepseek-r1-distill-llama-70b', 32768)
      // 2026-11-11, sizing T3: was 3 (memory count), now 4 because 64 heads are not divisible by 3
      expect(n).toBe(4)
      expect(fitsInference(perGPU, 80)).toBe(true)
    })

    it('Devstral 2 123B fp16, 8k, H100 80 GB', () => {
      const { perGPU } = advise('mistralai-devstral-2-123b', 8192)
      expect(fitsInference(perGPU, 80)).toBe(true)
    })
  })

  it('past one server, advises more servers, never a full server alone (re-review M1)', () => {
    // 2026-11-11, sizing re-review: was "Use 8x" for a model needing ~12 GPUs by memory,
    // which under-provisions; one server is not enough, so the advice names servers
    const h100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
    const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-405b')
    if (!h100 || !model) throw new Error('fixture missing from data')
    const vram = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 8192,
      batchSize: 1,
    })
    render(
      <Recommendations
        gpu={h100}
        model={model}
        breakdown={vram}
        currentQuantization="fp16"
        currentSequenceLength={8192}
        numGPUs={1}
        multiGPUBreakdown={null}
      />,
    )
    expect(screen.getByText('Add more servers')).toBeInTheDocument()
    const text = screen.getByText(/more than one server holds/).textContent ?? ''
    const servers = Number(text.match(/Try (\d+) servers of 8x/)?.[1])
    expect(servers).toBeGreaterThanOrEqual(2)
    expect(screen.queryByText(/^Use 8x/)).not.toBeInTheDocument()
  })

  describe('never advises a TP degree vLLM refuses (final review M1)', () => {
    // 2026-11-11, sizing final review M1: was "Use 6x" (the raw memory count) when no valid
    // degree inside one server reaches it; GLM 4.7 Flash accepts only 1, 2, 4 and 5 here.
    const h100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
    const flash = validateModels(modelsData).find((m) => m.id === 'zai-org-glm-4.7-flash')
    if (!h100 || !flash) throw new Error('fixture not found')

    it.each([6, 7, 8])('single GPU, memory count %i: no "Use Nx" advice', (count) => {
      // total sized so ceil((total - framework) / (72 - framework - 0.25)) lands on `count`
      const total = (count - 0.5) * 64
      render(
        <Recommendations
          gpu={h100}
          model={flash}
          breakdown={breakdown(total)}
          currentQuantization="fp16"
          currentSequenceLength={2048}
          numGPUs={1}
          multiGPUBreakdown={null}
        />,
      )
      expect(screen.queryByText(new RegExp(`Use ${count}x`))).not.toBeInTheDocument()
      expect(screen.queryByText(/Use [3678]x/)).not.toBeInTheDocument()
    })

    it('already multi-GPU: never advises 6x either', () => {
      const vram = breakdown(6 * 64)
      const multi = calculateMultiGPUVRAM(vram, flash, 80, 2, 'tensor-parallel', h100, 'fp16')
      render(
        <Recommendations
          gpu={h100}
          model={flash}
          breakdown={vram}
          currentQuantization="fp16"
          currentSequenceLength={2048}
          numGPUs={2}
          multiGPUBreakdown={multi}
        />,
      )
      expect(screen.queryByText(/Try [3678]x/)).not.toBeInTheDocument()
    })
  })
})
