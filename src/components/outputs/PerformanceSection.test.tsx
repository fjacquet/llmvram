import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { estimatePerformance } from '@engines/performance'
import { classPrefills } from '@engines/ttft'
import { render, screen } from '@testing-library/react'
import { formatDuration } from '@utils/formatDuration'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { PerformanceSection } from './PerformanceSection'

const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
const gpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
if (!model || !gpu) throw new Error('fixture not found')
// 2026-11-11: built from the real engine (Task 6 added `expected`; the hand literal went stale)
const performance = estimatePerformance({
  model,
  gpu,
  quantization: 'fp8',
  sequenceLength: 4096,
  batchSize: 1,
})

const classes = classPrefills({ model, gpu, quantization: 'fp8' }, [{ tokens: 4096, share: 1 }])

const ui = (
  props: { concurrentUsers: number; batchSize: number; target?: number | null },
  child?: React.ReactNode,
) => (
  <PerformanceSection
    performance={performance}
    concurrentUsers={props.concurrentUsers}
    batchSize={props.batchSize}
    classes={classes}
    ttftTargetSeconds={props.target ?? null}
  >
    {child}
  </PerformanceSection>
)

describe('PerformanceSection', () => {
  it('renders the aggregate metrics', () => {
    render(ui({ concurrentUsers: 1, batchSize: 1 }))
    expect(
      screen.getByText(`${performance.expected.tokensPerSecond.toFixed(1)} tokens/sec`),
    ).toBeInTheDocument()
  })

  it('headlines expected decode, TTFT and prefill with each ceiling beside it (spec P3)', () => {
    render(ui({ concurrentUsers: 1, batchSize: 1 }))
    const { expected } = performance
    if (!expected.prefillSeconds || !performance.prefillSeconds) throw new Error('no prefill')
    const pairs = [
      [
        `${expected.tokensPerSecond.toFixed(1)} tokens/sec`,
        `${performance.tokensPerSecond.toFixed(1)} tokens/sec`,
      ],
      [formatDuration(expected.timeToFirstToken), formatDuration(performance.timeToFirstToken)],
      [formatDuration(expected.prefillSeconds), formatDuration(performance.prefillSeconds)],
    ]
    for (const [headline, ceiling] of pairs) {
      expect(headline).not.toBe(ceiling)
      expect(screen.getByText(headline ?? '')).toBeInTheDocument()
      expect(screen.getByText(`Theoretical ceiling: ${ceiling}`)).toBeInTheDocument()
    }
  })

  it('derives per-user speed and TTFT from the expected figures (spec P2)', () => {
    render(ui({ concurrentUsers: 4, batchSize: 2 }))
    const { expected } = performance
    expect(
      screen.getByText(`${expected.tokensPerSecond.div(4).toFixed(1)} tok/s`),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        `Theoretical ceiling: ${performance.tokensPerSecond.div(4).toFixed(1)} tok/s`,
      ),
    ).toBeInTheDocument()
  })

  it('writes the TTFT line and the worst-case label for a burst (no queueing copy)', () => {
    render(ui({ concurrentUsers: 5, batchSize: 2 }))
    const { expected } = performance
    expect(
      screen.getByText(
        `TTFT ${formatDuration(expected.timeToFirstToken)} first, ${formatDuration(expected.timeToFirstToken.mul(5))} last (5 users at once)`,
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        'worst case: all 5 prompts at once, first come first served, no prefix cache',
      ),
    ).toBeInTheDocument()
    expect(screen.queryByText(/queue/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Per-user TTFT/)).not.toBeInTheDocument()
  })

  it('shows Max users within target only with a target, 0 when one prompt already misses', () => {
    const { rerender } = render(ui({ concurrentUsers: 5, batchSize: 5 }))
    expect(screen.queryByText(/Max users within target/)).not.toBeInTheDocument()
    rerender(ui({ concurrentUsers: 5, batchSize: 5, target: 0.0001 }))
    expect(screen.getByText('Max users within target: 0')).toBeInTheDocument()
  })

  it('omits the multi-user block entirely for a single user', () => {
    render(ui({ concurrentUsers: 1, batchSize: 1 }))
    expect(screen.queryByText(/decode at once/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Multi-user metrics/)).not.toBeInTheDocument()
  })

  it('renders children between the metric grid and the multi-user block (CapacitySection slot)', () => {
    const { container } = render(
      ui({ concurrentUsers: 2, batchSize: 2 }, <div>CAPACITY MARKER</div>),
    )
    const text = container.textContent ?? ''
    const gridIndex = text.indexOf('Decode Speed')
    const markerIndex = text.indexOf('CAPACITY MARKER')
    const multiUserIndex = text.indexOf('Multi-user metrics')
    expect(gridIndex).toBeGreaterThanOrEqual(0)
    expect(markerIndex).toBeGreaterThan(gridIndex)
    expect(multiUserIndex).toBeGreaterThan(markerIndex)
  })

  it('labels the first-token figure as amortized over the batch when B > 1', () => {
    render(ui({ concurrentUsers: 8, batchSize: 8 }))
    expect(screen.getByText('Prefill per request (amortized over batch 8)')).toBeInTheDocument()
    expect(screen.queryByText(/Time to First Token/i)).not.toBeInTheDocument()
  })
})
