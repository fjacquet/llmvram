import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { estimatePerformance } from '@engines/performance'
import { classPrefills } from '@engines/ttft'
import { render, screen } from '@testing-library/react'
import { formatDuration } from '@utils/formatDuration'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { VerdictBlock } from './VerdictBlock'

const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.1-70b')
const gpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
if (!model || !gpu) throw new Error('fixture not found')
const performance = estimatePerformance({
  model,
  gpu,
  quantization: 'fp8',
  sequenceLength: 4096,
  batchSize: 1,
})

const one = classPrefills({ model, gpu, quantization: 'fp8' }, [{ tokens: 4096, share: 1 }])
const mixed = classPrefills({ model, gpu, quantization: 'fp8' }, [
  { tokens: 4096, share: 0.9 },
  { tokens: 262144, share: 0.1 },
])

describe('VerdictBlock (spec P3)', () => {
  const renderVerdict = (over: Partial<React.ComponentProps<typeof VerdictBlock>> = {}) =>
    render(
      <VerdictBlock
        fit={null}
        performance={performance}
        batchSize={1}
        maxSessions={10}
        sequenceLength={4096}
        classes={one}
        concurrentUsers={1}
        ttftTargetSeconds={null}
        largestClassMiss={null}
        {...over}
      />,
    )

  it('keeps the plain sessions label and shows no target or guard by default', () => {
    renderVerdict()
    expect(screen.getByText('Max sessions at 4,096 tokens')).toBeInTheDocument()
    expect(screen.queryByText(/Target/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Does not fit/)).not.toBeInTheDocument()
  })

  it('names the mix in the sessions label when it is not the default', () => {
    renderVerdict({ classes: mixed, sequenceLength: 262144 })
    expect(screen.getByText('Max sessions (mix: 90% 4K, 10% 256K)')).toBeInTheDocument()
  })

  it('says does not fit, naming the largest class, when one session of it cannot fit', () => {
    renderVerdict({ classes: mixed, sequenceLength: 262144, largestClassMiss: 262144 })
    expect(screen.getByText(/Does not fit: one session at the largest class/)).toHaveTextContent(
      '256K tokens',
    )
  })

  it('reports the target against the last served', () => {
    renderVerdict({ concurrentUsers: 100, ttftTargetSeconds: 0.001 })
    expect(screen.getByText(/^Target .*: missed by /)).toBeInTheDocument()
    renderVerdict({ ttftTargetSeconds: 3600 })
    expect(screen.getByText(/^Target 1\.0 h: met$/)).toBeInTheDocument()
  })

  it('headlines expected tok/s with the theoretical ceiling beside it', () => {
    renderVerdict()
    const expected = `${performance.expected.tokensPerSecond.toFixed(1)} tokens/sec`
    const ceiling = `${performance.tokensPerSecond.toFixed(1)} tokens/sec`
    expect(expected).not.toBe(ceiling)
    expect(screen.getByText(expected)).toBeInTheDocument()
    expect(screen.getByText(`Theoretical ceiling: ${ceiling}`)).toBeInTheDocument()
  })

  it('headlines expected TTFT with the theoretical ceiling beside it', () => {
    renderVerdict()
    const expected = formatDuration(performance.expected.timeToFirstToken)
    const ceiling = formatDuration(performance.timeToFirstToken)
    expect(expected).not.toBe(ceiling)
    expect(screen.getByText(expected)).toBeInTheDocument()
    expect(screen.getByText(`Theoretical ceiling: ${ceiling}`)).toBeInTheDocument()
  })
})
