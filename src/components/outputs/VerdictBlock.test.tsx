import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { estimatePerformance } from '@engines/performance'
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

describe('VerdictBlock (spec P3)', () => {
  const renderVerdict = () =>
    render(
      <VerdictBlock
        fit={null}
        performance={performance}
        batchSize={1}
        maxSessions={10}
        sequenceLength={4096}
      />,
    )

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
