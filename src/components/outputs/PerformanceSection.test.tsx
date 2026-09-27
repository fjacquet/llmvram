import type { PerformanceEstimate } from '@engines/types'
import { render, screen } from '@testing-library/react'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { PerformanceSection } from './PerformanceSection'

const performance: PerformanceEstimate = {
  tokensPerSecond: new Decimal(100),
  timeToFirstToken: new Decimal(0.5),
  prefillSeconds: new Decimal(0.2),
  prefillBottleneck: 'linear',
  prefillEstimateDegraded: false,
  isComputeBound: false,
  isMemoryBound: true,
  bottleneck: 'memory',
}

describe('PerformanceSection', () => {
  it('renders the aggregate metrics', () => {
    render(<PerformanceSection performance={performance} concurrentUsers={1} batchSize={1} />)
    expect(screen.getByText('100.0 tokens/sec')).toBeInTheDocument()
  })

  it('shows a queueing warning when batch size is below concurrent users', () => {
    render(<PerformanceSection performance={performance} concurrentUsers={5} batchSize={2} />)
    expect(
      screen.getByText(
        'Only 2 of 5 users decode at once (batch size 2); the rest queue, hence the long TTFT. Raise batch size to serve them concurrently.',
      ),
    ).toBeInTheDocument()
  })

  it('omits the warning when batch size covers every concurrent user', () => {
    render(<PerformanceSection performance={performance} concurrentUsers={4} batchSize={4} />)
    expect(screen.queryByText(/decode at once/)).not.toBeInTheDocument()
  })

  it('omits the multi-user block entirely for a single user', () => {
    render(<PerformanceSection performance={performance} concurrentUsers={1} batchSize={1} />)
    expect(screen.queryByText(/decode at once/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Multi-user metrics/)).not.toBeInTheDocument()
  })
})
