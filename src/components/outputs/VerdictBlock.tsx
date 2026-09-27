import type { PerformanceEstimate } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import { firstTokenLabel } from '@utils/perfLabels'
import type { ReactNode } from 'react'

interface VerdictBlockProps {
  /** The fit gauge (FitIndicator) for the displayed breakdown */
  fit: ReactNode
  performance: PerformanceEstimate
  batchSize: number
  maxSessions: number | null
  sequenceLength: number
}

/** The answer first: does it fit, how fast, first-token delay, sessions (ADR 0005) */
export function VerdictBlock({
  fit,
  performance,
  batchSize,
  maxSessions,
  sequenceLength,
}: VerdictBlockProps) {
  return (
    <section data-testid="verdict" aria-label="Verdict" className="space-y-4">
      {fit}
      <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">Decode speed</dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {performance.tokensPerSecond.toFixed(1)} tokens/sec
          </dd>
        </div>
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            {firstTokenLabel(batchSize)}
          </dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {formatDuration(performance.timeToFirstToken)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            Max sessions at {sequenceLength.toLocaleString('en-US')} tokens
          </dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {maxSessions === null ? 'n/a' : maxSessions.toLocaleString('en-US')}
          </dd>
        </div>
      </dl>
    </section>
  )
}
