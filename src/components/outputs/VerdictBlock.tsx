import { InfoTip } from '@components/common/InfoTip'
import { isDefaultMix } from '@engines/context-mix'
import { burstTTFT } from '@engines/ttft'
import type { ClassPrefill, PerformanceEstimate } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import { formatTokensShort, mixLabel, targetLine } from '@utils/formatTTFT'
import { firstTokenLabel } from '@utils/perfLabels'
import type { ReactNode } from 'react'

interface VerdictBlockProps {
  /** The fit gauge (FitIndicator) for the displayed breakdown */
  fit: ReactNode
  performance: PerformanceEstimate
  batchSize: number
  maxSessions: number | null
  sequenceLength: number
  /** One batch-1 prefill per context class */
  classes: ClassPrefill[]
  concurrentUsers: number
  ttftTargetSeconds: number | null
  /** Tokens of the largest class when one session of it does not fit at 90%; null when it fits */
  largestClassMiss: number | null
}

/** The answer first: does it fit, how fast, first-token delay, sessions (ADR 0005) */
export function VerdictBlock({
  fit,
  performance,
  batchSize,
  maxSessions,
  sequenceLength,
  classes,
  concurrentUsers,
  ttftTargetSeconds,
  largestClassMiss,
}: VerdictBlockProps) {
  const target = targetLine(burstTTFT(classes, concurrentUsers), ttftTargetSeconds)
  return (
    <section data-testid="verdict" aria-label="Verdict" className="space-y-4">
      {fit}
      {largestClassMiss !== null && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-800 dark:text-red-200">
            Does not fit: one session at the largest class ({formatTokensShort(largestClassMiss)}{' '}
            tokens) exceeds 90% of GPU memory
          </p>
        </div>
      )}
      <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1 flex items-center gap-1">
            Decode speed
            <InfoTip text="Expected = theoretical ceiling × measured vLLM efficiency (median of published runs; sources: ADR 0009). Decode efficiency depends on load: 0.47 up to batch 32, 0.425 for 33-128, 0.25 above 128 (provisional, 1 point)." />
          </dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {performance.expected.tokensPerSecond.toFixed(1)} tokens/sec
          </dd>
          <dd className="text-xs text-gray-500 dark:text-gray-400">
            Theoretical ceiling: {performance.tokensPerSecond.toFixed(1)} tokens/sec
          </dd>
        </div>
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            {firstTokenLabel(batchSize)}
          </dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {formatDuration(performance.expected.timeToFirstToken)}
          </dd>
          <dd className="text-xs text-gray-500 dark:text-gray-400">
            Theoretical ceiling: {formatDuration(performance.timeToFirstToken)}
          </dd>
        </div>
        <div>
          <dt className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            {isDefaultMix(classes, sequenceLength)
              ? `Max sessions at ${sequenceLength.toLocaleString('en-US')} tokens`
              : `Max sessions (${mixLabel(classes)})`}
          </dt>
          <dd className="text-lg font-semibold text-gray-900 dark:text-white">
            {maxSessions === null ? 'n/a' : maxSessions.toLocaleString('en-US')}
          </dd>
        </div>
      </dl>
      {target !== null && (
        <p
          className={`text-sm font-medium ${
            target.endsWith(': met')
              ? 'text-green-700 dark:text-green-400'
              : 'text-red-700 dark:text-red-400'
          }`}
        >
          {target}
        </p>
      )}
    </section>
  )
}
