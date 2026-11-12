import { perUserTokensPerSecond } from '@engines/concurrency'
import { burstTTFT, maxUsersWithinTarget } from '@engines/ttft'
import type { ClassPrefill, PerformanceEstimate } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import {
  formatTokensShort,
  maxUsersLine,
  ttftCeilingLine,
  ttftLine,
  worstCaseLabel,
} from '@utils/formatTTFT'
import { firstTokenLabel } from '@utils/perfLabels'
import type { ReactNode } from 'react'

interface PerformanceSectionProps {
  performance: PerformanceEstimate
  concurrentUsers: number
  batchSize: number
  /** One batch-1 prefill per context class (classPrefills) */
  classes: ClassPrefill[]
  /** Seconds; null when no target is set */
  ttftTargetSeconds: number | null
  /**
   * Rendered between the metric grid and the per-user metrics block — this is
   * where CapacitySection (max sessions, KV tier) sits, matching the DOM
   * order this card had before the ResultsPanel split. Each top-level child
   * is expected to carry its own `mt-4 pt-4 border-t` divider, as
   * CapacitySection's blocks do; this slot adds no wrapper of its own so
   * dividers don't stack.
   */
  children?: ReactNode
}

/**
 * Aggregate performance metrics (decode speed, TTFT, bottleneck, prefill) plus
 * per-user metrics when more than one concurrent user is configured.
 */
export function PerformanceSection({
  performance,
  concurrentUsers,
  batchSize,
  classes,
  ttftTargetSeconds,
  children,
}: PerformanceSectionProps) {
  const burst = burstTTFT(classes, concurrentUsers)
  const maxUsers =
    ttftTargetSeconds === null ? null : maxUsersWithinTarget(classes, ttftTargetSeconds)
  return (
    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Decode Speed</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {performance.expected.tokensPerSecond.toFixed(1)} tokens/sec
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Theoretical ceiling: {performance.tokensPerSecond.toFixed(1)} tokens/sec
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            {firstTokenLabel(batchSize)}
          </p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {formatDuration(performance.expected.timeToFirstToken)}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Theoretical ceiling: {formatDuration(performance.timeToFirstToken)}
          </p>
          {performance.prefillEstimateDegraded && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              rough estimate — no FLOPS data for this GPU
            </p>
          )}
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Bottleneck</p>
          <p
            className={`text-lg font-semibold ${
              performance.bottleneck === 'balanced'
                ? 'text-green-600 dark:text-green-400'
                : performance.bottleneck === 'memory'
                  ? 'text-yellow-600 dark:text-yellow-400'
                  : 'text-blue-600 dark:text-blue-400'
            }`}
          >
            {performance.bottleneck === 'balanced'
              ? 'Balanced'
              : performance.bottleneck === 'memory'
                ? 'Memory bandwidth'
                : 'Compute'}
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Prompt Processing</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {performance.expected.prefillSeconds
              ? formatDuration(performance.expected.prefillSeconds)
              : 'n/a'}
          </p>
          {performance.prefillSeconds && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Theoretical ceiling: {formatDuration(performance.prefillSeconds)}
            </p>
          )}
          {performance.prefillSeconds && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {performance.prefillBottleneck === 'attention'
                ? 'attention-dominated (quadratic)'
                : 'weight-dominated (linear)'}
            </p>
          )}
        </div>
      </div>

      {children}

      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600 space-y-1">
        <p className="text-sm font-semibold text-gray-900 dark:text-white">{ttftLine(burst)}</p>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          {worstCaseLabel(
            burst,
            classes.length > 1 ? Math.max(...classes.map((c) => c.tokens)) : undefined,
          )}
        </p>
        <p className="text-xs text-gray-500 dark:text-gray-400">{ttftCeilingLine(burst)}</p>
        {classes.length > 1 &&
          classes.map((c) => (
            <p key={c.tokens} className="text-xs text-gray-500 dark:text-gray-400">
              {`${formatTokensShort(c.tokens)} prompt alone: ${formatDuration(c.expectedTTFT)} (ceiling ${formatDuration(c.ceilingTTFT)})`}
            </p>
          ))}
        {maxUsers !== null && (
          <p className="text-xs text-gray-700 dark:text-gray-300">{maxUsersLine(maxUsers)}</p>
        )}
      </div>

      {/* Per-user metrics (only visible when concurrentUsers > 1) */}
      {concurrentUsers > 1 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600">
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
            Multi-user metrics ({concurrentUsers} concurrent users)
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Per-user speed</p>
              <p className="text-base font-semibold text-gray-900 dark:text-white">
                {perUserTokensPerSecond(
                  performance.expected.tokensPerSecond,
                  concurrentUsers,
                ).toFixed(1)}{' '}
                tok/s
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Theoretical ceiling:{' '}
                {perUserTokensPerSecond(performance.tokensPerSecond, concurrentUsers).toFixed(1)}{' '}
                tok/s
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
