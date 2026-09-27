import { perUserTimeToFirstToken, perUserTokensPerSecond } from '@engines/concurrency'
import type { PerformanceEstimate } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'

interface PerformanceSectionProps {
  performance: PerformanceEstimate
  concurrentUsers: number
  batchSize: number
}

/**
 * Aggregate performance metrics (decode speed, TTFT, bottleneck, prefill) plus
 * per-user metrics when more than one concurrent user is configured.
 */
export function PerformanceSection({
  performance,
  concurrentUsers,
  batchSize,
}: PerformanceSectionProps) {
  return (
    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Decode Speed</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {performance.tokensPerSecond.toFixed(1)} tokens/sec
          </p>
        </div>
        <div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Time to First Token</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {formatDuration(performance.timeToFirstToken)}
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
            {performance.prefillSeconds ? formatDuration(performance.prefillSeconds) : 'n/a'}
          </p>
          {performance.prefillSeconds && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {performance.prefillBottleneck === 'attention'
                ? 'attention-dominated (quadratic)'
                : 'weight-dominated (linear)'}
            </p>
          )}
        </div>
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
                {perUserTokensPerSecond(performance.tokensPerSecond, concurrentUsers).toFixed(1)}{' '}
                tok/s
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Per-user TTFT (est.)</p>
              <p className="text-base font-semibold text-gray-900 dark:text-white">
                {formatDuration(
                  perUserTimeToFirstToken(performance.timeToFirstToken, concurrentUsers, batchSize),
                )}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
