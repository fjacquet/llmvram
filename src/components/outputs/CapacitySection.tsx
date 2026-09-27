import type { KVTierSettings, kvTierSummary } from '@engines/kv-tier'
import { formatDuration } from '@utils/formatDuration'
import Decimal from 'decimal.js'

type TierSummary = ReturnType<typeof kvTierSummary>

interface CapacitySectionProps {
  maxSessions: number | null
  sequenceLength: number
  concurrentUsers: number
  tierSummary: TierSummary
  kvTier: KVTierSettings
  prefillSeconds: Decimal | null
  /** Repo the weights were measured from, or null when the format is estimated */
  weightSourceRepo: string | null
  quantization: string
}

/**
 * Capacity info: how many sessions fit, what the KV storage tier holds, and
 * whether the weight sizes behind those numbers are measured or estimated.
 */
export function CapacitySection({
  maxSessions,
  sequenceLength,
  concurrentUsers,
  tierSummary,
  kvTier,
  prefillSeconds,
  weightSourceRepo,
  quantization,
}: CapacitySectionProps) {
  return (
    <>
      {/* Max concurrent sessions at this context (vLLM's "Maximum concurrency") */}
      {maxSessions !== null && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">
            Max concurrent sessions at {sequenceLength.toLocaleString('en-US')} tokens
          </p>
          <p
            className={`text-base font-semibold ${
              concurrentUsers > maxSessions
                ? 'text-red-600 dark:text-red-400'
                : 'text-gray-900 dark:text-white'
            }`}
          >
            {maxSessions.toLocaleString('en-US')}
            {concurrentUsers > maxSessions &&
              ` (below the ${concurrentUsers.toLocaleString('en-US')} configured)`}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            At 90% of each GPU&apos;s memory (vLLM default), after weights and overhead
          </p>
        </div>
      )}

      {tierSummary && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-1">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            With the KV storage tier ({Math.round(kvTier.activeShare * 100)}% active)
          </p>
          <p className="text-base font-semibold text-gray-900 dark:text-white">
            {tierSummary.sessionsHeld.toLocaleString('en-US')} sessions held
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Resume {formatDuration(new Decimal(tierSummary.resumeSeconds))}
            {prefillSeconds &&
              ` vs recompute ${formatDuration(prefillSeconds)} (${
                tierSummary.resumeFaster ? 'resume is faster' : 'recompute is faster'
              })`}
          </p>
          <p
            className={`text-xs ${
              tierSummary.overloaded
                ? 'text-red-600 dark:text-red-400'
                : 'text-gray-600 dark:text-gray-400'
            }`}
          >
            Tier reads {tierSummary.trafficGBps.toFixed(1)} GB/s of{' '}
            {tierSummary.tierGBps.toFixed(0)} GB/s available
          </p>
        </div>
      )}

      <p className="text-xs text-gray-500 dark:text-gray-400 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        {weightSourceRepo ? (
          <>
            Weights measured from{' '}
            <a
              href={`https://huggingface.co/${weightSourceRepo}`}
              target="_blank"
              rel="noreferrer"
              className="underline"
            >
              {weightSourceRepo}
            </a>
          </>
        ) : (
          `Weights estimated: no reference checkpoint for ${quantization}`
        )}
      </p>
    </>
  )
}
