import { InfoTip } from '@components/common/InfoTip'
import { KV_TIER_PRESETS, type KVTierSettings } from '@engines/kv-tier'
import type { ReplicaCandidate, SizingResult } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import type { GPU } from '@utils/schemas'
import Decimal from 'decimal.js'

interface SizingResultsProps {
  result: SizingResult
  gpu: GPU
  kvTier: KVTierSettings
}

const CARD =
  'bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 space-y-4'
const TILE =
  'bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700'
const MUTED = 'text-xs text-gray-500 dark:text-gray-400'

function Stat(props: { label: string; value: string | number; testId: string }) {
  return (
    <div className={TILE}>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">{props.label}</p>
      <p data-testid={props.testId} className="text-lg font-semibold text-gray-900 dark:text-white">
        {props.value}
      </p>
    </div>
  )
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

function Sessions({ c, kvTier }: { c: ReplicaCandidate; kvTier: KVTierSettings }) {
  if (!c.tierUsed || c.tierUsed === 'none') {
    return (
      <div>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          {`${c.sessionsPerReplica} sessions per replica`}
        </p>
        {c.tierOverloaded && (
          <p className="text-sm text-amber-700 dark:text-amber-400">
            Tier cannot sustain the session churn; sized on HBM only
          </p>
        )}
      </div>
    )
  }
  const seconds = (s: number | null) => (s === null ? 'unknown' : formatDuration(new Decimal(s)))
  return (
    <div className="space-y-1">
      <p className="text-sm text-gray-700 dark:text-gray-300">
        {`${c.sessionsPerReplica} open sessions per replica (${c.hotSessions} in HBM, the rest parked in ${KV_TIER_PRESETS[c.tierUsed].label})`}
      </p>
      {/* This assumption can cut the rack count about 4x: keep it next to the figure it drives */}
      <p className="text-sm font-medium text-amber-700 dark:text-amber-400">
        {`assumes ${Math.round(kvTier.activeShare * 100)}% of sessions decode at once`}
      </p>
      <p className={MUTED}>
        {`Resume ${seconds(c.resumeSeconds)} vs recompute ${seconds(c.recomputeSeconds)}`}
      </p>
    </div>
  )
}

/** The answer in servers or racks: replica shape, packing and per-user speed (expected, with ceiling) */
export function SizingResults({ result, gpu, kvTier }: SizingResultsProps) {
  const { chosen: c, closest } = result
  if (!result.fits || !c) {
    return (
      <div className={CARD}>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">{`No ${result.unit}s of ${gpu.name} fit`}</h2>
        {closest && (
          <p role="alert" className="text-sm text-red-700 dark:text-red-400">
            {`Does not fit: closest is TP ${closest.candidate.tp} × ${closest.candidate.nodes}, lacking ${closest.lackingGB.toFixed(1)} GB per GPU`}
          </p>
        )}
      </div>
    )
  }
  const multiServer = c.nodes > 1
  return (
    <div className={CARD}>
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          {`${plural(result.units, result.unit)} ${gpu.name}`}
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {`replica = ${c.tp} GPUs (tensor parallel)${multiServer ? ` × ${c.nodes} servers` : ''}`}
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="Replicas" value={result.replicas} testId="sizing-replicas" />
        {!multiServer && (
          <>
            <Stat
              label={`Replicas per ${result.unit}`}
              value={result.replicasPerUnit}
              testId="sizing-per-unit"
            />
            <Stat
              label={`Idle GPUs per ${result.unit}`}
              value={result.idleGPUsPerUnit}
              testId="sizing-idle"
            />
          </>
        )}
      </div>

      <Sessions c={c} kvTier={kvTier} />

      <div className={TILE}>
        <div className="flex items-center gap-1 mb-2">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Per user, at {c.decodeBatch} {c.decodeBatch === 1 ? 'session' : 'sessions'} decoding at
            once
          </p>
          <InfoTip text="Unit counts use the expected speed, which includes the efficiency losses real serving shows. The theoretical ceiling is the bandwidth and compute limit, shown beside each figure for reference." />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Decode speed</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              {c.perUserTokensPerSecond.toFixed(1)} tokens/sec
            </p>
            <p className={MUTED}>
              Theoretical ceiling: {c.perUserCeilingTokensPerSecond.toFixed(1)} tokens/sec
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Time to first token</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              {formatDuration(c.timeToFirstToken)}
            </p>
            <p className={MUTED}>
              Theoretical ceiling: {formatDuration(c.ceilingTimeToFirstToken)}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
