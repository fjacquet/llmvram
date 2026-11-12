import { InfoTip } from '@components/common/InfoTip'
import { isDefaultMix } from '@engines/context-mix'
import { KV_TIER_PRESETS } from '@engines/kv-tier'
import type { ContextMix, ReplicaCandidate, SizingResult } from '@engines/types'
import { formatDuration } from '@utils/formatDuration'
import {
  formatTokensShort,
  maxUsersLine,
  mixLabel,
  targetLine,
  ttftCeilingLine,
  ttftLine,
  worstCaseLabel,
} from '@utils/formatTTFT'
import type { GPU } from '@utils/schemas'
import { decodeBatchNote, SIZING_SPEED_NOTE } from '@utils/sizingNote'
import Decimal from 'decimal.js'

interface SizingResultsProps {
  result: SizingResult
  gpu: GPU
  sequenceLength: number
  contextMix: ContextMix
  ttftTargetSeconds: number | null
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

function Sessions({ c }: { c: ReplicaCandidate }) {
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
        {`assumes ${Math.round(c.activeShare * 100)}% of sessions decode at once`}
      </p>
      <p className={MUTED}>
        {`Resume ${seconds(c.resumeSeconds)} vs recompute ${seconds(c.recomputeSeconds)}`}
      </p>
    </div>
  )
}

/** The answer in servers or racks: replica shape, packing and per-user speed (expected, with ceiling) */
export function SizingResults({
  result,
  gpu,
  sequenceLength,
  contextMix,
  ttftTargetSeconds,
}: SizingResultsProps) {
  const { chosen: c, closest, neverMeetsTarget } = result
  if (result.fits && !c && neverMeetsTarget && ttftTargetSeconds !== null) {
    return (
      <div className={CARD}>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          {`No ${result.unit} replica meets ${formatDuration(new Decimal(ttftTargetSeconds))} at ${formatTokensShort(neverMeetsTarget.largestClassTokens)} tokens`}
        </h2>
      </div>
    )
  }
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
  const batchNote = decodeBatchNote(c.decodeBatch)
  const mixActive = !isDefaultMix(contextMix, sequenceLength)
  const target = targetLine(c.ttft, ttftTargetSeconds)
  return (
    <div className={CARD}>
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          {`${plural(result.units, result.unit)} ${gpu.name}`}
        </h2>
        <p data-testid="sized-by" className="text-sm text-gray-600 dark:text-gray-400">
          {`Sized by ${c.boundBy === 'ttft' ? 'TTFT target' : 'memory'}${mixActive ? ` (memory sized on the mix average, ${mixLabel(contextMix)})` : ''}`}
        </p>
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

      <Sessions c={c} />

      {batchNote && <p className="text-sm text-amber-700 dark:text-amber-400">{batchNote}</p>}

      <div className={TILE}>
        <div className="flex items-center gap-1 mb-2">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Per user, at {c.decodeBatch} {c.decodeBatch === 1 ? 'session' : 'sessions'} decoding at
            once
          </p>
          <InfoTip text={SIZING_SPEED_NOTE} />
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
              {ttftLine(c.ttft)}
            </p>
            <p className={MUTED}>{worstCaseLabel(c.ttft)}</p>
            <p className={MUTED}>{ttftCeilingLine(c.ttft)}</p>
            {target && (
              <p
                className={`text-sm ${target.met ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}`}
              >
                {target.text}
              </p>
            )}
            {c.burstCap !== null && <p className={MUTED}>{maxUsersLine(c.burstCap)}</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
