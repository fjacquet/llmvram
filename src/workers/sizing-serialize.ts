import type { ReplicaCandidate, SizingResult } from '@engines/types'
import Decimal from 'decimal.js'

type DecimalKey =
  | 'perUserTokensPerSecond'
  | 'perUserCeilingTokensPerSecond'
  | 'timeToFirstToken'
  | 'ceilingTimeToFirstToken'
const DECIMAL_KEYS: readonly DecimalKey[] = [
  'perUserTokensPerSecond',
  'perUserCeilingTokensPerSecond',
  'timeToFirstToken',
  'ceilingTimeToFirstToken',
]

type SerializedCandidate = Omit<ReplicaCandidate, DecimalKey> & Record<DecimalKey, string>
export type SerializedSizingResult = Omit<SizingResult, 'chosen' | 'closest'> & {
  chosen: SerializedCandidate | null
  closest: { candidate: SerializedCandidate; lackingGB: number } | null
}

// Decimals lose their methods in structured cloning: cross the worker boundary as strings
const toStrings = (c: ReplicaCandidate): SerializedCandidate => ({
  ...c,
  ...(Object.fromEntries(DECIMAL_KEYS.map((k) => [k, c[k].toString()])) as Record<
    DecimalKey,
    string
  >),
})
const toDecimals = (c: SerializedCandidate): ReplicaCandidate => ({
  ...c,
  ...(Object.fromEntries(DECIMAL_KEYS.map((k) => [k, new Decimal(c[k])])) as Record<
    DecimalKey,
    Decimal
  >),
})

export function serializeSizingResult(r: SizingResult): SerializedSizingResult {
  return {
    ...r,
    chosen: r.chosen && toStrings(r.chosen),
    closest: r.closest && { ...r.closest, candidate: toStrings(r.closest.candidate) },
  }
}

export function reconstructSizingResult(r: SerializedSizingResult): SizingResult {
  return {
    ...r,
    chosen: r.chosen && toDecimals(r.chosen),
    closest: r.closest && { ...r.closest, candidate: toDecimals(r.closest.candidate) },
  }
}
