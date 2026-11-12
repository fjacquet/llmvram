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

type Strung<T> = { [K in keyof T]: T[K] extends Decimal ? string : T[K] }
type SerializedCandidate = Omit<ReplicaCandidate, DecimalKey | 'ttft' | 'classes'> &
  Record<DecimalKey, string> & {
    ttft: Strung<ReplicaCandidate['ttft']>
    classes: Array<Strung<ReplicaCandidate['classes'][number]>>
  }
export type SerializedSizingResult = Omit<SizingResult, 'chosen' | 'closest'> & {
  chosen: SerializedCandidate | null
  closest: { candidate: SerializedCandidate; lackingGB: number } | null
}

// Decimals lose their methods in structured cloning: cross the worker boundary as strings
const mapDecimals = <T extends object, R>(o: T, f: (d: Decimal) => unknown): R =>
  Object.fromEntries(
    Object.entries(o).map(([k, v]) => [
      k,
      v instanceof Decimal || typeof v === 'string' ? f(v as Decimal) : v,
    ]),
  ) as R
const toStrings = (c: ReplicaCandidate): SerializedCandidate => ({
  ...c,
  ttft: mapDecimals(c.ttft, (d) => d.toString()),
  classes: c.classes.map((k) => mapDecimals(k, (d) => d.toString())),
  ...(Object.fromEntries(DECIMAL_KEYS.map((k) => [k, c[k].toString()])) as Record<
    DecimalKey,
    string
  >),
})
const toDecimals = (c: SerializedCandidate): ReplicaCandidate => ({
  ...c,
  ttft: mapDecimals(c.ttft, (d) => new Decimal(d)),
  classes: c.classes.map((k) => mapDecimals(k, (d) => new Decimal(d))),
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
