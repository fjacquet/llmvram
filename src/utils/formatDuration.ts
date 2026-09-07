import Decimal from 'decimal.js'

/**
 * Format a duration given in seconds, choosing readable units.
 *
 * Durations across this app range from sub-millisecond decode steps to
 * multi-hour prefills at 10M-token context (122,233 s for Llama 4 Scout at
 * 10M tokens, which renders as "34.0 h" instead of the unreadable
 * "122232.82 s").
 *
 * Branches:
 * - under 1000 ms   → `${ms.toFixed(1)} ms`
 * - under 60 s      → `${seconds.toFixed(2)} s`
 * - 60 s and above  → minutes below 60 min (`${minutes.toFixed(1)} min`),
 *                      otherwise hours (`${hours.toFixed(1)} h`)
 */
export function formatDuration(seconds: Decimal): string {
  const ms = seconds.mul(1000)
  if (ms.lessThan(1000)) return `${ms.toFixed(1)} ms`
  if (seconds.lessThan(60)) return `${seconds.toFixed(2)} s`

  const minutes = seconds.div(60)
  if (minutes.lessThan(60)) return `${minutes.toFixed(1)} min`

  const hours = minutes.div(60)
  return `${hours.toFixed(1)} h`
}

/**
 * Sibling of {@link formatDuration} for plain millisecond numbers (e.g.
 * comparison snapshots, which store `timeToFirstToken` as a `number` in ms
 * rather than a `Decimal` in seconds).
 */
export function formatDurationMs(ms: number): string {
  return formatDuration(new Decimal(ms).div(1000))
}
