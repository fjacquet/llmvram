import type Decimal from 'decimal.js'

/**
 * Per-user decode throughput under concurrent load
 *
 * `tokensPerSecond` from estimatePerformance is AGGREGATE: both rooflines
 * already multiply by batchSize (see performance.ts, memoryBoundTPS and
 * computeBoundTPS). Splitting it across the users sharing the machine gives
 * what one user sees.
 *
 * Multiplying by batchSize here would apply it twice, and produced the
 * impossible result of a single user outrunning the whole machine.
 */
export function perUserTokensPerSecond(
  aggregateTokensPerSecond: Decimal,
  concurrentUsers: number,
): Decimal {
  return aggregateTokensPerSecond.div(Math.max(1, concurrentUsers))
}

/**
 * Per-user time to first token under concurrent load
 *
 * `timeToFirstToken` is a per-request latency for ONE sequence — prefill
 * deliberately does not apply batch (see performance.ts). Users are served
 * batchSize at a time, so a user waits for the waves ahead of them plus their
 * own: ceil(concurrentUsers / batchSize).
 *
 * The result is therefore never below the single-request figure. Concurrency
 * cannot make time-to-first-token faster than an idle machine, which the
 * previous `× users ÷ batch` form claimed whenever batchSize exceeded the
 * user count.
 */
export function perUserTimeToFirstToken(
  timeToFirstToken: Decimal,
  concurrentUsers: number,
  batchSize: number,
): Decimal {
  const waves = Math.ceil(Math.max(1, concurrentUsers) / Math.max(1, batchSize))
  return timeToFirstToken.mul(Math.max(1, waves))
}
