import type Decimal from 'decimal.js'
import { GPU_MEMORY_UTILIZATION } from './constants'

/**
 * Per-user decode throughput under concurrent load
 *
 * `tokensPerSecond` from estimatePerformance is AGGREGATE: batchSize tokens
 * per decode step (see performance.ts). Splitting it across the users sharing
 * the machine gives what one user sees.
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

/**
 * Maximum concurrent sessions at the configured context length
 *
 * Mirrors vLLM's "Maximum concurrency for N tokens per request" log line
 * (kv_cache_utils.get_max_concurrency_for_kv_cache_config): the memory left
 * after weights, activations and overhead, divided by one request's KV at the
 * full context. vLLM claims GPU_MEMORY_UTILIZATION of the device.
 *
 * Every engine is linear in session count, so the per-GPU breakdown already
 * computed for `concurrentUsers` gives both terms: fixed = total - KV, and one
 * session's KV = KV / concurrentUsers. That KV share already reflects the
 * strategy (TP stops at the KV head count and duplicates MLA; EP and PP split it).
 *
 * @returns sessions (0 when the fixed costs alone do not fit), or null when no
 *          KV sits on the GPU (offloaded), so VRAM does not bound sessions
 */
export function maxConcurrentSessions(params: {
  totalPerGPUGB: number
  kvPerGPUGB: number
  concurrentUsers: number
  gpuVramGB: number
}): number | null {
  const { totalPerGPUGB, kvPerGPUGB, concurrentUsers, gpuVramGB } = params
  if (kvPerGPUGB <= 0) return null
  const fixed = totalPerGPUGB - kvPerGPUGB
  const perSession = kvPerGPUGB / Math.max(1, concurrentUsers)
  return Math.max(0, Math.floor((gpuVramGB * GPU_MEMORY_UTILIZATION - fixed) / perSession))
}
