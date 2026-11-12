import Decimal from 'decimal.js'
import { estimatePerformance, type PerformanceParams } from './performance'
import type { BurstTTFT, ClassPrefill, ContextMix } from './types'

export type { BurstTTFT, ClassPrefill }

/** One batch-1 estimate per class; prefill is per-prompt, so no batch enters it. */
export function classPrefills(
  params: Omit<PerformanceParams, 'batchSize' | 'sequenceLength' | 'contextMix'>,
  mix: ContextMix,
): ClassPrefill[] {
  return mix.map(({ tokens, share }) => {
    const p = estimatePerformance({ ...params, batchSize: 1, sequenceLength: tokens })
    return {
      tokens,
      share,
      expectedTTFT: p.expected.timeToFirstToken,
      ceilingTTFT: p.timeToFirstToken,
    }
  })
}

const meanOf = (classes: ClassPrefill[], pick: (c: ClassPrefill) => Decimal) =>
  classes.reduce((sum, c) => sum.plus(pick(c).times(c.share)), new Decimal(0))

const largestOf = (classes: ClassPrefill[]) =>
  classes.reduce((a, c) => (c.tokens > a.tokens ? c : a))

/** Last served = one largest-class prompt (a burst holds at least one) + (B - 1) mean prompts. */
export function burstTTFT(classes: ClassPrefill[], burst: number): BurstTTFT {
  const n = Math.max(1, burst)
  const largest = largestOf(classes)
  const last = (one: Decimal, pick: (c: ClassPrefill) => Decimal) =>
    one.plus(meanOf(classes, pick).times(n - 1))
  return {
    burst: n,
    firstExpected: largest.expectedTTFT,
    firstCeiling: largest.ceilingTTFT,
    lastExpected: last(largest.expectedTTFT, (c) => c.expectedTTFT),
    lastCeiling: last(largest.ceilingTTFT, (c) => c.ceilingTTFT),
  }
}

/** Most simultaneous prompts whose last-served expected TTFT stays within the target; 0 if the largest alone misses. */
export function maxUsersWithinTarget(classes: ClassPrefill[], targetSeconds: number): number {
  const largest = largestOf(classes).expectedTTFT
  if (largest.gt(targetSeconds)) return 0
  const mean = meanOf(classes, (c) => c.expectedTTFT)
  return Math.floor(new Decimal(targetSeconds).minus(largest).div(mean).toNumber() + 1e-9) + 1
}

export function meetsTarget(b: BurstTTFT, targetSeconds: number | null): boolean | null {
  return targetSeconds === null ? null : b.lastExpected.lte(targetSeconds)
}
