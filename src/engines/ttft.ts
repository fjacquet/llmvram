import Decimal from 'decimal.js'
import { estimatePerformance, type PerformanceParams } from './performance'
import type { ContextMix } from './types'

export interface ClassPrefill {
  tokens: number
  share: number
  expectedTTFT: Decimal
  ceilingTTFT: Decimal
}

export interface BurstTTFT {
  burst: number
  /** A lone prompt at the largest class */
  firstExpected: Decimal
  firstCeiling: Decimal
  /** Worst case: the last of `burst` prompts, prefill serialised (compute-bound) */
  lastExpected: Decimal
  lastCeiling: Decimal
}

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

export function burstTTFT(classes: ClassPrefill[], burst: number): BurstTTFT {
  const n = Math.max(1, burst)
  const largest = classes.reduce((a, c) => (c.tokens > a.tokens ? c : a))
  const mean = (pick: (c: ClassPrefill) => Decimal) =>
    classes.reduce((sum, c) => sum.plus(pick(c).times(c.share)), new Decimal(0))
  return {
    burst: n,
    firstExpected: largest.expectedTTFT,
    firstCeiling: largest.ceilingTTFT,
    lastExpected: mean((c) => c.expectedTTFT).times(n),
    lastCeiling: mean((c) => c.ceilingTTFT).times(n),
  }
}

/** Most simultaneous prompts whose last-served expected TTFT stays within the target; 0 if one misses. */
export function maxUsersWithinTarget(classes: ClassPrefill[], targetSeconds: number): number {
  const perPrompt = classes.reduce((s, c) => s.plus(c.expectedTTFT.times(c.share)), new Decimal(0))
  return Math.floor(new Decimal(targetSeconds).div(perPrompt).toNumber())
}

export function meetsTarget(b: BurstTTFT, targetSeconds: number | null): boolean | null {
  return targetSeconds === null ? null : b.lastExpected.lte(targetSeconds)
}
