import type { BurstTTFT, ContextMix } from '@engines/types'
import Decimal from 'decimal.js'
import { formatDuration } from './formatDuration'

/** Short token count: 4096 -> "4K", 262144 -> "256K", 1048576 -> "1M" */
export function formatTokensShort(tokens: number): string {
  const unit = (v: number, s: string) => `${Number(v.toFixed(1))}${s}`
  if (tokens >= 1_048_576) return unit(tokens / 1_048_576, 'M')
  if (tokens >= 1024) return unit(tokens / 1024, 'K')
  return String(tokens)
}

/** "mix: 90% 4K, 10% 256K" */
export function mixLabel(mix: ContextMix): string {
  return `mix: ${mix.map((c) => `${Math.round(c.share * 100)}% ${formatTokensShort(c.tokens)}`).join(', ')}`
}

/** The TTFT line shared by Inference, Sizing and the exports; first = headline, last = worst case */
export function ttftLine(b: BurstTTFT): string {
  return `TTFT ${formatDuration(b.firstExpected)} first, ${formatDuration(b.lastExpected)} last (${b.burst} ${b.burst === 1 ? 'user' : 'users'} at once)`
}

export function worstCaseLabel(b: BurstTTFT): string {
  return `worst case: all ${b.burst} prompts at once, first come first served, no prefix cache`
}

export function ttftCeilingLine(b: BurstTTFT): string {
  return `Theoretical ceiling: ${formatDuration(b.firstCeiling)} first, ${formatDuration(b.lastCeiling)} last`
}

/** The target applies to the last served; null target gives no line */
export function targetLine(b: BurstTTFT, targetSeconds: number | null): string | null {
  if (targetSeconds === null) return null
  const t = `Target ${formatDuration(new Decimal(targetSeconds))}`
  return b.lastExpected.lte(targetSeconds)
    ? `${t}: met`
    : `${t}: missed by ${formatDuration(b.lastExpected.minus(targetSeconds))}`
}

export function maxUsersLine(n: number): string {
  return `Max users within target: ${n}`
}
