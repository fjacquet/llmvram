import Decimal from 'decimal.js'
import type { ContextMix } from './types'

export const CONTEXT_PRESETS: ReadonlyArray<{
  label: 'Chat' | 'RAG' | 'Long documents'
  tokens: number
}> = [
  { label: 'Chat', tokens: 4096 },
  { label: 'RAG', tokens: 32768 },
  { label: 'Long documents', tokens: 262144 },
]

const MAX_CLASSES = 3

export function defaultMix(sequenceLength: number): ContextMix {
  return [{ tokens: sequenceLength, share: 1 }]
}

/** Keeps 1 to 3 classes with positive shares; the last takes the remainder so shares sum to 1. */
export function normalizeMix(mix: ContextMix): ContextMix {
  const kept = mix.filter((c) => c.share > 0).slice(0, MAX_CLASSES)
  if (kept.length === 0) throw new Error('Context mix needs at least one class with a share > 0')
  // A last class left with no remainder is dropped; the one before it then takes the remainder.
  for (;;) {
    const rest = kept.slice(0, -1).reduce((s, c) => s + c.share, 0)
    if (kept.length === 1 || 1 - rest > 1e-12) {
      const last = kept[kept.length - 1]
      if (last) kept[kept.length - 1] = { tokens: last.tokens, share: 1 - rest }
      return kept.map((c) => ({ ...c }))
    }
    kept.pop()
  }
}

export function largestContext(mix: ContextMix): number {
  return Math.max(...mix.map((c) => c.tokens))
}

export function mixWeighted(mix: ContextMix, f: (tokens: number) => Decimal): Decimal {
  return mix.reduce((sum, c) => sum.plus(f(c.tokens).times(c.share)), new Decimal(0))
}

export function isDefaultMix(mix: ContextMix, sequenceLength: number): boolean {
  const [c] = mix
  return mix.length === 1 && c?.tokens === sequenceLength && c.share === 1
}
