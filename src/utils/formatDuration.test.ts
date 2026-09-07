import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { formatDuration, formatDurationMs } from './formatDuration'

describe('formatDuration', () => {
  it('formats just under the ms/s boundary as milliseconds', () => {
    expect(formatDuration(new Decimal(0.999))).toBe('999.0 ms')
  })

  it('formats exactly 1000 ms as seconds', () => {
    expect(formatDuration(new Decimal(1))).toBe('1.00 s')
  })

  it('formats just under the s/min boundary as seconds', () => {
    expect(formatDuration(new Decimal(59.9))).toBe('59.90 s')
  })

  it('formats exactly 60 s as minutes', () => {
    expect(formatDuration(new Decimal(60))).toBe('1.0 min')
  })

  it('formats a multi-hour duration as hours', () => {
    // 122,233 s ≈ Llama 4 Scout prefill at 10M tokens
    expect(formatDuration(new Decimal(122233))).toBe('34.0 h')
  })
})

describe('formatDurationMs', () => {
  it('converts plain milliseconds and delegates to formatDuration', () => {
    expect(formatDurationMs(999)).toBe('999.0 ms')
    expect(formatDurationMs(1000)).toBe('1.00 s')
    expect(formatDurationMs(3568073)).toBe('59.5 min')
  })
})
