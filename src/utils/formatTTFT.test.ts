import { burstTTFT } from '@engines/ttft'
import type { ClassPrefill } from '@engines/types'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import {
  formatTokensShort,
  maxUsersLine,
  mixLabel,
  targetLine,
  ttftCeilingLine,
  ttftLine,
  worstCaseLabel,
} from './formatTTFT'

const one: ClassPrefill[] = [
  { tokens: 4096, share: 1, expectedTTFT: new Decimal(0.1), ceilingTTFT: new Decimal(0.05) },
]
const b = burstTTFT(one, 100)

describe('formatTTFT', () => {
  it('writes the TTFT line for one class and 100 users', () => {
    expect(ttftLine(b)).toBe('TTFT 100.0 ms first, 10.00 s last (100 users at once)')
    expect(worstCaseLabel(b)).toBe(
      'worst case: all 100 prompts at once, first come first served, no prefix cache',
    )
    expect(ttftCeilingLine(b)).toBe('Theoretical ceiling: 50.0 ms first, 5.00 s last')
  })

  it('says "1 user" for a burst of one', () => {
    expect(ttftLine(burstTTFT(one, 1))).toContain('(1 user at once)')
  })

  it('reports a target as met or missed by the last-served overshoot', () => {
    expect(targetLine(b, 10)).toEqual({ text: 'Target 10.00 s: met', met: true })
    expect(targetLine(b, 4)).toEqual({ text: 'Target 4.00 s: missed by 6.00 s', met: false })
    expect(targetLine(b, null)).toBeNull()
  })

  it('formats the max-users line and the mix label', () => {
    expect(maxUsersLine(0)).toBe('Max users within target: 0')
    expect(
      mixLabel([
        { tokens: 4096, share: 0.9 },
        { tokens: 262144, share: 0.1 },
      ]),
    ).toBe('mix: 90% 4K, 10% 256K')
    expect(formatTokensShort(1048576)).toBe('1M')
    expect(formatTokensShort(512)).toBe('512')
  })
})
