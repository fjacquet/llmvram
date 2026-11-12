import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { defaultMix, isDefaultMix, largestContext, mixWeighted, normalizeMix } from './context-mix'

const three = [
  { tokens: 4096, share: 0.7 },
  { tokens: 32768, share: 0.2 },
  { tokens: 262144, share: 0.5 },
]

describe('context mix', () => {
  it('defaults to one class at the sequence length', () => {
    expect(defaultMix(4096)).toEqual([{ tokens: 4096, share: 1 }])
  })

  it('lets the last class take the remainder', () => {
    const m = normalizeMix(three)
    expect(m[2]?.share).toBeCloseTo(0.1, 12)
    expect(m.reduce((s, c) => s + c.share, 0)).toBeCloseTo(1, 12)
  })

  it('cuts four classes to three and drops non-positive shares', () => {
    expect(normalizeMix([...three, { tokens: 8, share: 0.1 }])).toHaveLength(3)
    expect(
      normalizeMix([
        { tokens: 8, share: 0 },
        { tokens: 16, share: 0.3 },
      ]),
    ).toEqual([{ tokens: 16, share: 1 }])
  })

  it('throws on an empty mix', () => {
    expect(() => normalizeMix([])).toThrow()
  })

  it('finds the largest context', () => {
    expect(largestContext(three)).toBe(262144)
  })

  it('weights by share', () => {
    expect(mixWeighted(defaultMix(8192), (t) => new Decimal(t)).toNumber()).toBe(8192)
    expect(
      mixWeighted(
        [
          { tokens: 100, share: 0.5 },
          { tokens: 300, share: 0.5 },
        ],
        (t) => new Decimal(t),
      ).toNumber(),
    ).toBe(200)
  })

  it('recognises the default mix', () => {
    expect(isDefaultMix(defaultMix(8192), 8192)).toBe(true)
    expect(isDefaultMix(defaultMix(8192), 4096)).toBe(false)
    expect(isDefaultMix(three, 4096)).toBe(false)
  })
})
