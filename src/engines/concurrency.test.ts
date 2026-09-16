import { perUserTimeToFirstToken, perUserTokensPerSecond } from '@engines/concurrency'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'

describe('perUserTokensPerSecond', () => {
  it('never lets one user outrun the whole machine', () => {
    // The defect this replaces: batchSize was applied a second time, so at
    // batch 16 / 8 users the panel reported 17188.5 tok/s per user against an
    // aggregate of 8594.2 — each of eight users twice as fast as the machine.
    const aggregate = new Decimal(8594.2)
    const perUser = perUserTokensPerSecond(aggregate, 8)

    expect(perUser.toNumber()).toBeLessThanOrEqual(aggregate.toNumber())
    expect(perUser.toNumber()).toBeCloseTo(1074.275, 3)
  })

  it('splits the aggregate evenly, so the users sum back to it', () => {
    const aggregate = new Decimal(1000)
    for (const users of [1, 2, 8, 37]) {
      const perUser = perUserTokensPerSecond(aggregate, users)
      expect(perUser.mul(users).toNumber()).toBeCloseTo(aggregate.toNumber(), 6)
    }
  })

  it('returns the aggregate unchanged for a single user', () => {
    expect(perUserTokensPerSecond(new Decimal(500), 1).toNumber()).toBe(500)
  })

  it('does not divide by zero on a degenerate user count', () => {
    expect(perUserTokensPerSecond(new Decimal(500), 0).toNumber()).toBe(500)
  })
})

describe('perUserTimeToFirstToken', () => {
  it('is never faster than the idle single-request latency', () => {
    // The defect this replaces: `× users ÷ batch` reported 28.41s per user
    // against a 56.82s single-request TTFT — concurrency making the machine
    // faster than having it to yourself.
    const ttft = new Decimal(56.82)

    for (const [users, batch] of [
      [8, 16],
      [1, 64],
      [2, 2],
      [64, 64],
    ] as const) {
      expect(perUserTimeToFirstToken(ttft, users, batch).toNumber()).toBeGreaterThanOrEqual(
        ttft.toNumber(),
      )
    }
  })

  it('charges one wave when every user fits in a single batch', () => {
    const ttft = new Decimal(10)
    expect(perUserTimeToFirstToken(ttft, 8, 16).toNumber()).toBe(10)
    expect(perUserTimeToFirstToken(ttft, 16, 16).toNumber()).toBe(10)
  })

  it('charges a wave per batch once users exceed the batch', () => {
    const ttft = new Decimal(10)
    expect(perUserTimeToFirstToken(ttft, 32, 16).toNumber()).toBe(20)
    expect(perUserTimeToFirstToken(ttft, 33, 16).toNumber()).toBe(30)
  })

  it('does not divide by zero on a degenerate batch size', () => {
    expect(perUserTimeToFirstToken(new Decimal(10), 4, 0).toNumber()).toBe(40)
  })
})
