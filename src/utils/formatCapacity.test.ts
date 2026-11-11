import { describe, expect, it } from 'vitest'
import { formatCapacity, totalCapacityLine } from './formatCapacity'

describe('formatCapacity', () => {
  it('uses GB under 1000 and TB above', () => {
    expect(formatCapacity(640)).toBe('640 GB')
    expect(formatCapacity(2100)).toBe('2.1 TB')
  })
  it('states the machine total next to the per-GPU share', () => {
    expect(totalCapacityLine(150, 262.5, 8)).toBe('total 1.2 TB of 2.1 TB across 8 GPUs')
  })
})
