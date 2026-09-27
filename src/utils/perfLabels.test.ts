import { describe, expect, it } from 'vitest'
import { firstTokenLabel } from './perfLabels'

describe('firstTokenLabel', () => {
  it('is a plain time to first token for one request', () => {
    expect(firstTokenLabel(1)).toBe('Time to first token')
  })

  it('says the figure is amortized over the batch for B > 1', () => {
    expect(firstTokenLabel(32)).toBe('Prefill per request (amortized over batch 32)')
  })
})
