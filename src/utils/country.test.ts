import { describe, expect, it } from 'vitest'
import { countryName } from './country'

describe('countryName', () => {
  it('maps an ISO 3166-1 alpha-2 code to its English name', () => {
    expect(countryName('FR')).toBe('France')
    expect(countryName('CN')).toBe('China')
    expect(countryName('US')).toBe('United States')
  })
})
