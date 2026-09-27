import { DEFAULT_UI_CONFIG } from '@store/uiStore'
import { describe, expect, it, vi } from 'vitest'
import { countAdvancedChanges } from './advancedChanges'

vi.hoisted(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  })
})
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return { ...actual, persist: (config: unknown) => config }
})

describe('countAdvancedChanges', () => {
  it('is 0 at the defaults', () => {
    expect(countAdvancedChanges(DEFAULT_UI_CONFIG)).toBe(0)
  })

  it('counts each non-default advanced setting', () => {
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, batchSize: 8 })).toBe(1)
    expect(
      countAdvancedChanges({
        ...DEFAULT_UI_CONFIG,
        batchSize: 8,
        kvQuantization: 'fp8',
        offloadingEnabled: true,
      }),
    ).toBe(3)
    expect(
      countAdvancedChanges({
        ...DEFAULT_UI_CONFIG,
        kvTier: { ...DEFAULT_UI_CONFIG.kvTier, tier: 'network' },
      }),
    ).toBe(1)
  })

  it('ignores the fabric while there is one server, and inert inputs in training', () => {
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, interNodeFabric: 'ethernet-100g' })).toBe(0)
    expect(
      countAdvancedChanges({ ...DEFAULT_UI_CONFIG, numNodes: 2, interNodeFabric: 'ethernet-100g' }),
    ).toBe(1)
    expect(
      countAdvancedChanges({
        ...DEFAULT_UI_CONFIG,
        mode: 'training',
        kvQuantization: 'fp8',
        shardingStrategy: 'pipeline-parallel',
      }),
    ).toBe(0)
  })

  it('counts a customFabric change as the fabric group, gated by numNodes > 1', () => {
    const customFabric = { name: 'Lab switch', port_gbps: 25 }
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, customFabric })).toBe(0)
    expect(countAdvancedChanges({ ...DEFAULT_UI_CONFIG, numNodes: 2, customFabric })).toBe(1)
    expect(
      countAdvancedChanges({
        ...DEFAULT_UI_CONFIG,
        numNodes: 2,
        customFabric,
        interNodeFabric: 'ethernet-100g',
      }),
    ).toBe(1)
  })
})
