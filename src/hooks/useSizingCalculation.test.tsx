import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { act, renderHook } from '@testing-library/react'
import { validateGPUs, validateModels } from '@utils/schemas'
import { afterEach, describe, expect, it, vi } from 'vitest'

const model = validateModels(modelsData)[0]
const gpu = validateGPUs(gpusData)[0]
const state = {
  selectedModel: model,
  selectedGPU: gpu,
  quantization: 'fp16',
  kvQuantization: 'fp16',
  sequenceLength: 4096,
  concurrentUsers: 10,
  interNodeFabric: 'ethernet-800g',
  customFabric: null,
  kvTier: DEFAULT_KV_TIER,
}
vi.mock('@store/uiStore', () => ({
  useUIStore: (select: (s: typeof state) => unknown) => select(state),
}))

import { useSizingCalculation } from './useSizingCalculation'

class FakeWorker {
  static last: FakeWorker | null = null
  onmessage: ((e: { data: unknown }) => void) | null = null
  onerror: ((e: { message: string }) => void) | null = null
  constructor() {
    FakeWorker.last = this
  }
  postMessage() {}
  terminate() {}
}

afterEach(() => vi.unstubAllGlobals())

describe('useSizingCalculation', () => {
  // 2026-11-11, sizing final review M7: a failed calculation kept the previous result on screen
  it('clears the previous result when the calculation fails', async () => {
    vi.stubGlobal('Worker', FakeWorker)
    const { sizeDeployment } = await import('@engines/sizing')
    const { resolveFabricSpec } = await import('@engines/fabric')
    const { defaultMix } = await import('@engines/context-mix')
    const good = sizeDeployment({
      model,
      gpu,
      quantization: 'fp16',
      kvQuantization: 'fp16',
      sequenceLength: 4096,
      users: 10,
      fabric: resolveFabricSpec('ethernet-800g', null),
      kvTier: DEFAULT_KV_TIER,
      contextMix: defaultMix(4096),
      ttftTargetSeconds: null,
    })
    const { result } = renderHook(() => useSizingCalculation())
    const { serializeSizingResult } = await import('../workers/sizing-serialize')
    act(() => {
      FakeWorker.last?.onmessage?.({
        data: { type: 'SIZING_RESULT', result: serializeSizingResult(good) },
      })
    })
    expect(result.current.result).not.toBeNull()
    act(() => {
      FakeWorker.last?.onmessage?.({ data: { type: 'CALCULATION_ERROR', error: 'boom' } })
    })
    expect(result.current.result).toBeNull()
    expect(result.current.error).toBe('boom')
  })
})
