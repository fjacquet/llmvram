import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { estimatePerformance } from '@engines/performance'
import { reconstructPerformanceEstimate } from '@hooks/useInferenceCalculation'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it, vi } from 'vitest'

// The hook module imports the persisted store, which needs browser APIs jsdom lacks
vi.mock('@store/uiStore', () => ({ useUIStore: vi.fn() }))

describe('calculation worker round trip', () => {
  it('preserves expected through worker serialisation and hook reconstruction', async () => {
    const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.3-70b')
    const gpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
    if (!model || !gpu) throw new Error('fixture rows missing')
    const post = vi.spyOn(self, 'postMessage').mockImplementation(() => undefined)
    await import('./calculation.worker')
    const params = { model, gpu, quantization: 'fp8' as const, sequenceLength: 8192, batchSize: 4 }
    self.onmessage?.(
      new MessageEvent('message', {
        data: {
          type: 'CALCULATE_INFERENCE',
          payload: {
            ...params,
            kvQuantization: 'fp16',
            numGPUs: 1,
            numNodes: 1,
            interNodeFabric: 'infiniband-ndr',
            customFabric: null,
            concurrentUsers: 4,
            shardingStrategy: 'tensor-parallel',
            offloadingEnabled: false,
            offloadTarget: 'cpu',
            offloadMode: 'percentage',
            offloadPercentage: 0,
            offloadLayers: 0,
            kvCacheOffload: false,
          },
        },
      }),
    )
    const response = structuredClone(post.mock.calls.at(-1)?.[0])
    expect(response.type).toBe('CALCULATION_RESULT')
    const roundTripped = reconstructPerformanceEstimate(response.payload.performance)
    const direct = estimatePerformance({ ...params, kvQuantization: 'fp16' })
    expect(roundTripped.expected.tokensPerSecond.equals(direct.expected.tokensPerSecond)).toBe(true)
    expect(roundTripped.expected.timeToFirstToken.equals(direct.expected.timeToFirstToken)).toBe(
      true,
    )
    expect(roundTripped.expected.prefillSeconds?.equals(direct.expected.prefillSeconds ?? 0)).toBe(
      true,
    )
    post.mockRestore()
  })
})
