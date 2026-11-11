import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { estimatePerformance } from '@engines/performance'
import { reconstructPerformanceEstimate } from '@hooks/useInferenceCalculation'
import type { GPU } from '@utils/schemas'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it, vi } from 'vitest'

// The hook module imports the persisted store, which needs browser APIs jsdom lacks
vi.mock('@store/uiStore', () => ({ useUIStore: vi.fn() }))

const model = validateModels(modelsData).find((m) => m.id === 'meta-llama-llama-3.3-70b')
const h100 = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h100-80gb-sxm')
if (!model || !h100) throw new Error('fixture rows missing')

/** Runs the real worker handler, structured-clones its reply, and reconstructs it as the hook does */
async function roundTrip(gpu: GPU) {
  const post = vi.spyOn(self, 'postMessage').mockImplementation(() => undefined)
  try {
    await import('./calculation.worker')
    self.onmessage?.(
      new MessageEvent('message', {
        data: {
          type: 'CALCULATE_INFERENCE',
          payload: {
            model,
            gpu,
            quantization: 'fp8',
            sequenceLength: 8192,
            batchSize: 4,
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
    return reconstructPerformanceEstimate(response.payload.performance)
  } finally {
    post.mockRestore()
  }
}

describe('calculation worker round trip', () => {
  it('preserves expected through worker serialisation and hook reconstruction', async () => {
    const roundTripped = await roundTrip(h100)
    const direct = estimatePerformance({
      model,
      gpu: h100,
      quantization: 'fp8',
      sequenceLength: 8192,
      batchSize: 4,
      kvQuantization: 'fp16',
    })
    expect(roundTripped.expected.tokensPerSecond.equals(direct.expected.tokensPerSecond)).toBe(true)
    expect(roundTripped.expected.timeToFirstToken.equals(direct.expected.timeToFirstToken)).toBe(
      true,
    )
    expect(roundTripped.expected.prefillSeconds?.equals(direct.expected.prefillSeconds ?? 0)).toBe(
      true,
    )
  })

  it('keeps a null expected prefill null (GPU without FLOPS data)', async () => {
    const roundTripped = await roundTrip({
      ...h100,
      fp16_tflops: undefined,
      fp32_tflops: undefined,
    })
    expect(roundTripped.prefillSeconds).toBeNull()
    expect(roundTripped.expected.prefillSeconds).toBeNull()
    expect(roundTripped.expected.timeToFirstToken.isFinite()).toBe(true)
  })
})
