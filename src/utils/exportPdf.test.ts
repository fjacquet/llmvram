import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { useUIStore } from '@store/uiStore'
import { validateGPUs, validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { assumptionRows } from './exportPdf'

const model = validateModels(modelsData)[0]
const gpu = validateGPUs(gpusData).find((g) => g.id === 'nvidia-h200-141gb')
if (!model || !gpu) throw new Error('fixture not found')

const rows = (patch: Partial<ReturnType<typeof useUIStore.getState>>) =>
  Object.fromEntries(
    assumptionRows({ ...useUIStore.getState(), selectedModel: model, selectedGPU: gpu, ...patch }),
  )

describe('assumptionRows', () => {
  it('is empty until a model and GPU are chosen', () => {
    expect(assumptionRows({ ...useUIStore.getState(), selectedModel: null })).toEqual([])
  })

  it('inference: weights, KV precision, GPU count across servers', () => {
    const r = rows({ mode: 'inference', numGPUs: 8, numNodes: 2, quantization: 'fp8' })
    expect(r.Weights).toBe('FP8')
    expect(r['KV precision']).toBe('FP16')
    expect(r.GPUs).toBe('16 (8 per server x 2 servers)')
    expect(r['Network between servers']).toBeDefined()
  })

  it('sizing: users instead of a GPU count', () => {
    const r = rows({ mode: 'sizing', concurrentUsers: 500 })
    expect(r.Users).toBe('500')
    expect(r.GPUs).toBeUndefined()
  })

  it('training: method, optimizer and precision replace the quantization rows', () => {
    const r = rows({ mode: 'training', trainingMethod: 'qlora' })
    expect(r.Method).toBe('QLORA')
    expect(r.Weights).toBeUndefined()
  })
})
