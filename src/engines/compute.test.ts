import gpusData from '@data/gpus.json'
import { validateGPUs } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { createCustomGPU } from '@/types/gpu'
import { attentionComputeRate, linearComputeRate } from './compute'
import { PREFILL_MFU, PREFILL_MFU_BY_PRECISION } from './constants'
import type { QuantizationFormat } from './types'

const gpus = validateGPUs(gpusData)
function row(id: string) {
  const gpu = gpus.find((g) => g.id === id)
  if (!gpu) throw new Error(`missing GPU ${id}`)
  return gpu
}

const b200 = row('nvidia-b200-192gb')
const h100 = row('nvidia-h100-80gb-sxm')
const mi355x = row('amd-mi355x')
const rtxPro = row('nvidia-rtx-pro-6000-server')

describe('fixture rows', () => {
  it('match the literals used below', () => {
    expect([b200.fp16_tflops, b200.fp8_tflops, b200.fp4_tflops]).toEqual([2250, 4500, 9000])
    expect(h100.fp16_tflops).toBe(989)
    expect(h100.fp8_tflops).toBeGreaterThan(0)
    expect(h100.fp4_tflops).toBeUndefined()
    expect(mi355x.fp16_tflops).toBe(2516)
    expect(mi355x.fp4_tflops).toBeGreaterThan(0)
    expect(rtxPro.fp16_tflops).toBe(240)
    expect(rtxPro.fp8_tflops).toBeUndefined()
  })
})

describe('linearComputeRate', () => {
  // D9 fallback (Outcome B): no prefill-isolating FP4 source, so FP4 formats on an FP4 card
  // compute at the card's FP8 rate.
  it.each([
    ['nvidia-b200-192gb', 'fp8', 'fp8', 4500],
    ['nvidia-b200-192gb', 'nvfp4', 'fp8', 4500],
    ['nvidia-b200-192gb', 'mxfp4', 'fp8', 4500],
    ['nvidia-b200-192gb', 'bf16', 'fp16', 2250],
    ['nvidia-b200-192gb', 'awq', 'fp16', 2250],
    ['nvidia-b200-192gb', 'gguf-q4_k_m', 'fp16', 2250],
    ['nvidia-b200-192gb', 'nvfp6', 'fp16', 2250],
    ['nvidia-b200-192gb', 'int8', 'fp16', 2250],
    ['nvidia-b200-192gb', 'fp32', 'fp16', 2250],
    ['nvidia-h100-80gb-sxm', 'fp8', 'fp8', h100.fp8_tflops],
    ['nvidia-h100-80gb-sxm', 'nvfp4', 'fp16', 989], // Marlin W4A16, not FP8
    ['amd-mi355x', 'mxfp4', 'fp8', mi355x.fp8_tflops],
    ['amd-mi355x', 'nvfp4', 'fp16', 2516], // AMD runs MXFP4, not NVFP4
    ['nvidia-rtx-pro-6000-server', 'nvfp4', 'fp16', 240],
    ['nvidia-rtx-pro-6000-server', 'fp8', 'fp16', 240],
  ] as const)('%s + %s computes at %s', (id, format, precision, tflops) => {
    expect(linearComputeRate(row(id), format)).toEqual({ precision, tflops })
  })

  const formats: QuantizationFormat[] = ['fp8', 'nvfp4', 'mxfp4', 'bf16', 'awq', 'nvfp6']

  it('a custom GPU with only fp16 falls back to fp16 for every format', () => {
    const gpu = createCustomGPU({ name: 'x', vram_gb: 80, fp16_tflops: 500 })
    for (const f of formats) {
      expect(linearComputeRate(gpu, f)).toEqual({ precision: 'fp16', tflops: 500 })
    }
  })

  it('a GPU with only fp32 returns fp32 as the 16-bit rate (today fp16 ?? fp32)', () => {
    const gpu = createCustomGPU({ name: 'x', vram_gb: 80, fp32_tflops: 60 })
    for (const f of formats) {
      expect(linearComputeRate(gpu, f)).toEqual({ precision: 'fp16', tflops: 60 })
    }
  })

  it('returns tflops 0 when the GPU has no FLOPS data', () => {
    const gpu = createCustomGPU({ name: 'x', vram_gb: 80 })
    expect(linearComputeRate(gpu, 'fp8')).toEqual({ precision: 'fp16', tflops: 0 })
  })

  it('every database GPU keeps today’s 16-bit rate for 16-bit formats', () => {
    for (const gpu of gpus) {
      expect(linearComputeRate(gpu, 'bf16').tflops).toBe(gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0)
    }
  })
})

describe('attentionComputeRate', () => {
  it('attention always uses the 16-bit figure', () => {
    expect(attentionComputeRate(b200)).toEqual({ precision: 'fp16', tflops: 2250 })
    expect(linearComputeRate(b200, 'nvfp4').tflops).not.toBe(attentionComputeRate(b200).tflops)
  })
})

describe('PREFILL_MFU_BY_PRECISION', () => {
  it('PREFILL_MFU stays the fp16 entry of PREFILL_MFU_BY_PRECISION', () => {
    expect(PREFILL_MFU_BY_PRECISION.fp16).toBe(PREFILL_MFU)
    expect(PREFILL_MFU.toNumber()).toBe(0.45)
  })

  it('fp8 keeps 0.45 without a prefill-isolating source (spec D9)', () => {
    expect(PREFILL_MFU_BY_PRECISION.fp8.toNumber()).toBe(0.45)
  })
})
