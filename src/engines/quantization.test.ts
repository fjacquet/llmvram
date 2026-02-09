import { calculateModelWeightVRAM, getBytesPerParameter } from '@engines/quantization'
import type { QuantizationFormat } from '@engines/types'
import { describe, expect, it } from 'vitest'

describe('getBytesPerParameter', () => {
  it('should return correct bytes for all quantization formats', () => {
    const formatTests: Array<[QuantizationFormat, number]> = [
      // Float formats
      ['fp32', 4.0],
      ['fp16', 2.0],
      ['bf16', 2.0],

      // NVIDIA FP formats
      ['nvfp6', 0.75],
      ['nvfp4', 0.5625],

      // Integer formats
      ['int8', 1.0],
      ['int4', 0.5],
      ['nf4', 0.5],

      // Compressed formats with overhead
      ['gptq', 0.6],
      ['awq', 0.6],

      // GGUF formats (empirical bpp from llama.cpp block sizes)
      ['gguf-q8_0', 1.0625], // 8.5 bpp
      ['gguf-q6_k', 0.82], // 6.5625 bpp
      ['gguf-q5_k_s', 0.6875], // 5.5 bpp
      ['gguf-q5_k_m', 0.711], // 5.69 bpp
      ['gguf-q5_0', 0.6875], // 5.5 bpp
      ['gguf-q4_k_s', 0.5625], // 4.5 bpp
      ['gguf-q4_k_m', 0.6], // 4.8 bpp
      ['gguf-q4_0', 0.5625], // 4.5 bpp
      ['gguf-q3_k_l', 0.516], // 4.13 bpp
      ['gguf-q3_k_m', 0.489], // 3.9 bpp
      ['gguf-q3_k_s', 0.43], // 3.44 bpp
      ['gguf-q2_k', 0.328], // 2.625 bpp
    ]

    for (const [format, expectedBytes] of formatTests) {
      const result = getBytesPerParameter(format)
      expect(result.toNumber()).toBeCloseTo(expectedBytes, 4)
    }
  })

  it('should return Decimal instances, not primitive numbers', () => {
    const result = getBytesPerParameter('fp16')
    expect(result.constructor.name).toBe('Decimal')
  })

  describe('GPTQ and AWQ overhead verification', () => {
    it('should show GPTQ has 1.2x overhead over pure 4-bit (0.5 bytes)', () => {
      const gptqBytes = getBytesPerParameter('gptq').toNumber()
      const pure4bit = 0.5

      // GPTQ should be > 0.5 but <= 0.65 (conservative upper bound)
      expect(gptqBytes).toBeGreaterThan(pure4bit)
      expect(gptqBytes).toBeLessThanOrEqual(0.65)

      // Should be approximately 0.6 (0.5 * 1.2)
      expect(gptqBytes).toBeCloseTo(0.6, 2)
    })

    it('should show AWQ has same overhead as GPTQ', () => {
      const awqBytes = getBytesPerParameter('awq').toNumber()
      const gptqBytes = getBytesPerParameter('gptq').toNumber()

      expect(awqBytes).toBeCloseTo(gptqBytes, 4)
    })
  })

  describe('GGUF empirical bits-per-parameter', () => {
    it('should show Q4_K_M uses 4.8 bpp (not 4.0)', () => {
      const q4kmBytes = getBytesPerParameter('gguf-q4_k_m').toNumber()
      const expected4p8bpp = 4.8 / 8

      expect(q4kmBytes).toBeCloseTo(expected4p8bpp, 4)
      expect(q4kmBytes).toBeGreaterThan(0.5) // More than pure 4-bit
    })

    it('should show Q8_0 uses 8.5 bpp (not 8.0)', () => {
      const q8Bytes = getBytesPerParameter('gguf-q8_0').toNumber()
      const expected8p5bpp = 8.5 / 8

      expect(q8Bytes).toBeCloseTo(expected8p5bpp, 4)
      expect(q8Bytes).toBeGreaterThan(1.0) // More than pure int8
    })

    it('should show progressive bit depths across GGUF Q4/Q5/Q6/Q8', () => {
      const q4 = getBytesPerParameter('gguf-q4_k_m').toNumber()
      const q5 = getBytesPerParameter('gguf-q5_k_m').toNumber()
      const q6 = getBytesPerParameter('gguf-q6_k').toNumber()
      const q8 = getBytesPerParameter('gguf-q8_0').toNumber()

      // Should be monotonically increasing
      expect(q5).toBeGreaterThan(q4)
      expect(q6).toBeGreaterThan(q5)
      expect(q8).toBeGreaterThan(q6)
    })
  })

  describe('Float format consistency', () => {
    it('should return exactly 4.0 for FP32', () => {
      const fp32Bytes = getBytesPerParameter('fp32').toNumber()
      expect(fp32Bytes).toBe(4.0)
    })

    it('should return exactly 2.0 for both FP16 and BF16', () => {
      const fp16Bytes = getBytesPerParameter('fp16').toNumber()
      const bf16Bytes = getBytesPerParameter('bf16').toNumber()

      expect(fp16Bytes).toBe(2.0)
      expect(bf16Bytes).toBe(2.0)
    })
  })
})

describe('calculateModelWeightVRAM', () => {
  describe('known reference calculations', () => {
    it('should calculate 7B FP16 model as ~13.04 GB', () => {
      const vram = calculateModelWeightVRAM(7.0, 'fp16')

      // 7B * 2 bytes = 14GB, then 14e9 / 1024^3 = ~13.04 GB
      expect(vram.toNumber()).toBeCloseTo(13.04, 1)
    })

    it('should calculate 70B GPTQ model as ~39.12 GB', () => {
      const vram = calculateModelWeightVRAM(70.0, 'gptq')

      // 70B * 0.6 bytes = 42GB, then 42e9 / 1024^3 = ~39.12 GB
      expect(vram.toNumber()).toBeCloseTo(39.12, 1)
    })

    it('should calculate Mixtral 8x7B FP16 using TOTAL params (46.7B) as ~86.97 GB', () => {
      // MoE models: Use TOTAL parameters, not active (46.7B, not 13B)
      // All expert weights must fit in VRAM
      const vram = calculateModelWeightVRAM(46.7, 'fp16')

      // 46.7B * 2 bytes = 93.4GB, then 93.4e9 / 1024^3 = ~86.97 GB
      expect(vram.toNumber()).toBeCloseTo(86.97, 1)
    })

    it('should calculate 13B INT4 model as ~6.06 GB', () => {
      const vram = calculateModelWeightVRAM(13.0, 'int4')

      // 13B * 0.5 bytes = 6.5GB, then 6.5e9 / 1024^3 = ~6.05 GB
      expect(vram.toNumber()).toBeCloseTo(6.06, 1)
    })
  })

  it('should return Decimal instance, not primitive number', () => {
    const vram = calculateModelWeightVRAM(7.0, 'fp16')
    expect(vram.constructor.name).toBe('Decimal')
  })

  describe('Decimal.js precision verification', () => {
    it('should not produce floating-point artifacts (no 13.0000000001)', () => {
      const vram = calculateModelWeightVRAM(7.0, 'fp16')
      const vramStr = vram.toString()

      // Should be a clean decimal, not scientific notation
      expect(vramStr).not.toMatch(/e[+-]\d+/) // No scientific notation
      expect(vramStr.split('.')[1]?.length ?? 0).toBeLessThan(25) // Decimal.js default precision
    })

    it('should handle large models (405B) without precision loss', () => {
      const vram = calculateModelWeightVRAM(405.0, 'fp16')

      // 405B * 2 bytes = 810GB, then 810e9 / 1024^3 = ~754.37 GB
      expect(vram.toNumber()).toBeCloseTo(754.37, 1)
    })

    it('should handle small quantized models (1.5B INT4) accurately', () => {
      const vram = calculateModelWeightVRAM(1.5, 'int4')

      // 1.5B * 0.5 bytes = 0.75GB, then 0.75e9 / 1024^3 = ~0.70 GB
      expect(vram.toNumber()).toBeCloseTo(0.7, 1)
    })
  })

  describe('quantization format impact', () => {
    it('should show FP32 uses 2x memory of FP16', () => {
      const fp32 = calculateModelWeightVRAM(7.0, 'fp32')
      const fp16 = calculateModelWeightVRAM(7.0, 'fp16')

      const ratio = fp32.div(fp16).toNumber()
      expect(ratio).toBeCloseTo(2.0, 2)
    })

    it('should show GPTQ reduces VRAM by ~3.3x vs FP16 (not 4x)', () => {
      const gptq = calculateModelWeightVRAM(70.0, 'gptq')
      const fp16 = calculateModelWeightVRAM(70.0, 'fp16')

      const ratio = fp16.div(gptq).toNumber()

      // GPTQ is 0.6 bytes, FP16 is 2.0 bytes
      // Ratio should be 2.0 / 0.6 = 3.33x (not 4x due to overhead)
      expect(ratio).toBeCloseTo(3.33, 1)
      expect(ratio).toBeLessThan(4.0) // Should NOT be 4x
    })
  })

  describe('edge cases', () => {
    it('should handle fractional billion params (MoE models like 46.7B)', () => {
      const vram = calculateModelWeightVRAM(46.7, 'fp16')

      // Should not crash or produce NaN
      expect(vram.isNaN()).toBe(false)
      expect(vram.isFinite()).toBe(true)
      expect(vram.toNumber()).toBeGreaterThan(0)
    })

    it('should handle very small models (0.5B)', () => {
      const vram = calculateModelWeightVRAM(0.5, 'fp16')

      // 0.5B * 2 bytes = 1GB, then 1e9 / 1024^3 = ~0.93 GB
      expect(vram.toNumber()).toBeCloseTo(0.93, 1)
    })
  })
})
