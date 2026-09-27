import modelsData from '@data/models.json'
import {
  calculateModelWeightVRAM,
  effectiveBytesPerParameter,
  getBytesPerParameter,
  weightRef,
  weightSource,
} from '@engines/quantization'
import type { QuantizationFormat } from '@engines/types'
import type { Model } from '@utils/schemas'
import { validateModels } from '@utils/schemas'
import { describe, expect, it } from 'vitest'

describe('getBytesPerParameter', () => {
  it('should return correct bytes for all quantization formats', () => {
    const formatTests: Array<[QuantizationFormat, number]> = [
      // Float formats
      ['fp32', 4.0],
      ['fp16', 2.0],
      ['bf16', 2.0],
      ['fp8', 1.0], // E4M3/E5M2, as DeepSeek, Kimi K2 and MiniMax ship
      ['mxfp4', 0.53125], // OCP MX: 4-bit E2M1 + one 8-bit scale per 32 → 4.25 bpp

      // NVIDIA FP formats
      ['nvfp6', 0.75],
      ['nvfp4', 0.5625],

      // Integer formats
      ['int8', 1.0],
      ['int4', 0.5625], // 4-bit + 16-bit scale per group of 32 → 4.5 bpp
      ['nf4', 0.5],

      // Compressed formats with overhead
      ['gptq', 0.52], // 4-bit + 16-bit scale and 4-bit zero per group of 128
      ['awq', 0.52], // 4-bit + 16-bit scale and 4-bit zero per group of 128

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
      ['gguf-q2_k', 0.366], // ~2.93 bpp
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
    it('should show GPTQ has overhead over pure 4-bit (0.5 bytes)', () => {
      const gptqBytes = getBytesPerParameter('gptq').toNumber()
      const pure4bit = 0.5

      // GPTQ should be > 0.5 but <= 0.65 (conservative upper bound)
      expect(gptqBytes).toBeGreaterThan(pure4bit)
      expect(gptqBytes).toBeLessThanOrEqual(0.65)

      // 4-bit + group-128 scale and zero
      expect(gptqBytes).toBeCloseTo(0.52, 2)
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

    it('should calculate 70B GPTQ model as ~33.90 GB', () => {
      const vram = calculateModelWeightVRAM(70.0, 'gptq')

      // 70B * 0.52 bytes = 36.4GB, then 36.4e9 / 1024^3 = ~33.90 GB
      expect(vram.toNumber()).toBeCloseTo(33.9, 1)
    })

    it('should calculate Mixtral 8x7B FP16 using TOTAL params (46.7B) as ~86.97 GB', () => {
      // MoE models: Use TOTAL parameters, not active (46.7B, not 13B)
      // All expert weights must fit in VRAM
      const vram = calculateModelWeightVRAM(46.7, 'fp16')

      // 46.7B * 2 bytes = 93.4GB, then 93.4e9 / 1024^3 = ~86.97 GB
      expect(vram.toNumber()).toBeCloseTo(86.97, 1)
    })

    it('should calculate 13B INT4 model as ~6.81 GB', () => {
      const vram = calculateModelWeightVRAM(13.0, 'int4')

      // 13B * 0.5625 bytes = 7.3125GB, then 7.3125e9 / 1024^3 = ~6.81 GB
      expect(vram.toNumber()).toBeCloseTo(6.81, 1)
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

      // 1.5B * 0.5625 bytes = 0.84375GB, then 0.84375e9 / 1024^3 = ~0.786 GB
      expect(vram.toNumber()).toBeCloseTo(0.786, 1)
    })
  })

  describe('quantization format impact', () => {
    it('should show FP32 uses 2x memory of FP16', () => {
      const fp32 = calculateModelWeightVRAM(7.0, 'fp32')
      const fp16 = calculateModelWeightVRAM(7.0, 'fp16')

      const ratio = fp32.div(fp16).toNumber()
      expect(ratio).toBeCloseTo(2.0, 2)
    })

    it('should show GPTQ reduces VRAM by ~3.85x vs FP16 (not 4x)', () => {
      const gptq = calculateModelWeightVRAM(70.0, 'gptq')
      const fp16 = calculateModelWeightVRAM(70.0, 'fp16')

      const ratio = fp16.div(gptq).toNumber()

      // GPTQ is 0.52 bytes, FP16 is 2.0 bytes
      // Ratio should be 2.0 / 0.52 = 3.846x (not 4x due to overhead)
      expect(ratio).toBeCloseTo(3.846, 1)
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

describe('measured weight_refs', () => {
  const gemma: Model = {
    id: 'google-gemma-4-31b',
    name: 'Gemma 4 31B',
    architecture: 'dense',
    num_parameters_billion: 32.7,
    hidden_size: 5376,
    num_hidden_layers: 60,
    num_attention_heads: 32,
    intermediate_size: 21504,
    weight_refs: { nvfp4: { repo: 'nvidia/Gemma-4-31B-IT-NVFP4', gib: 30.4 } },
  }

  it('returns the measured checkpoint size when the format has a ref', () => {
    expect(calculateModelWeightVRAM(32.7, 'nvfp4', gemma).toNumber()).toBeCloseTo(30.4, 9)
  })

  it('scales a parameter subset by the measured bytes per parameter (decode)', () => {
    const half = calculateModelWeightVRAM(32.7 / 2, 'nvfp4', gemma).toNumber()
    expect(half).toBeCloseTo(15.2, 9)
  })

  it('falls back to the constant for a format without a ref', () => {
    expect(effectiveBytesPerParameter('int4', gemma).toNumber()).toBe(0.5625)
    expect(calculateModelWeightVRAM(32.7, 'int4', gemma).toString()).toBe(
      calculateModelWeightVRAM(32.7, 'int4').toString(),
    )
  })

  it('leaves models without refs (custom, URL-restored) on the constants', () => {
    const custom: Model = { ...gemma, weight_refs: undefined }
    expect(calculateModelWeightVRAM(32.7, 'nvfp4', custom).toString()).toBe(
      calculateModelWeightVRAM(32.7, 'nvfp4').toString(),
    )
  })

  it('names the measured source, or null when estimated', () => {
    expect(weightSource(gemma, 'nvfp4')).toBe('nvidia/Gemma-4-31B-IT-NVFP4')
    expect(weightSource(gemma, 'fp8')).toBeNull()
  })
})

// I-3 (spec §3): fp16 and bf16 are the same size, but the data only ever measures bf16
// checkpoints (35 bf16 refs, 0 fp16), so fp16 — the store's default format — must fall
// back to the bf16 ref rather than showing "estimated" for every model.
describe('fp16 and bf16 share a measured ref', () => {
  const base: Model = {
    id: 'google-gemma-4-31b',
    name: 'Gemma 4 31B',
    architecture: 'dense',
    num_parameters_billion: 32.7,
    hidden_size: 5376,
    num_hidden_layers: 60,
    num_attention_heads: 32,
    intermediate_size: 21504,
  }
  const bf16Only: Model = {
    ...base,
    weight_refs: { bf16: { repo: 'google/gemma-4-31B-it', gib: 58.25 } },
  }
  const fp16Only: Model = {
    ...base,
    weight_refs: { fp16: { repo: 'some-org/gemma-4-31b-fp16', gib: 58.25 } },
  }

  it('fp16 uses the bf16 ref when only bf16 is measured', () => {
    expect(effectiveBytesPerParameter('fp16', bf16Only).toString()).toBe(
      effectiveBytesPerParameter('bf16', bf16Only).toString(),
    )
    expect(calculateModelWeightVRAM(32.7, 'fp16', bf16Only).toString()).toBe(
      calculateModelWeightVRAM(32.7, 'bf16', bf16Only).toString(),
    )
  })

  it('bf16 uses the fp16 ref when only fp16 is measured', () => {
    expect(effectiveBytesPerParameter('bf16', fp16Only).toString()).toBe(
      effectiveBytesPerParameter('fp16', fp16Only).toString(),
    )
  })

  it('leaves other formats unaffected by the fp16/bf16 twin', () => {
    // bf16Only has no nvfp4 ref, and fp16/bf16 are not nvfp4's twin: falls to the constant.
    expect(effectiveBytesPerParameter('nvfp4', bf16Only).toString()).toBe(
      effectiveBytesPerParameter('nvfp4').toString(),
    )
  })

  it('weightSource(fp16) names the bf16 repo', () => {
    expect(weightSource(bf16Only, 'fp16')).toBe('google/gemma-4-31B-it')
    expect(weightSource(fp16Only, 'bf16')).toBe('some-org/gemma-4-31b-fp16')
  })
})

describe('source-derived fallback constants', () => {
  it('int4 carries a 16-bit scale per group of 32 (Kimi K2 quantization_config)', () => {
    expect(effectiveBytesPerParameter('int4').toNumber()).toBe((4 + 16 / 32) / 8)
  })

  it('AWQ and GPTQ carry a 16-bit scale and 4-bit zero per group of 128', () => {
    expect(effectiveBytesPerParameter('awq').toNumber()).toBe(0.52)
    expect(effectiveBytesPerParameter('gptq').toNumber()).toBe(0.52)
  })

  it('GGUF Q2_K matches the median published file (2.93 bpp)', () => {
    expect(effectiveBytesPerParameter('gguf-q2_k').toNumber()).toBe(0.366)
  })

  it('NVFP4 is E2M1 plus an FP8 scale per 16 values', () => {
    expect(effectiveBytesPerParameter('nvfp4').toNumber()).toBe((4 + 8 / 16) / 8)
  })
})

describe('weightRef', () => {
  it('resolves the fp16/bf16 twin and exposes high_precision', () => {
    const m = validateModels(modelsData).find((x) => x.id === 'moonshotai-kimi-k3')
    if (!m) throw new Error('fixture')
    expect(weightRef('mxfp4', m)?.repo).toBe('moonshotai/Kimi-K3')
    expect(weightRef('int8', m)).toBeUndefined()
  })
})
