import { describe, expect, it } from 'vitest'
import { ZodError } from 'zod'
import {
  GPUSchema,
  ModelSchema,
  validateGPU,
  validateGPUs,
  validateModel,
  validateModels,
} from './schemas'

const validDatacenterGPU = {
  id: 'nvidia-h100-80gb-sxm',
  name: 'NVIDIA H100 80GB SXM',
  manufacturer: 'nvidia',
  vram_gb: 80,
  memory_bandwidth_gbps: 3350,
  memory_type: 'HBM3',
  bus_width: 5120,
  fp16_tflops: 989,
  fp32_tflops: 51,
  tdp_watts: 700,
  interconnect: 'nvlink-4',
  tier: 'datacenter',
}

const validAppleSiliconGPU = {
  id: 'apple-m4-max',
  name: 'Apple M4 Max',
  manufacturer: 'apple',
  vram_gb: 128,
  memory_bandwidth_gbps: 546,
  memory_type: 'Unified',
  bus_width: 0,
  tier: 'apple-silicon',
  interconnect: 'unified',
}

const validDenseModel = {
  id: 'llama-3.1-70b',
  name: 'LLaMA 3.1 70B',
  architecture: 'dense',
  num_parameters_billion: 70.6,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,
  intermediate_size: 28672,
}

const validMoEModel = {
  id: 'mixtral-8x7b',
  name: 'Mixtral 8x7B',
  architecture: 'moe',
  num_parameters_billion: 46.7,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_kv_heads: 8,
  intermediate_size: 14336,
  num_experts: 8,
  num_experts_per_token: 2,
}

describe('GPUSchema', () => {
  describe('valid GPU data', () => {
    it('should validate a complete datacenter GPU', () => {
      const result = GPUSchema.parse(validDatacenterGPU)
      expect(result.id).toBe('nvidia-h100-80gb-sxm')
      expect(result.vram_gb).toBe(80)
      expect(result.fp16_tflops).toBe(989)
    })

    it('should validate Apple Silicon with bus_width=0 and unified interconnect', () => {
      const result = GPUSchema.parse(validAppleSiliconGPU)
      expect(result.bus_width).toBe(0)
      expect(result.interconnect).toBe('unified')
      expect(result.tier).toBe('apple-silicon')
    })

    it('should validate a GPU with optional fields omitted', () => {
      const minimal = {
        id: 'test-gpu',
        name: 'Test GPU',
        manufacturer: 'nvidia',
        vram_gb: 24,
        memory_bandwidth_gbps: 1000,
        memory_type: 'GDDR6X',
        bus_width: 384,
        tier: 'consumer',
      }
      const result = GPUSchema.parse(minimal)
      expect(result.fp16_tflops).toBeUndefined()
      expect(result.fp32_tflops).toBeUndefined()
      expect(result.tdp_watts).toBeUndefined()
      expect(result.interconnect).toBeUndefined()
    })
  })

  describe('invalid GPU data', () => {
    it('should reject GPU with negative vram_gb', () => {
      expect(() => GPUSchema.parse({ ...validDatacenterGPU, vram_gb: -10 })).toThrow(ZodError)
    })

    it('should reject GPU with empty id', () => {
      expect(() => GPUSchema.parse({ ...validDatacenterGPU, id: '' })).toThrow(ZodError)
    })

    it('should reject GPU with invalid manufacturer', () => {
      expect(() => GPUSchema.parse({ ...validDatacenterGPU, manufacturer: 'qualcomm' })).toThrow(
        ZodError,
      )
    })

    it('should reject GPU with invalid tier', () => {
      expect(() => GPUSchema.parse({ ...validDatacenterGPU, tier: 'gaming' })).toThrow(ZodError)
    })

    it('should reject GPU with missing required fields', () => {
      expect(() => GPUSchema.parse({ id: 'test', name: 'Test' })).toThrow(ZodError)
    })

    it('should reject GPU with invalid interconnect', () => {
      expect(() => GPUSchema.parse({ ...validDatacenterGPU, interconnect: 'thunderbolt' })).toThrow(
        ZodError,
      )
    })
  })
})

describe('ModelSchema', () => {
  describe('valid model data', () => {
    it('should validate a dense model', () => {
      const result = ModelSchema.parse(validDenseModel)
      expect(result.architecture).toBe('dense')
      expect(result.num_parameters_billion).toBe(70.6)
      expect(result.num_kv_heads).toBe(8)
    })

    it('should validate an MoE model with expert fields', () => {
      const result = ModelSchema.parse(validMoEModel)
      expect(result.architecture).toBe('moe')
      expect(result.num_experts).toBe(8)
      expect(result.num_experts_per_token).toBe(2)
    })

    it('should validate a model without optional num_kv_heads', () => {
      const { num_kv_heads: _, ...modelWithoutKV } = validDenseModel
      const result = ModelSchema.parse(modelWithoutKV)
      expect(result.num_kv_heads).toBeUndefined()
    })
  })

  describe('invalid model data', () => {
    it('should reject model with negative parameters', () => {
      expect(() => ModelSchema.parse({ ...validDenseModel, num_parameters_billion: -5 })).toThrow(
        ZodError,
      )
    })

    it('should reject model with invalid architecture', () => {
      expect(() => ModelSchema.parse({ ...validDenseModel, architecture: 'hybrid' })).toThrow(
        ZodError,
      )
    })

    it('should reject model with missing hidden_size', () => {
      const { hidden_size: _, ...incomplete } = validDenseModel
      expect(() => ModelSchema.parse(incomplete)).toThrow(ZodError)
    })

    it('should reject model with non-integer num_hidden_layers', () => {
      expect(() => ModelSchema.parse({ ...validDenseModel, num_hidden_layers: 32.5 })).toThrow(
        ZodError,
      )
    })
  })
})

describe('validateGPU / validateModel helpers', () => {
  it('should return typed GPU for valid data', () => {
    const gpu = validateGPU(validDatacenterGPU)
    expect(gpu.name).toBe('NVIDIA H100 80GB SXM')
  })

  it('should throw ZodError for invalid GPU data', () => {
    expect(() => validateGPU({ id: 'bad' })).toThrow(ZodError)
  })

  it('should return typed Model for valid data', () => {
    const model = validateModel(validDenseModel)
    expect(model.name).toBe('LLaMA 3.1 70B')
  })

  it('should throw ZodError for invalid Model data', () => {
    expect(() => validateModel({ id: 'bad' })).toThrow(ZodError)
  })
})

describe('validateGPUs / validateModels array helpers', () => {
  it('should validate an array of valid GPUs', () => {
    const gpus = validateGPUs([validDatacenterGPU, validAppleSiliconGPU])
    expect(gpus).toHaveLength(2)
  })

  it('should reject if any GPU in array is invalid', () => {
    expect(() => validateGPUs([validDatacenterGPU, { id: 'bad' }])).toThrow(ZodError)
  })

  it('should handle empty GPU arrays', () => {
    const gpus = validateGPUs([])
    expect(gpus).toHaveLength(0)
  })

  it('should validate an array of valid Models', () => {
    const models = validateModels([validDenseModel, validMoEModel])
    expect(models).toHaveLength(2)
  })

  it('should reject if any Model in array is invalid', () => {
    expect(() => validateModels([validDenseModel, { id: 'bad' }])).toThrow(ZodError)
  })

  it('should handle empty Model arrays', () => {
    const models = validateModels([])
    expect(models).toHaveLength(0)
  })
})
