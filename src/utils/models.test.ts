import modelsData from '@data/models.json'
import { describe, expect, it } from 'vitest'
import { ModelSchema, validateModels } from './schemas'

describe('Model Database Validation', () => {
  it('should have at least 30 models', () => {
    expect(modelsData.length).toBeGreaterThanOrEqual(30)
  })

  it('should validate all model entries against schema', () => {
    const result = validateModels(modelsData)
    expect(result.length).toBe(modelsData.length)
  })

  it('should include LLaMA 2 variants', () => {
    const llama2Models = modelsData.filter((m) => m.name.includes('LLaMA 2'))
    expect(llama2Models.length).toBeGreaterThanOrEqual(3)
  })

  it('should include LLaMA 3 variants', () => {
    const llama3Models = modelsData.filter((m) => m.name.includes('LLaMA 3'))
    expect(llama3Models.length).toBeGreaterThanOrEqual(3)
  })

  it('should include Mistral models', () => {
    const mistralModels = modelsData.filter((m) => m.name.includes('Mistral'))
    expect(mistralModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include Mixtral MoE models with correct total parameters', () => {
    const mixtral8x7b = modelsData.find((m) => m.id.includes('mixtral-8x7b'))
    expect(mixtral8x7b).toBeDefined()
    expect(mixtral8x7b?.architecture).toBe('moe')
    // Must be 46.7B (total), NOT 13B (active) - research pitfall #1
    expect(mixtral8x7b?.num_parameters_billion).toBeCloseTo(46.7, 1)
    expect(mixtral8x7b?.num_experts).toBe(8)
    expect(mixtral8x7b?.num_experts_per_token).toBe(2)
  })

  it('should include Qwen models', () => {
    const qwenModels = modelsData.filter((m) => m.name.includes('Qwen'))
    expect(qwenModels.length).toBeGreaterThanOrEqual(3)
  })

  it('should include Phi models', () => {
    const phiModels = modelsData.filter((m) => m.name.includes('Phi'))
    expect(phiModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include DeepSeek models', () => {
    const deepSeekModels = modelsData.filter((m) => m.name.includes('DeepSeek'))
    expect(deepSeekModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include Gemma models', () => {
    const gemmaModels = modelsData.filter((m) => m.name.includes('Gemma'))
    expect(gemmaModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include Command-R models', () => {
    const commandRModels = modelsData.filter((m) => m.name.includes('Command-R'))
    expect(commandRModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should have valid parameter counts', () => {
    modelsData.forEach((model) => {
      expect(model.num_parameters_billion).toBeGreaterThan(0)
      expect(model.num_parameters_billion).toBeLessThan(2000) // Sanity check (Kimi K2 has 1026B total MoE params)
    })
  })

  it('should have valid architecture configurations', () => {
    modelsData.forEach((model) => {
      expect(model.hidden_size).toBeGreaterThan(0)
      expect(model.num_hidden_layers).toBeGreaterThan(0)
      expect(model.num_attention_heads).toBeGreaterThan(0)
      expect(model.intermediate_size).toBeGreaterThan(0)
    })
  })

  it('should correctly identify MoE vs dense architectures', () => {
    const moeModels = modelsData.filter((m) => m.architecture === 'moe')
    const denseModels = modelsData.filter((m) => m.architecture === 'dense')

    // Should have both types
    expect(moeModels.length).toBeGreaterThan(0)
    expect(denseModels.length).toBeGreaterThan(0)

    // MoE models must have expert fields
    moeModels.forEach((model) => {
      expect(model.num_experts).toBeDefined()
      expect(model.num_experts_per_token).toBeDefined()
    })
  })

  it('should specify num_kv_heads for GQA models', () => {
    const llama3_8b = modelsData.find((m) => m.id.includes('llama-3.1-8b'))
    const mistral7b = modelsData.find((m) => m.id.includes('mistral-7b'))

    // LLaMA 3.1 and Mistral use GQA with num_kv_heads < num_attention_heads
    if (llama3_8b) {
      expect(llama3_8b.num_kv_heads).toBeDefined()
      expect(llama3_8b.num_kv_heads).toBeLessThan(llama3_8b.num_attention_heads)
    }
    if (mistral7b) {
      expect(mistral7b.num_kv_heads).toBeDefined()
      expect(mistral7b.num_kv_heads).toBeLessThan(mistral7b.num_attention_heads)
    }
  })

  it('should validate individual model entry structure', () => {
    const sampleModel = modelsData[0]
    const result = ModelSchema.safeParse(sampleModel)

    expect(result.success).toBe(true)
    if (result.success) {
      expect(result.data).toHaveProperty('id')
      expect(result.data).toHaveProperty('name')
      expect(result.data).toHaveProperty('architecture')
      expect(result.data).toHaveProperty('num_parameters_billion')
    }
  })

  it('should have unique model IDs', () => {
    const ids = modelsData.map((model) => model.id)
    const uniqueIds = new Set(ids)
    expect(uniqueIds.size).toBe(ids.length)
  })

  it('should have consistent hidden_size and intermediate_size ratio', () => {
    // Most models have intermediate_size ≈ 4x hidden_size (FFN expansion)
    const denseModels = modelsData.filter((m) => m.architecture === 'dense')
    denseModels.forEach((model) => {
      const ratio = model.intermediate_size / model.hidden_size
      // Allow some variation (2x to 8x is reasonable)
      expect(ratio).toBeGreaterThan(2)
      expect(ratio).toBeLessThanOrEqual(8)
    })
  })

  it('should have num_kv_heads <= num_attention_heads', () => {
    modelsData.forEach((model) => {
      if (model.num_kv_heads !== undefined) {
        expect(model.num_kv_heads).toBeLessThanOrEqual(model.num_attention_heads)
      }
    })
  })
})
