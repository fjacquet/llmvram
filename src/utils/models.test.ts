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

  it('should include Gemma 4 variants', () => {
    const gemma4Models = modelsData.filter((m) => m.name.includes('Gemma 4'))
    expect(gemma4Models.length).toBeGreaterThanOrEqual(3)
  })

  it('should include LLaMA 3 / Llama 4 variants', () => {
    const llamaModels = modelsData.filter(
      (m) => m.name.includes('LLaMA 3') || m.name.includes('Llama 4'),
    )
    expect(llamaModels.length).toBeGreaterThanOrEqual(3)
  })

  it('should include Mistral models', () => {
    const mistralModels = modelsData.filter((m) => m.name.includes('Mistral'))
    expect(mistralModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should store MoE models with TOTAL parameters, not active', () => {
    const qwenMoe = modelsData.find((m) => m.id === 'qwen-qwen3.6-35b-a3b')
    expect(qwenMoe).toBeDefined()
    expect(qwenMoe?.architecture).toBe('moe')
    // 36.0B total (all experts), NOT ~3B active - research pitfall #1
    expect(qwenMoe?.num_parameters_billion).toBeCloseTo(36.0, 1)
    expect(qwenMoe?.num_experts).toBe(256)
    expect(qwenMoe?.num_experts_per_token).toBe(8)
  })

  it('should include Qwen models', () => {
    const qwenModels = modelsData.filter((m) => m.name.includes('Qwen'))
    expect(qwenModels.length).toBeGreaterThanOrEqual(3)
  })

  it('should include Kimi models', () => {
    const kimiModels = modelsData.filter((m) => m.name.includes('Kimi'))
    expect(kimiModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include DeepSeek models', () => {
    const deepSeekModels = modelsData.filter((m) => m.name.includes('DeepSeek'))
    expect(deepSeekModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include Gemma models', () => {
    const gemmaModels = modelsData.filter((m) => m.name.includes('Gemma'))
    expect(gemmaModels.length).toBeGreaterThanOrEqual(2)
  })

  it('should include GLM models', () => {
    const glmModels = modelsData.filter((m) => m.name.includes('GLM'))
    expect(glmModels.length).toBeGreaterThanOrEqual(2)
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
    const qwen27b = modelsData.find((m) => m.id === 'qwen-qwen3.6-27b')

    // LLaMA 3.1 and Qwen3.6 use GQA with num_kv_heads < num_attention_heads
    if (llama3_8b) {
      expect(llama3_8b.num_kv_heads).toBeDefined()
      expect(llama3_8b.num_kv_heads).toBeLessThan(llama3_8b.num_attention_heads)
    }
    if (qwen27b) {
      expect(qwen27b.num_kv_heads).toBeDefined()
      expect(qwen27b.num_kv_heads).toBeLessThan(qwen27b.num_attention_heads)
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

  it('should be sorted alphabetically by name (code-unit order)', () => {
    const names = modelsData.map((m) => m.name)
    const sorted = [...names].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
    expect(names).toEqual(sorted)
  })

  it('sets kv_cache_elements_per_token on exactly the exotic-attention models', () => {
    const EXOTIC_KV: Record<string, number> = {
      'deepseek-r1': 35136,
      'deepseek-v4-flash': 24768,
      'deepseek-v4-pro': 35136,
      'moonshotai-kimi-k2-thinking': 35136,
      'moonshotai-kimi-k2-instruct': 35136,
      'moonshotai-kimi-k2.5': 35136,
      'moonshotai-kimi-linear-48b-a3b': 4032,
      'zai-org-glm-5.2': 44928,
      'minimax-m3': 61440,
      'minimax-m2.1': 126976,
      'minimax-m2.5': 126976,
      'minimax-m2.7': 126976,
      'nvidia-nemotron-3-nano-4b': 8192,
      'nvidia-nemotron-3-nano-30b-a3b': 3072,
      'nvidia-nemotron-3-super-120b-a12b': 4096,
      'nvidia-nemotron-3-ultra-550b-a55b': 6144,
    }
    for (const [id, expected] of Object.entries(EXOTIC_KV)) {
      const model = modelsData.find((m) => m.id === id) as
        | { kv_cache_elements_per_token?: number }
        | undefined
      expect(model, `model ${id} missing from database`).toBeDefined()
      expect(model?.kv_cache_elements_per_token, `wrong value for ${id}`).toBe(expected)
    }
    const withField = modelsData.filter(
      (m) => (m as { kv_cache_elements_per_token?: number }).kv_cache_elements_per_token,
    )
    expect(withField.length).toBe(Object.keys(EXOTIC_KV).length)
  })

  it('stores the corrected Nemotron Ultra layer count (108, not 128)', () => {
    const ultra = modelsData.find((m) => m.id === 'nvidia-nemotron-3-ultra-550b-a55b')
    expect(ultra?.num_hidden_layers).toBe(108)
  })
})
