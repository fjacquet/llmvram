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
    // ModelSchema already enforces `> 0`; this catches a unit slip (params written in
    // millions, or a stray digit). Stated as a rationale, not as "bigger than today's
    // biggest model" — the latter needs raising every time a larger model lands.
    modelsData.forEach((model) => {
      expect(model.num_parameters_billion).toBeLessThan(100_000)
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

  // Known-good KV sizes. Each value is what vLLM allocates (read from the model's
  // vllm/model_executor/models/<arch>.py against its HF config.json) AND agrees with a
  // second source: the per-layer cache shapes HF transformers builds on the meta device,
  // or a published figure. Nothing here is re-derived by formula. Audit of 2026-09-26.
  it('sets kv_cache_elements_per_token on exactly the verified models', () => {
    const EXOTIC_KV: Record<string, number> = {
      'deepseek-r1': 35136, // MLA: 61 x (512 + 64). DeepSeek-V2 paper formula
      // DeepSeek V4: compressed MLA. C4 layers keep 1 entry per 4 tokens, C128 layers 1 per
      // 128, C4 layers add a 128-dim indexer key per 4 tokens. vLLM deepseek_v4; the paper's
      // "~2% of a BF16 GQA8 baseline" matches (2.2%). Sliding part in EXOTIC_SLIDING.
      'deepseek-v4-flash': 3440, // 21x128 + 20x4 + 21x32
      'deepseek-v4-pro': 4924, // 30x128 + 31x4 + 30x32
      'moonshotai-kimi-k2-thinking': 35136,
      'moonshotai-kimi-k2-instruct': 35136,
      'moonshotai-kimi-k2.5': 35136,
      'moonshotai-kimi-k2.6': 35136, // MLA: 61 layers x (kv_lora_rank 512 + qk_rope 64)
      'moonshotai-kimi-k2.7-code': 35136, // MLA: 61 x 576
      'moonshotai-kimi-k3': 13824, // MLA: 24 of 93 layers x 576 (69 KDA layers cache-free)
      'moonshotai-kimi-linear-48b-a3b': 4032,
      'inclusionai-ling-3.0-flash': 4032, // MLA: 7 of 42 layers x 576
      'inclusionai-ling-3.0-tiny': 3456, // MLA: 6 of 24 layers x 576
      'liquidai-lfm2.5-2.6b': 8192, // GQA: 8 of 30 layers x (8 kv heads x 64 head_dim x 2)
      'qwen-qwen3.8-27b': 32768, // GQA: 16 of 64 layers x (4 kv heads x 256 head_dim x 2)
      'qwen-qwen3.8-2.4t-a95b': 47104, // GQA: 23 of 92 layers x 2048
      'qwen-qwen3.6-27b': 32768, // 16 full of 64 (48 Gated DeltaNet), 4 kv x 256
      'qwen-qwen3.6-35b-a3b': 10240, // 10 full of 40, 2 kv x 256
      'qwen-qwen3-235b-a22b': 96256, // 94 x 4 kv x head_dim 128 (not hidden/heads = 64) x 2
      'zai-org-glm-4.7': 188416, // 92 x 8 kv x head_dim 128 (not hidden/heads) x 2
      'zai-org-glm-4.7-flash': 27072, // MLA: 47 x 576
      'zai-org-glm-5.2': 44928, // MLA: 78 x 576 (DSA indexer, ~3%, not included)
      'minimax-m3': 68736, // 60 x 4 kv x 128 x 2 + 128-dim indexer key on 57 sparse layers
      'minimax-m2.1': 126976,
      'minimax-m2.5': 126976,
      'minimax-m2.7': 126976,
      'mistralai-mistral-small-4-119b': 11520, // MLA: 36 x (kv_lora_rank 256 + rope 64)
      'mistralai-magistral-small-2507': 81920, // 40 x 8 kv x head_dim 128 (not 160) x 2
      'mistralai-ministral-3-14b-reasoning': 81920, // same shape as Magistral
      'nvidia-nemotron-3-nano-4b': 8192,
      'nvidia-nemotron-3-nano-30b-a3b': 3072,
      'nvidia-nemotron-3.5-lightning-30b-a3b': 3072,
      'nvidia-nemotron-3-super-120b-a12b': 4096,
      'nvidia-nemotron-3-ultra-550b-a55b': 6144,
      // Sliding-window models: full-attention layers only; window layers in EXOTIC_SLIDING
      'google-gemma-3-1b': 2048, // 4 global of 26 (every 6th), 1 kv x 256 x 2
      'google-gemma-3-4b': 10240, // 5 global of 34, 4 kv x 256 x 2
      'google-gemma-3-12b': 32768, // 8 global of 48, 8 kv x 256 x 2
      'google-gemma-3-27b': 40960, // 10 global of 62, 16 kv x 128 x 2
      // Gemma 4 global layers: global_head_dim 512, num_global_key_value_heads. vLLM caches
      // K and V separately even though attention_k_eq_v makes them equal.
      'google-gemma-4-12b': 8192, // 8 global x 1 kv x 512 x 2
      'google-gemma-4-26b-a4b': 10240, // 5 global x 2 kv x 512 x 2
      'google-gemma-4-31b': 40960, // 10 global x 4 kv x 512 x 2
      'openai-gpt-oss-120b': 18432, // 18 dense of 36, 8 kv x head_dim 64 x 2
      'openai-gpt-oss-20b': 12288, // 12 dense of 24
      'meta-llama-llama-4-scout': 24576, // 12 global NoPE of 48, 8 kv x 128 x 2
      'meta-llama-llama-4-maverick': 24576,
    }
    // One assertion in both directions: every id in the map carries that value, and no
    // model outside the map carries the field at all. Names the offending model on failure.
    const actual = Object.fromEntries(
      modelsData
        .filter((m) => 'kv_cache_elements_per_token' in m)
        .map((m) => [
          m.id,
          (m as { kv_cache_elements_per_token: number }).kv_cache_elements_per_token,
        ]),
    )
    expect(actual).toEqual(EXOTIC_KV)
  })

  it('records the sliding-window layers of exactly the windowed models', () => {
    // [elements per token across the windowed layers, window in tokens]
    const EXOTIC_SLIDING: Record<string, [number, number]> = {
      'deepseek-v4-flash': [22016, 128], // all 43 layers keep a 128-token window, 512 each
      'deepseek-v4-pro': [31232, 128], // 61 x 512
      'google-gemma-3-1b': [11264, 512], // 22 local x 1 kv x 256 x 2
      'google-gemma-3-4b': [59392, 1024], // 29 local x 4 kv x 256 x 2
      'google-gemma-3-12b': [163840, 1024], // 40 local x 8 kv x 256 x 2
      'google-gemma-3-27b': [212992, 1024], // 52 local x 16 kv x 128 x 2
      'google-gemma-4-12b': [163840, 1024], // 40 local x 8 kv x 256 x 2
      'google-gemma-4-26b-a4b': [102400, 1024], // 25 local x 8 kv x 256 x 2
      'google-gemma-4-31b': [409600, 1024], // 50 local x 16 kv x 256 x 2
      'openai-gpt-oss-120b': [18432, 128], // 18 banded layers
      'openai-gpt-oss-20b': [12288, 128],
      'meta-llama-llama-4-scout': [73728, 8192], // 36 chunked-local layers, chunk 8192
      'meta-llama-llama-4-maverick': [73728, 8192],
    }
    const actual = Object.fromEntries(
      modelsData
        .filter((m) => 'kv_sliding_elements_per_token' in m)
        .map((m) => {
          const w = m as { kv_sliding_elements_per_token: number; kv_sliding_window: number }
          return [m.id, [w.kv_sliding_elements_per_token, w.kv_sliding_window]]
        }),
    )
    expect(actual).toEqual(EXOTIC_SLIDING)
  })

  it('stores DeepSeek V4 at its published parameter counts (arXiv 2606.19348)', () => {
    const flash = modelsData.find((m) => m.id === 'deepseek-v4-flash')
    const pro = modelsData.find((m) => m.id === 'deepseek-v4-pro')
    expect(flash?.num_parameters_billion).toBe(284)
    expect(flash?.active_parameters_billion).toBe(13)
    expect(pro?.num_parameters_billion).toBe(1600)
    expect(pro?.active_parameters_billion).toBe(49)
  })

  it('stores the context lengths the model cards and configs state', () => {
    const CONTEXT: Record<string, number> = {
      'zai-org-glm-4.7': 202752, // config max_position_embeddings
      'zai-org-glm-4.7-flash': 202752,
      'moonshotai-kimi-k2.5': 262144, // card: 256k
      'mistralai-ministral-3-14b-reasoning': 262144, // card: 256k
      'mistralai-devstral-2-123b': 262144, // card: 256k
      'minimax-m2.1': 196608, // config max_position_embeddings
    }
    for (const [id, ctx] of Object.entries(CONTEXT)) {
      expect(modelsData.find((m) => m.id === id)?.context_length, id).toBe(ctx)
    }
  })

  it('points Devstral 2 at its real repository and names the Kimi K3 license', () => {
    const devstral = modelsData.find((m) => m.id === 'mistralai-devstral-2-123b')
    expect(devstral?.hf_url).toBe('https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512')
    expect(modelsData.find((m) => m.id === 'moonshotai-kimi-k3')?.license).toBe('Kimi K3 License')
    expect(
      modelsData.find((m) => m.id === 'moonshotai-kimi-k2-thinking')?.num_parameters_billion,
    ).toBe(1026)
  })

  it('stores the corrected Nemotron Ultra layer count (108, not 128)', () => {
    const ultra = modelsData.find((m) => m.id === 'nvidia-nemotron-3-ultra-550b-a55b')
    expect(ultra?.num_hidden_layers).toBe(108)
  })
})
