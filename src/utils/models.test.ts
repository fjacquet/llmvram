import modelsData from '@data/models.json'
import { BYTES_PER_PARAMETER } from '@engines/constants'
import { calculateInferenceVRAM } from '@engines/inference'
import { describe, expect, it } from 'vitest'
import { type Model, ModelSchema, validateModels } from './schemas'

const models: Model[] = validateModels(modelsData)

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

  it('flags exactly the multi-head latent attention models', () => {
    // Every HF config (params.json for Mistral Large 3) carrying kv_lora_rank.
    // DeepSeek V4 has none: its compressed attention already stores num_kv_heads 1.
    const MLA = [
      'deepseek-r1',
      'inclusionai-ling-3.0-flash',
      'inclusionai-ling-3.0-tiny',
      'mistralai-mistral-large-3-675b',
      'mistralai-mistral-small-4-119b',
      'moonshotai-kimi-k2-instruct',
      'moonshotai-kimi-k2-thinking',
      'moonshotai-kimi-k2.5',
      'moonshotai-kimi-k2.6',
      'moonshotai-kimi-k2.7-code',
      'moonshotai-kimi-k3',
      'moonshotai-kimi-linear-48b-a3b',
      'zai-org-glm-4.7-flash',
      'zai-org-glm-5.2',
    ]
    const actual = modelsData
      .filter((m) => 'use_mla' in m && m.use_mla)
      .map((m) => m.id)
      .sort()
    expect(actual).toEqual(MLA)
  })

  it('records the linear-attention / SSM state of exactly the hybrid models', () => {
    // Bytes per session at TP=1, as vLLM allocates them (MambaStateShapeCalculator):
    // conv state (conv_dim, kernel - 1) in bf16 plus the recurrent state, fp32 where
    // the config sets mamba_ssm_dtype / mamba_ssm_cache_dtype (vLLM Qwen3.5 and
    // NemotronH config handlers), else bf16. Recurrent elements and conv widths match
    // HF transformers cache shapes on the meta device (transformers keeps `kernel`
    // conv columns, vLLM `kernel - 1`).
    const EXOTIC_STATE: Record<string, number> = {
      // gated delta net: conv (2*16*128 + v_heads*128) x 3, recurrent v_heads x 128 x 128 fp32
      'qwen-qwen3.6-27b': 153944064, // 48 linear layers, 48 v-heads
      'qwen-qwen3.6-35b-a3b': 64389120, // 30 layers, 32 v-heads
      'qwen-qwen3.8-27b': 153944064, // 48 layers, 48 v-heads
      'qwen-qwen3.8-2.4t-a95b': 587292672, // 69 layers, 128 v-heads
      // KDA: conv q,k,v 3 x heads x 128 x 3, recurrent heads x 128 x 128, all bf16
      'moonshotai-kimi-k3': 232316928, // 69 KDA layers, 96 heads
      'moonshotai-kimi-linear-48b-a3b': 22446080, // 20 layers, 32 heads
      'inclusionai-ling-3.0-flash': 39280640, // 35 of 42 (layer_group_size 6), 32 heads; single source
      'inclusionai-ling-3.0-tiny': 10100736, // 18 of 24 (layer_group_size 4), 16 heads; single source
      // mamba2: conv (heads*head_dim + 2*8*128) x 3 bf16, recurrent heads x head_dim x 128 fp32
      'nvidia-nemotron-3-nano-30b-a3b': 49082368, // 23 mamba layers
      'nvidia-nemotron-3-nano-4b': 83801088, // 21 layers, 96 x 80
      'nvidia-nemotron-3-super-120b-a12b': 170229760, // 40 layers, 128 heads
      'nvidia-nemotron-3-ultra-550b-a55b': 407961600, // 48 layers, 256 heads
      'nvidia-nemotron-3.5-lightning-30b-a3b': 49082368, // 23 layers
      // short conv: hidden x (L_cache - 1) bf16
      'liquidai-lfm2.5-2.6b': 180224, // 22 conv layers, 2048 x 2
    }
    const actual = Object.fromEntries(
      modelsData
        .filter((m) => 'linear_state_bytes_per_session' in m)
        .map((m) => [
          m.id,
          (m as { linear_state_bytes_per_session: number }).linear_state_bytes_per_session,
        ]),
    )
    expect(actual).toEqual(EXOTIC_STATE)
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

  it('gives every weight ref a size plausible for its format', () => {
    // Implied bytes per parameter; a size pasted under the wrong format falls outside.
    const band = (format: string): [number, number] => {
      if (format.startsWith('gguf-')) {
        const c = BYTES_PER_PARAMETER[format as keyof typeof BYTES_PER_PARAMETER].toNumber()
        return [c * 0.6, c * 1.4]
      }
      const bands: Record<string, [number, number]> = {
        // Upper bound above 1.0: some vendor NVFP4 recipes keep a real fraction of
        // tensors 16-bit rather than quantizing them (Gemma 4 31B nvidia/Gemma-4-31B-IT-NVFP4
        // keeps 10.5B of 31.27B params BF16 per the design spec's recipe table), which
        // pushes the checkpoint's average bytes/param past 1.0. Verified 2026-09-27.
        nvfp4: [0.5, 1.1],
        mxfp4: [0.5, 1.0],
        int4: [0.5, 1.0],
        awq: [0.5, 1.0],
        gptq: [0.5, 1.0],
        fp8: [0.95, 1.3],
        bf16: [1.9, 2.1],
        fp16: [1.9, 2.1],
      }
      return bands[format] ?? [0, Number.POSITIVE_INFINITY]
    }
    let refs = 0
    for (const m of modelsData) {
      const w = (m as { weight_refs?: Record<string, { repo: string; gib: number }> }).weight_refs
      for (const [format, ref] of Object.entries(w ?? {})) {
        refs++
        expect(ref.repo, `${m.id} ${format}`).toMatch(/^[\w.-]+\/[\w.-]+$/)
        const bpp = (ref.gib * 1024 ** 3) / (m.num_parameters_billion * 1e9)
        const [lo, hi] = band(format)
        expect(bpp, `${m.id} ${format} ${bpp.toFixed(3)} bytes/param`).toBeGreaterThanOrEqual(lo)
        expect(bpp, `${m.id} ${format} ${bpp.toFixed(3)} bytes/param`).toBeLessThanOrEqual(hi)
      }
    }
    expect(refs).toBeGreaterThan(100)
  })

  it('agrees with num_parameters_billion within 2% for every bf16 weight ref', () => {
    // A bf16 checkpoint is pure 16-bit, so its file size implies the real parameter
    // count directly: gib * 1024^3 / 2 bytes. Exempted models keep non-BF16 tensors
    // that inflate the file beyond what num_parameters_billion (a deduplicated count)
    // predicts, for a documented reason — never a silent tolerance loosening.
    const EXEMPT: Record<string, string> = {
      // safetensors API: {"parameters":{"F32":3072,"BF16":32913263168},"total":31577937344}.
      // num_parameters_billion (31.6) matches the deduplicated `total` (31.578B) within
      // 0.1%, but the checkpoint stores the tied embedding table twice on disk (32.913B
      // BF16 elements, not deduplicated) plus 3072 F32 elements, so the measured gib
      // implies ~2.083 bytes/param against the dedup count. Verified 2026-09-27.
      'nvidia-nemotron-3.5-lightning-30b-a3b': '3072 F32 elements + duplicated BF16 embedding',
    }
    for (const m of modelsData) {
      const ref = (m as { weight_refs?: Record<string, { repo: string; gib: number }> }).weight_refs
        ?.bf16
      if (!ref) continue
      const impliedParamsBillion = (ref.gib * 1024 ** 3) / 2 / 1e9
      if (m.id in EXEMPT) continue
      const diff =
        Math.abs(impliedParamsBillion - m.num_parameters_billion) / m.num_parameters_billion
      expect(
        diff,
        `${m.id}: implied ${impliedParamsBillion.toFixed(3)}B vs curated ${m.num_parameters_billion}B`,
      ).toBeLessThanOrEqual(0.02)
    }
  })

  it('matches the independent 2026-09-26 checkpoint measurements through the engine', () => {
    // [model id, format, reference repo, file GiB measured by the spike]
    const ANCHORS: [string, string, string, number][] = [
      ['google-gemma-4-31b', 'nvfp4', 'nvidia/Gemma-4-31B-IT-NVFP4', 30.4],
      ['moonshotai-kimi-k3', 'mxfp4', 'moonshotai/Kimi-K3', 1453.7],
      ['meta-llama-llama-3.1-8b', 'fp8', 'RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8', 8.5],
      ['deepseek-r1', 'gguf-q2_k', 'unsloth/DeepSeek-R1-GGUF', 227.3],
      ['qwen-qwen3.8-27b', 'awq', 'cyankiwi/Qwen3.8-27B-AWQ-INT4', 19.6],
    ]
    for (const [id, format, repo, fileGiB] of ANCHORS) {
      const m = validateModels(modelsData).find((x) => x.id === id)
      expect(m?.weight_refs?.[format as never]?.repo, `${id} ${format}`).toBe(repo)
      const weights = calculateInferenceVRAM({
        model: m as never,
        quantization: format as never,
        sequenceLength: 4096,
        batchSize: 1,
      }).modelWeights.toNumber()
      expect(Math.abs(weights - fileGiB) / fileGiB, `${id} ${format}`).toBeLessThan(0.01)
    }
  })

  it('high_precision is consistent with its ref and model', () => {
    for (const m of models) {
      for (const [format, ref] of Object.entries(m.weight_refs ?? {})) {
        const hp = ref?.high_precision
        if (!hp || !ref) continue
        expect(m.architecture, `${m.id} ${format}`).toBe('moe')
        expect(['fp32', 'fp16', 'bf16'].includes(format) || format.startsWith('gguf-')).toBe(false)
        expect(hp.gib, `${m.id} ${format}`).toBeLessThanOrEqual(ref.gib)
        expect(hp.params_b, `${m.id} ${format}`).toBeLessThanOrEqual(
          m.num_parameters_billion * 1.02,
        )
        const bytesPerParam = (hp.gib * 1024 ** 3) / (hp.params_b * 1e9)
        expect(bytesPerParam, `${m.id} ${format}`).toBeGreaterThanOrEqual(1.9)
        expect(bytesPerParam, `${m.id} ${format}`).toBeLessThanOrEqual(4.2)
      }
    }
  })

  it('pins measured high_precision anchors (HF dtype summaries, 2026-09-27)', () => {
    const hp = (id: string, f: string) =>
      models.find((m) => m.id === id)?.weight_refs?.[f as keyof NonNullable<Model['weight_refs']>]
        ?.high_precision
    expect(hp('moonshotai-kimi-k3', 'mxfp4')).toEqual({ params_b: 57.191, gib: 106.55 })
    expect(hp('deepseek-r1', 'fp8')).toEqual({ params_b: 3.919, gib: 7.3 })
    expect(hp('qwen-qwen3-235b-a22b', 'bf16')).toBeUndefined()
  })
})
