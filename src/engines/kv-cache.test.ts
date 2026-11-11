import modelsData from '@data/models.json'
import { type Model, ModelSchema, validateModels } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import {
  attendedTokens,
  calculateDecodeKVReadVRAM,
  calculateKVCacheVRAM,
  calculateLinearStateVRAM,
  calculateSparseIndexerVRAM,
} from './kv-cache'

// Test fixtures - inline model definitions for test isolation
const llama70bGQA: Model = {
  id: 'test-llama-70b-gqa',
  name: 'Test Llama 3 70B (GQA)',
  architecture: 'dense',
  num_parameters_billion: 70.0,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8, // GQA: 8x reduction
  intermediate_size: 28672,
}

const llama70bMHA: Model = {
  ...llama70bGQA,
  id: 'test-llama-70b-mha',
  name: 'Test Llama 3 70B (MHA equivalent)',
  num_kv_heads: 64, // No GQA reduction
}

const llama7b: Model = {
  id: 'test-llama-7b',
  name: 'Test Llama 2 7B',
  architecture: 'dense',
  num_parameters_billion: 7.0,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  // No num_kv_heads - tests default fallback
  intermediate_size: 11008,
}

const mqaModel: Model = {
  id: 'test-mqa',
  name: 'Test MQA Model',
  architecture: 'dense',
  num_parameters_billion: 7.0,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_kv_heads: 1, // MQA: 32x reduction
  intermediate_size: 11008,
}

const mlaModel: Model = {
  id: 'test-mla-deepseek-r1',
  name: 'Test DeepSeek R1 (MLA)',
  architecture: 'moe',
  num_parameters_billion: 671,
  hidden_size: 7168,
  num_hidden_layers: 61,
  num_attention_heads: 128,
  num_kv_heads: 128, // nominal dummy value — override field takes precedence
  intermediate_size: 18432,
  num_experts: 256,
  num_experts_per_token: 8,
  // MLA: 61 layers × (kv_lora_rank 512 + qk_rope_head_dim 64) = 35136
  kv_cache_elements_per_token: 35136,
}

describe('calculateKVCacheVRAM', () => {
  it('applies GQA reduction correctly - 8x smaller cache for Llama 3 70B vs MHA equivalent', () => {
    const params = {
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16' as const,
    }

    const gqaResult = calculateKVCacheVRAM({ model: llama70bGQA, ...params })
    const mhaResult = calculateKVCacheVRAM({ model: llama70bMHA, ...params })

    // MHA result should be exactly 8x larger than GQA result (64 heads / 8 heads)
    const ratio = mhaResult.div(gqaResult)
    expect(ratio.toNumber()).toBeCloseTo(8.0, 10) // 10 decimal places precision
  })

  it('calculates known reference value for Llama 3 70B with GQA', () => {
    // Formula: 2 * 80 * 8192 * 4096 * 1 * 2 (fp16) * (8/64) / (1024^3)
    // = 2 * 80 * 8192 * 4096 * 2 * 0.125 / 1073741824
    // = 1,342,177,280 / 1,073,741,824
    // = 1.25 GB
    const result = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    // Expected calculation
    const expected = new Decimal(2)
      .mul(80)
      .mul(8192)
      .mul(4096)
      .mul(1)
      .mul(2) // fp16 bytes
      .mul(new Decimal(8).div(64)) // GQA ratio
      .div(new Decimal(1024).pow(3))

    expect(result.toString()).toBe(expected.toString())
    expect(result.toNumber()).toBeCloseTo(1.25, 4) // 1.25 GB
  })

  it('scales linearly with sequence length - 2x seq_len = 2x KV cache', () => {
    const base = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    const doubled = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 8192,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    const ratio = doubled.div(base)
    expect(ratio.toNumber()).toBeCloseTo(2.0, 10)
  })

  it('scales linearly with batch size - 4x batch = 4x KV cache', () => {
    const base = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    const quadrupled = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 4096,
      batchSize: 4,
      kvPrecision: 'fp16',
    })

    const ratio = quadrupled.div(base)
    expect(ratio.toNumber()).toBeCloseTo(4.0, 10)
  })

  it('applies KV precision correctly - int4 is 4x smaller than fp16 (INFER-05)', () => {
    const fp16Result = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    const int4Result = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'int4',
    })

    // fp16 uses 2 bytes, int4 uses 0.5 bytes → 4x reduction
    const ratio = fp16Result.div(int4Result)
    expect(ratio.toNumber()).toBeCloseTo(4.0, 10)
  })

  it('applies MQA reduction correctly - 32x smaller cache with single KV head', () => {
    const mqaResult = calculateKVCacheVRAM({
      model: mqaModel,
      sequenceLength: 2048,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    // Create MHA equivalent (no num_kv_heads, defaults to num_attention_heads)
    const mhaEquivalent: Model = { ...mqaModel, num_kv_heads: undefined }
    const mhaResult = calculateKVCacheVRAM({
      model: mhaEquivalent,
      sequenceLength: 2048,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    // MHA result should be exactly 32x larger (32 heads / 1 head)
    const ratio = mhaResult.div(mqaResult)
    expect(ratio.toNumber()).toBeCloseTo(32.0, 10)
  })

  it('defaults to standard MHA when num_kv_heads is undefined', () => {
    // llama7b fixture has no num_kv_heads field
    const result = calculateKVCacheVRAM({
      model: llama7b,
      sequenceLength: 2048,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    // Should calculate with gqaRatio = 1.0 (num_attention_heads / num_attention_heads)
    // Formula: 2 * 32 * 4096 * 2048 * 1 * 2 * 1.0 / (1024^3)
    const expected = new Decimal(2)
      .mul(32)
      .mul(4096)
      .mul(2048)
      .mul(1)
      .mul(2) // fp16
      .mul(1.0) // No GQA reduction
      .div(new Decimal(1024).pow(3))

    expect(result.toString()).toBe(expected.toString())
    expect(result.toNumber()).toBeCloseTo(1.0, 2) // ~1 GB
  })

  it('supports all KV precision formats', () => {
    const model = llama7b
    const params = { sequenceLength: 2048, batchSize: 1 }

    // Test all precision formats produce valid results
    const fp16 = calculateKVCacheVRAM({ model, ...params, kvPrecision: 'fp16' })
    const fp8 = calculateKVCacheVRAM({ model, ...params, kvPrecision: 'fp8' })
    const int8 = calculateKVCacheVRAM({ model, ...params, kvPrecision: 'int8' })
    const int4 = calculateKVCacheVRAM({ model, ...params, kvPrecision: 'int4' })

    // All should be positive
    expect(fp16.toNumber()).toBeGreaterThan(0)
    expect(fp8.toNumber()).toBeGreaterThan(0)
    expect(int8.toNumber()).toBeGreaterThan(0)
    expect(int4.toNumber()).toBeGreaterThan(0)

    // fp8 and int8 should be equal (both 1 byte)
    expect(fp8.toString()).toBe(int8.toString())

    // fp16 should be 2x fp8 (2 bytes vs 1 byte)
    expect(fp16.div(fp8).toNumber()).toBeCloseTo(2.0, 10)

    // fp16 should be 4x int4 (2 bytes vs 0.5 bytes)
    expect(fp16.div(int4).toNumber()).toBeCloseTo(4.0, 10)
  })

  it('uses kv_cache_elements_per_token override when present (MLA exact value)', () => {
    const result = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    // 35136 × 4096 × 1 × 2 bytes / 1024³ = 0.26806640625 GB exactly
    expect(result.toNumber()).toBe(0.26806640625)
  })

  it('override is much smaller than the nominal GQA formula for MLA models', () => {
    const nominal: Model = { ...mlaModel, kv_cache_elements_per_token: undefined }
    const overridden = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    const gqaFormula = calculateKVCacheVRAM({
      model: nominal,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    // GQA formula: 2×61×7168 = 874496 elem/token vs MLA 35136 → ~24.9× reduction
    expect(gqaFormula.div(overridden).toNumber()).toBeCloseTo(874496 / 35136, 3)
  })

  it('override path applies KV precision scaling (int8 = half of fp16)', () => {
    const fp16 = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 2,
      kvPrecision: 'fp16',
    })
    const int8 = calculateKVCacheVRAM({
      model: mlaModel,
      sequenceLength: 4096,
      batchSize: 2,
      kvPrecision: 'int8',
    })
    expect(int8.mul(2).toNumber()).toBeCloseTo(fp16.toNumber(), 10)
  })

  it('scales linearly to 1M tokens', () => {
    const at128k = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 131072,
      batchSize: 1,
      kvPrecision: 'fp16',
    })
    const at1m = calculateKVCacheVRAM({
      model: llama70bGQA,
      sequenceLength: 1048576,
      batchSize: 1,
      kvPrecision: 'fp16',
    })

    expect(at1m.div(at128k).toNumber()).toBeCloseTo(8, 9)
    expect(at1m.toNumber()).toBeGreaterThan(100)
  })
})

describe('calculateKVCacheVRAM - sliding-window layers', () => {
  const gemmaLike: Model = {
    id: 'sliding',
    name: 'Sliding',
    architecture: 'dense',
    num_parameters_billion: 12,
    hidden_size: 3840,
    num_hidden_layers: 48,
    num_attention_heads: 16,
    num_kv_heads: 8,
    intermediate_size: 15360,
    kv_cache_elements_per_token: 32768,
    kv_sliding_elements_per_token: 163840,
    kv_sliding_window: 1024,
  }
  const kv = (sequenceLength: number) =>
    calculateKVCacheVRAM({ model: gemmaLike, sequenceLength, batchSize: 1, kvPrecision: 'fp16' })
      .mul(1024 ** 3)
      .toNumber()

  it('caps the windowed layers at the window (vLLM allocates min(window, context))', () => {
    expect(kv(32768)).toBeCloseTo((32768 * 32768 + 163840 * 1024) * 2, 0)
  })

  it('charges the windowed layers per token below the window', () => {
    expect(kv(512)).toBeCloseTo((32768 + 163840) * 512 * 2, 0)
  })
})

describe('calculateKVCacheVRAM - linear-attention state', () => {
  const hybrid: Model = {
    id: 'hybrid',
    name: 'Hybrid',
    architecture: 'dense',
    num_parameters_billion: 27,
    hidden_size: 5120,
    num_hidden_layers: 64,
    num_attention_heads: 24,
    num_kv_heads: 4,
    intermediate_size: 17408,
    kv_cache_elements_per_token: 32768,
    linear_state_bytes_per_session: 153944064,
  }
  const bytes = (sequenceLength: number, batchSize: number) =>
    calculateKVCacheVRAM({ model: hybrid, sequenceLength, batchSize, kvPrecision: 'fp8' })
      .mul(1024 ** 3)
      .toNumber()

  it('adds a constant state per session, independent of context and KV precision', () => {
    expect(bytes(1024, 1)).toBeCloseTo(32768 * 1024 * 1 + 153944064, 0)
    expect(bytes(8192, 1) - bytes(4096, 1)).toBeCloseTo(32768 * 4096, 0)
  })

  it('scales the state with sessions', () => {
    expect(bytes(1024, 4)).toBeCloseTo(4 * bytes(1024, 1), 0)
  })

  it('exposes the state as its own figure', () => {
    expect(
      calculateLinearStateVRAM(hybrid, 4)
        .mul(1024 ** 3)
        .toNumber(),
    ).toBeCloseTo(4 * 153944064, 0)
  })
})

describe('sparse attention (DSA) indexer cache and reads', () => {
  const row = validateModels(modelsData).find((m) => m.id === 'zai-org-glm-5.2')
  if (!row) throw new Error('zai-org-glm-5.2 missing from models.json')
  // 2026-11-11: sparse attention (DSA). The real row now carries the fields; the dense baseline
  // strips them. vLLM's indexer cache is uint8, index_head_dim 128 plus 4 scale bytes = 132 bytes
  // per token per indexer layer, 21 layers (sources in models.test.ts).
  const {
    sparse_attention_topk: _topk,
    sparse_indexer_bytes_per_token: _bytes,
    sparse_indexer_heads: _heads,
    sparse_indexer_layers: _layers,
    ...glm
  } = row
  const sparse: Model = {
    ...glm,
    sparse_attention_topk: 2048,
    sparse_indexer_bytes_per_token: 21 * 132,
  }
  // 2026-11-11, sizing T1: the DSA `sparse` copy now stores FP8 KV as fp8_ds_mla (656 B per 576
  // elements); the dense copies stripped of the sparse fields keep 1 B, so scale them by this.
  const dsa = new Decimal(656).div(576)
  const kv = (model: Model, sequenceLength: number, kvPrecision: 'fp16' | 'fp8' = 'fp8') =>
    calculateKVCacheVRAM({ model, sequenceLength, batchSize: 1, kvPrecision })
  const read = (model: Model, sequenceLength: number) =>
    calculateDecodeKVReadVRAM({ model, sequenceLength, batchSize: 1, kvPrecision: 'fp8' })

  it('attendedTokens is min(context, topk), or the context without sparse fields', () => {
    expect(attendedTokens(sparse, 1024)).toBe(1024)
    expect(attendedTokens(sparse, 131072)).toBe(2048)
    expect(attendedTokens(glm, 131072)).toBe(131072)
  })

  it('indexer cache is bytes x context x sessions, 0 without the fields', () => {
    expect(
      calculateSparseIndexerVRAM(sparse, 131072, 3)
        .mul(1024 ** 3)
        .toNumber(),
    ).toBe(21 * 132 * 131072 * 3)
    expect(calculateSparseIndexerVRAM(glm, 131072, 3).toNumber()).toBe(0)
  })

  it('memory stores every token plus the indexer cache, whatever the KV precision', () => {
    const indexer = calculateSparseIndexerVRAM(sparse, 131072, 1)
    expect(kv(sparse, 131072).sub(kv(glm, 131072).mul(dsa)).toNumber()).toBeCloseTo(
      indexer.toNumber(),
      12,
    )
    expect(
      kv(sparse, 131072, 'fp16')
        .sub(kv(glm, 131072, 'fp16'))
        .toNumber(),
    ).toBeCloseTo(indexer.toNumber(), 12)
  })

  it('decode reads the KV of topk tokens plus the indexer over the full context at 128k', () => {
    const expected = kv(glm, 2048)
      .mul(dsa)
      .add(new Decimal(21 * 132 * 131072).div(1024 ** 3))
    expect(read(sparse, 131072).toNumber()).toBeCloseTo(expected.toNumber(), 12)
  })

  it('Mistral Large 4 decode at 128k reads the KV of topk tokens plus its full indexer cache', () => {
    // 2026-11-11, final review M6 (spec success criterion): the real row, 44 indexer layers
    const ml4 = validateModels(modelsData).find((m) => m.id === 'mistralai-mistral-large-4-1t-a52b')
    if (!ml4) throw new Error('mistralai-mistral-large-4-1t-a52b missing from models.json')
    const {
      sparse_attention_topk: _k,
      sparse_indexer_bytes_per_token: _b,
      sparse_indexer_heads: _h,
      sparse_indexer_layers: _l,
      ...dense
    } = ml4
    const expected = kv(dense, 2048)
      .mul(dsa)
      .add(new Decimal(44 * 132 * 131072).div(1024 ** 3))
    expect(read(ml4, 131072).toNumber()).toBeCloseTo(expected.toNumber(), 12)
    expect(read(ml4, 131072).toNumber()).toBeLessThan(kv(ml4, 131072).toNumber())
  })

  it('decode reads equal memory below topk, and the full KV without sparse fields', () => {
    expect(read(sparse, 1024).toNumber()).toBeCloseTo(kv(sparse, 1024).toNumber(), 12)
    expect(read(glm, 131072).toNumber()).toBeCloseTo(kv(glm, 131072).toNumber(), 12)
  })

  it('schema sets topk and indexer bytes together, heads and layers together and with topk', () => {
    expect(ModelSchema.safeParse(sparse).success).toBe(true)
    expect(ModelSchema.safeParse({ ...glm, sparse_attention_topk: 2048 }).success).toBe(false)
    expect(ModelSchema.safeParse({ ...glm, sparse_indexer_bytes_per_token: 2772 }).success).toBe(
      false,
    )
    const scored = { ...sparse, sparse_indexer_heads: 32, sparse_indexer_layers: 21 }
    expect(ModelSchema.safeParse(scored).success).toBe(true)
    expect(ModelSchema.safeParse({ ...sparse, sparse_indexer_heads: 32 }).success).toBe(false)
    const { sparse_attention_topk, sparse_indexer_bytes_per_token, ...noTopk } = scored
    expect(ModelSchema.safeParse(noTopk).success).toBe(false)
  })
})

describe('FP8 KV on DSA models (vLLM fp8_ds_mla, 656 B per token per layer)', () => {
  const models = validateModels(modelsData)
  const byId = (id: string) => {
    const m = models.find((x) => x.id === id)
    if (!m) throw new Error(`${id} missing from models.json`)
    return m
  }
  const gb = (model: Model, kvPrecision: 'fp16' | 'fp8') =>
    calculateKVCacheVRAM({ model, sequenceLength: 1024, batchSize: 1, kvPrecision })

  it('Mistral Large 4 (DSA) at FP8 charges 656/576 of the 1-byte figure', () => {
    const ml4 = byId('mistralai-mistral-large-4-1t-a52b')
    const expected = new Decimal(ml4.kv_cache_elements_per_token ?? 0)
      .div(576)
      .mul(656)
      .mul(1024)
      .div(1024 ** 3)
      .add(calculateSparseIndexerVRAM(ml4, 1024, 1)) // indexer cache is unaffected by KV precision
    expect(gb(ml4, 'fp8').toNumber()).toBeCloseTo(expected.toNumber(), 12)
  })

  it('leaves FP16 unchanged', () => {
    const ml4 = byId('mistralai-mistral-large-4-1t-a52b')
    const expected = new Decimal(ml4.kv_cache_elements_per_token ?? 0)
      .mul(1024)
      .mul(2)
      .div(1024 ** 3)
      .add(calculateSparseIndexerVRAM(ml4, 1024, 1))
    expect(gb(ml4, 'fp16').toNumber()).toBeCloseTo(expected.toNumber(), 12)
  })

  it('leaves non-DSA MLA (DeepSeek R1) at 1 byte per element', () => {
    const r1 = byId('deepseek-r1')
    const expected = new Decimal(r1.kv_cache_elements_per_token ?? 0).mul(1024).div(1024 ** 3)
    expect(gb(r1, 'fp8').toNumber()).toBeCloseTo(expected.toNumber(), 12)
  })
})
