import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { calculateKVCacheVRAM } from './kv-cache'

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
})
