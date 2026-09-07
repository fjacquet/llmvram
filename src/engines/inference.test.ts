import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { PREFILL_CHUNK_TOKENS } from './constants'
import {
  calculateActivationMemory,
  calculateInferenceVRAM,
  calculateMoEActiveParams,
  calculateMoEBatchedParams,
} from './inference'

// Test fixtures - inline model definitions for test isolation
const llama7b: Model = {
  id: 'test-llama-7b',
  name: 'Test Llama 2 7B',
  architecture: 'dense',
  num_parameters_billion: 7.0,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  intermediate_size: 11008,
}

const llama70b: Model = {
  id: 'test-llama-70b',
  name: 'Test Llama 3 70B',
  architecture: 'dense',
  num_parameters_billion: 70.0,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8, // GQA
  intermediate_size: 28672,
}

const mixtral8x7b: Model = {
  id: 'test-mixtral-8x7b',
  name: 'Test Mixtral 8x7B',
  architecture: 'moe',
  num_parameters_billion: 46.7,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_kv_heads: 8, // GQA
  intermediate_size: 14336,
  num_experts: 8,
  num_experts_per_token: 2,
}

const smallModel: Model = {
  id: 'test-small-1b',
  name: 'Test Small 1B',
  architecture: 'dense',
  num_parameters_billion: 1.0,
  hidden_size: 2048,
  num_hidden_layers: 16,
  num_attention_heads: 16,
  intermediate_size: 5504,
}

describe('calculateActivationMemory', () => {
  it('calculates activation memory for dense model', () => {
    // Formula: batch * seq_len * intermediate_size * 4 / (1024^3)
    // = 1 * 2048 * 11008 * 4 / 1073741824
    const result = calculateActivationMemory(llama7b, 2048, 1)

    const expected = new Decimal(1)
      .mul(2048)
      .mul(11008)
      .mul(4) // FP32 bytes
      .div(new Decimal(1024).pow(3))

    expect(result.toString()).toBe(expected.toString())
    expect(result.toNumber()).toBeCloseTo(0.0835, 3) // ~0.0835 GB
  })

  it('calculates reduced activation memory for MoE model using active params', () => {
    // Mixtral active params (tier 2 derivation): 32 * 8 * 3 * 4096 * 14336 / 1e9 = 45.097156608
    // nonExpert = 46.7 - 45.097156608 = 1.602843392
    // active = 1.602843392 + 45.097156608 * (2/8) = 12.877132544B
    // Active ratio: 12.877132544 / 46.7 = 0.275742
    // Effective intermediate size: 14336 * 0.275742 = 3953.03 → 3953 (floored)
    const result = calculateActivationMemory(mixtral8x7b, 2048, 1)

    const activeParams = calculateMoEActiveParams(mixtral8x7b)
    const paramRatio = activeParams / mixtral8x7b.num_parameters_billion
    const effectiveIntermediate = Math.floor(mixtral8x7b.intermediate_size * paramRatio)

    const expected = new Decimal(1)
      .mul(2048)
      .mul(effectiveIntermediate)
      .mul(4)
      .div(new Decimal(1024).pow(3))

    expect(result.toString()).toBe(expected.toString())
    // Should be smaller than if we used full intermediate_size
    expect(result.toNumber()).toBeLessThan(0.06)
  })

  it('scales linearly with sequence length', () => {
    const base = calculateActivationMemory(llama7b, 2048, 1)
    const doubled = calculateActivationMemory(llama7b, 4096, 1)

    const ratio = doubled.div(base)
    expect(ratio.toNumber()).toBeCloseTo(2.0, 10)
  })

  it('scales linearly with batch size', () => {
    const base = calculateActivationMemory(llama7b, 2048, 1)
    const quadrupled = calculateActivationMemory(llama7b, 2048, 4)

    const ratio = quadrupled.div(base)
    expect(ratio.toNumber()).toBeCloseTo(4.0, 10)
  })

  describe('calculateActivationMemory - prefill chunk bound', () => {
    it('is unchanged at or below the prefill chunk', () => {
      const at4k = calculateActivationMemory(llama7b, 4096, 1)
      const at8k = calculateActivationMemory(llama7b, 8192, 1)

      // 1 * 4096 * 11008 * 4 / 1024^3
      expect(at4k.toNumber()).toBeCloseTo((4096 * 11008 * 4) / 1024 ** 3, 6)
      expect(at8k.toNumber()).toBeCloseTo((8192 * 11008 * 4) / 1024 ** 3, 6)
    })

    it('plateaus above the prefill chunk instead of growing with the context window', () => {
      const atChunk = calculateActivationMemory(llama7b, PREFILL_CHUNK_TOKENS, 1)
      const at128k = calculateActivationMemory(llama7b, 131072, 1)
      const at1m = calculateActivationMemory(llama7b, 1048576, 1)

      expect(at128k.toString()).toBe(atChunk.toString())
      expect(at1m.toString()).toBe(atChunk.toString())
    })

    it('scales with batch size only while the batch fits the token budget', () => {
      // The chunk mirrors vLLM's max_num_batched_tokens, a budget for one scheduler step
      // across the whole batch. Below it, doubling the batch doubles the tokens in flight.
      const seq = 512
      const batch1 = calculateActivationMemory(llama7b, seq, 1)
      const batch4 = calculateActivationMemory(llama7b, seq, 4)
      expect(batch4.div(batch1).toNumber()).toBeCloseTo(4, 9)
    })

    it('stops scaling with batch size once the budget is saturated', () => {
      // 1M tokens saturates the budget at batch 1 already, so more concurrent sequences
      // cannot put more tokens through a single step — they wait for the next one.
      const batch1 = calculateActivationMemory(llama7b, 1048576, 1)
      const batch64 = calculateActivationMemory(llama7b, 1048576, 64)

      expect(batch64.toString()).toBe(batch1.toString())
      expect(batch1.toNumber()).toBeCloseTo((PREFILL_CHUNK_TOKENS * 11008 * 4) / 1024 ** 3, 6)
    })
  })
})

describe('calculateMoEActiveParams', () => {
  it('returns unchanged params for dense model', () => {
    const result = calculateMoEActiveParams(llama70b)
    expect(result).toBe(70.0)
  })

  it('calculates active params for Mixtral 8x7B', () => {
    // Tier 2 derivation (per-expert dimensions):
    // expertParams = 32 * 8 * 3 * 4096 * 14336 / 1e9 = 45.097156608
    // nonExpert    = 46.7 - 45.097156608 = 1.602843392
    // active       = 1.602843392 + 45.097156608 * (2 / 8) = 12.877132544
    const result = calculateMoEActiveParams(mixtral8x7b)

    expect(result).toBeCloseTo(12.877132544, 5)
  })

  it('returns full params when num_experts is missing', () => {
    const modelWithoutExperts: Model = {
      ...mixtral8x7b,
      num_experts: undefined,
      num_experts_per_token: undefined,
    }

    const result = calculateMoEActiveParams(modelWithoutExperts)
    expect(result).toBe(46.7)
  })

  it('returns full params when num_experts_per_token is missing', () => {
    const modelWithoutActiveExperts: Model = {
      ...mixtral8x7b,
      num_experts_per_token: undefined,
    }

    const result = calculateMoEActiveParams(modelWithoutActiveExperts)
    expect(result).toBe(46.7)
  })

  it('handles different expert configurations', () => {
    // 16 experts, 4 active per token
    const largerMoE: Model = {
      ...mixtral8x7b,
      num_experts: 16,
      num_experts_per_token: 4,
    }

    // Tier 2 derivation (per-expert dimensions):
    // expertParams = 32 * 16 * 3 * 4096 * 14336 / 1e9 = 90.194313216
    // nonExpert    = max(46.7 - 90.194313216, 0) = 0 (derived expert params exceed total)
    // active       = 0 + 90.194313216 * (4 / 16) = 22.548578304
    const result = calculateMoEActiveParams(largerMoE)

    expect(result).toBeCloseTo(22.548578304, 5)
  })
})

describe('calculateInferenceVRAM', () => {
  it('calculates reference case: Llama 3 70B GPTQ on H100', () => {
    const result = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    // Model weights: 70B * 0.6 bytes/param / (1024^3) = ~39.12 GB
    const expectedWeights = new Decimal(70.0).mul(1e9).mul(0.6).div(new Decimal(1024).pow(3))

    expect(result.modelWeights.toString()).toBe(expectedWeights.toString())
    expect(result.modelWeights.toNumber()).toBeCloseTo(39.12, 2)

    // KV cache: ~1.25 GB (with GQA 8x reduction)
    expect(result.kvCache.toNumber()).toBeCloseTo(1.25, 2)

    // Framework overhead: 1.0 GB
    expect(result.frameworkOverhead.toNumber()).toBe(1.0)

    // Total should fit on H100 80GB
    expect(result.total.toNumber()).toBeLessThan(80)

    // Verify all components are positive
    expect(result.modelWeights.toNumber()).toBeGreaterThan(0)
    expect(result.kvCache.toNumber()).toBeGreaterThan(0)
    expect(result.activations.toNumber()).toBeGreaterThan(0)
    expect(result.frameworkOverhead.toNumber()).toBeGreaterThan(0)
  })

  it('uses TOTAL parameters for MoE model weights, NOT active params', () => {
    const result = calculateInferenceVRAM({
      model: mixtral8x7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    // Model weights MUST use 46.7B (total), NOT 12.88B (active, tier 2 derivation)
    // 46.7B * 2 bytes (fp16) / (1024^3) = ~86.986 GB
    const expectedWeights = new Decimal(46.7).mul(1e9).mul(2).div(new Decimal(1024).pow(3))

    expect(result.modelWeights.toString()).toBe(expectedWeights.toString())
    expect(result.modelWeights.toNumber()).toBeCloseTo(86.986, 2)

    // But activations should use active params (~12.88B effective)
    // This is smaller than if we used full 46.7B
    expect(result.activations.toNumber()).toBeLessThan(0.06)
  })

  it('applies KV quantization independently from weight quantization (INFER-05)', () => {
    const fp16KV = calculateInferenceVRAM({
      model: llama7b,
      quantization: 'gptq', // Weight quantization
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16', // KV quantization
    })

    const int4KV = calculateInferenceVRAM({
      model: llama7b,
      quantization: 'gptq', // Same weight quantization
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'int4', // Different KV quantization
    })

    // Model weights should be identical (same weight quantization)
    expect(fp16KV.modelWeights.toString()).toBe(int4KV.modelWeights.toString())

    // KV cache should be 4x smaller with int4 vs fp16
    const kvRatio = fp16KV.kvCache.div(int4KV.kvCache)
    expect(kvRatio.toNumber()).toBeCloseTo(4.0, 10)

    // Total should be different due to KV cache difference
    expect(fp16KV.total.toNumber()).toBeGreaterThan(int4KV.total.toNumber())
  })

  it('returns breakdown with all Decimal instances', () => {
    const result = calculateInferenceVRAM({
      model: llama7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Verify all fields are Decimal instances
    expect(result.modelWeights).toBeInstanceOf(Decimal)
    expect(result.kvCache).toBeInstanceOf(Decimal)
    expect(result.activations).toBeInstanceOf(Decimal)
    expect(result.frameworkOverhead).toBeInstanceOf(Decimal)
    expect(result.total).toBeInstanceOf(Decimal)
  })

  it('verifies total equals sum of all components', () => {
    const result = calculateInferenceVRAM({
      model: llama7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    const expectedTotal = result.modelWeights
      .add(result.kvCache)
      .add(result.activations)
      .add(result.frameworkOverhead)

    expect(result.total.toString()).toBe(expectedTotal.toString())
  })

  it('calculates small model: 1B FP32', () => {
    const result = calculateInferenceVRAM({
      model: smallModel,
      quantization: 'fp32',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    // Model weights: 1B * 4 bytes (fp32) / (1024^3) = ~3.73 GB
    const expectedWeights = new Decimal(1.0).mul(1e9).mul(4).div(new Decimal(1024).pow(3))

    expect(result.modelWeights.toString()).toBe(expectedWeights.toString())
    expect(result.modelWeights.toNumber()).toBeCloseTo(3.73, 2)

    // Total should be weights + KV + activations + 1GB overhead
    expect(result.total.toNumber()).toBeGreaterThan(3.73)
    expect(result.total.toNumber()).toBeLessThan(6.0) // Reasonable upper bound
  })

  it('defaults to fp16 KV quantization when not specified', () => {
    const resultWithDefault = calculateInferenceVRAM({
      model: llama7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      // kvQuantization not specified
    })

    const resultWithExplicitFP16 = calculateInferenceVRAM({
      model: llama7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    // Should be identical
    expect(resultWithDefault.total.toString()).toBe(resultWithExplicitFP16.total.toString())
  })

  it('handles different quantization formats correctly', () => {
    const formats: Array<{ format: 'fp32' | 'fp16' | 'gptq' | 'int4'; expectedBpp: number }> = [
      { format: 'fp32', expectedBpp: 4.0 },
      { format: 'fp16', expectedBpp: 2.0 },
      { format: 'gptq', expectedBpp: 0.6 },
      { format: 'int4', expectedBpp: 0.5 },
    ]

    const results = formats.map((f) =>
      calculateInferenceVRAM({
        model: llama7b,
        quantization: f.format,
        sequenceLength: 2048,
        batchSize: 1,
      }),
    )

    // Verify weights scale with bytes-per-parameter
    for (let i = 0; i < results.length; i++) {
      const format = formats[i]
      const result = results[i]
      if (!format || !result) continue

      const expectedWeights = new Decimal(7.0)
        .mul(1e9)
        .mul(format.expectedBpp)
        .div(new Decimal(1024).pow(3))

      expect(result.modelWeights.toString()).toBe(expectedWeights.toString())
    }

    // Verify FP32 > FP16 > GPTQ > INT4 (in terms of weight VRAM)
    expect(results[0]?.modelWeights.toNumber()).toBeGreaterThan(
      results[1]?.modelWeights.toNumber() ?? 0,
    )
    expect(results[1]?.modelWeights.toNumber()).toBeGreaterThan(
      results[2]?.modelWeights.toNumber() ?? 0,
    )
    expect(results[2]?.modelWeights.toNumber()).toBeGreaterThan(
      results[3]?.modelWeights.toNumber() ?? 0,
    )
  })
})

describe('calculateMoEActiveParams - three-tier resolution', () => {
  // Qwen3.6 35B A3B shape: 40 layers, 256 experts (8 active), hidden 2048,
  // per-expert intermediate 512, 36B total.
  const qwen35bA3b: Model = {
    id: 'test-qwen-35b-a3b',
    name: 'Test Qwen 35B A3B',
    architecture: 'moe',
    num_parameters_billion: 36,
    hidden_size: 2048,
    num_hidden_layers: 40,
    num_attention_heads: 16,
    num_kv_heads: 2,
    intermediate_size: 512,
    num_experts: 256,
    num_experts_per_token: 8,
  }

  it('tier 1: uses active_parameters_billion when present', () => {
    const withExplicit: Model = { ...qwen35bA3b, active_parameters_billion: 3 }
    expect(calculateMoEActiveParams(withExplicit)).toBe(3)
  })

  it('tier 2: derives from per-expert dimensions when the field is absent', () => {
    // expertParams = 40 * 256 * 3 * 2048 * 512 / 1e9 = 32.21225472
    // nonExpert    = 36 - 32.21225472 = 3.78774528
    // active       = 3.78774528 + 32.21225472 * (8 / 256) = 4.7942...
    expect(calculateMoEActiveParams(qwen35bA3b)).toBeCloseTo(4.7942, 3)
  })

  it('tier 2: never returns more than the total parameter count', () => {
    // Bad data: derived expert params (32.2B) exceed the declared total, AND the active
    // ratio is high enough that expertParams * ratio alone would still overshoot.
    // Both clamps have to fire: nonExpert floors at 0, then the sum caps at the total.
    // (A low ratio like 8/256 would pass without exercising the outer clamp at all.)
    const inconsistent: Model = {
      ...qwen35bA3b,
      num_parameters_billion: 10,
      num_experts_per_token: 128,
    }
    expect(calculateMoEActiveParams(inconsistent)).toBe(10)
  })

  it('tier 3: dense models return the full parameter count', () => {
    expect(calculateMoEActiveParams(llama7b)).toBe(llama7b.num_parameters_billion)
  })

  it('tier 3: MoE with missing expert fields returns the full parameter count', () => {
    const incomplete: Model = { ...qwen35bA3b, num_experts: undefined }
    expect(calculateMoEActiveParams(incomplete)).toBe(36)
  })
})

describe('calculateMoEBatchedParams', () => {
  it('equals the batch-1 active count at batch 1', () => {
    expect(calculateMoEBatchedParams(mixtral8x7b, 1)).toBeCloseTo(
      calculateMoEActiveParams(mixtral8x7b),
      9,
    )
  })

  it('grows toward the full expert set as the batch widens', () => {
    // Mixtral 8x7B: k/E = 2/8, so expertTotal = (46.7 - 12.877132544) / 0.75 = 45.097156608
    // and nonExpert = 1.602843392. At batch 4 the touched fraction is 1 - 0.75^4 = 0.68359375,
    // giving 1.602843392 + 45.097156608 * 0.68359375 = 32.430977792
    expect(calculateMoEBatchedParams(mixtral8x7b, 4)).toBeCloseTo(32.430977792, 9)
  })

  it('is monotonically increasing in batch size', () => {
    const values = [1, 2, 4, 16, 64].map((b) => calculateMoEBatchedParams(mixtral8x7b, b))
    for (let i = 1; i < values.length; i++) {
      expect(values[i] ?? 0).toBeGreaterThan(values[i - 1] ?? 0)
    }
  })

  it('never exceeds the full parameter count', () => {
    expect(calculateMoEBatchedParams(mixtral8x7b, 4096)).toBeLessThanOrEqual(
      mixtral8x7b.num_parameters_billion,
    )
    expect(calculateMoEBatchedParams(mixtral8x7b, 4096)).toBeCloseTo(46.7, 6)
  })

  it('holds its bounds when the anchor implies a negative non-expert share', () => {
    // total * (k/E) can exceed the stated active count, which makes the derived
    // expertTotal larger than the model and nonExpertParams negative. The clamps must
    // still hold: never below the batch-1 figure, never above the full weight set.
    const skewed: Model = {
      ...mixtral8x7b,
      id: 'test-skewed-moe',
      name: 'Skewed MoE',
      num_parameters_billion: 100,
      active_parameters_billion: 10,
      num_experts: 8,
      num_experts_per_token: 4,
    }

    const values = [1, 2, 4, 8, 64].map((b) => calculateMoEBatchedParams(skewed, b))

    expect(values[0]).toBeCloseTo(10, 9)
    for (let i = 1; i < values.length; i++) {
      expect(values[i] ?? 0).toBeGreaterThanOrEqual(values[i - 1] ?? 0)
      expect(values[i] ?? 0).toBeLessThanOrEqual(100)
    }
    expect(values[values.length - 1] ?? 0).toBeCloseTo(100, 6)
  })

  it('returns the total for a dense model at any batch size', () => {
    expect(calculateMoEBatchedParams(llama70b, 1)).toBe(70.0)
    expect(calculateMoEBatchedParams(llama70b, 128)).toBe(70.0)
  })
})
