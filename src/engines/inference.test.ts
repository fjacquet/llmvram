import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import {
  calculateActivationMemory,
  calculateInferenceVRAM,
  calculateMoEActiveParams,
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
    // Mixtral active params: 0.2 * 46.7 + 0.8 * 46.7 * (2/8) = 9.34 + 9.34 = 18.68B
    // Active ratio: 18.68 / 46.7 = 0.4
    // Effective intermediate size: 14336 * 0.4 = 5734.4 → 5734 (floored)
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
})

describe('calculateMoEActiveParams', () => {
  it('returns unchanged params for dense model', () => {
    const result = calculateMoEActiveParams(llama70b)
    expect(result).toBe(70.0)
  })

  it('calculates active params for Mixtral 8x7B', () => {
    // Shared: 46.7 * 0.2 = 9.34B
    // Expert contribution: 46.7 * 0.8 * (2/8) = 9.34B
    // Total active: 9.34 + 9.34 = 18.68B
    const result = calculateMoEActiveParams(mixtral8x7b)

    const expectedShared = 46.7 * 0.2
    const expectedExpert = 46.7 * 0.8 * (2 / 8)
    const expectedTotal = expectedShared + expectedExpert

    expect(result).toBeCloseTo(expectedTotal, 5)
    expect(result).toBeCloseTo(18.68, 2)
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

    const result = calculateMoEActiveParams(largerMoE)
    const expectedShared = 46.7 * 0.2
    const expectedExpert = 46.7 * 0.8 * (4 / 16) // 25% of expert params active
    const expectedTotal = expectedShared + expectedExpert

    expect(result).toBeCloseTo(expectedTotal, 5)
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

    // Model weights MUST use 46.7B (total), NOT 18.68B (active)
    // 46.7B * 2 bytes (fp16) / (1024^3) = ~86.986 GB
    const expectedWeights = new Decimal(46.7).mul(1e9).mul(2).div(new Decimal(1024).pow(3))

    expect(result.modelWeights.toString()).toBe(expectedWeights.toString())
    expect(result.modelWeights.toNumber()).toBeCloseTo(86.986, 2)

    // But activations should use active params (~18.68B effective)
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
      const expectedWeights = new Decimal(7.0)
        .mul(1e9)
        .mul(formats[i].expectedBpp)
        .div(new Decimal(1024).pow(3))

      expect(results[i].modelWeights.toString()).toBe(expectedWeights.toString())
    }

    // Verify FP32 > FP16 > GPTQ > INT4 (in terms of weight VRAM)
    expect(results[0].modelWeights.toNumber()).toBeGreaterThan(results[1].modelWeights.toNumber())
    expect(results[1].modelWeights.toNumber()).toBeGreaterThan(results[2].modelWeights.toNumber())
    expect(results[2].modelWeights.toNumber()).toBeGreaterThan(results[3].modelWeights.toNumber())
  })
})
