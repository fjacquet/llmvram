/**
 * Integration tests for full calculation pipeline
 *
 * Tests end-to-end flow: quantization → KV cache → inference → performance
 * Uses realistic model and GPU configurations to verify correctness.
 * These are NON-Worker tests (Workers don't work in jsdom environment).
 */

import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { calculateInferenceVRAM } from './inference'
import { estimatePerformance } from './performance'

// GPU fixtures
const h100_80gb_sxm: GPU = {
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

const rtx4090: GPU = {
  id: 'nvidia-rtx-4090',
  name: 'NVIDIA RTX 4090',
  manufacturer: 'nvidia',
  vram_gb: 24,
  memory_bandwidth_gbps: 1008,
  memory_type: 'GDDR6X',
  bus_width: 384,
  fp16_tflops: 165,
  fp32_tflops: 82.5,
  tdp_watts: 450,
  tier: 'consumer',
}

// Model fixtures
const llama3_70b: Model = {
  id: 'meta-llama/Meta-Llama-3-70B',
  name: 'Llama 3 70B',
  architecture: 'dense',
  num_parameters_billion: 70.0,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8, // GQA: 8x reduction
  intermediate_size: 28672,
}

const mixtral_8x7b: Model = {
  id: 'mistralai/Mixtral-8x7B-v0.1',
  name: 'Mixtral 8x7B',
  architecture: 'moe',
  num_parameters_billion: 46.7, // Total params, not active (13B)
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_kv_heads: 8, // GQA
  intermediate_size: 14336,
  num_experts: 8,
  num_experts_per_token: 2,
}

const small_1b: Model = {
  id: 'test-small-1b',
  name: 'Test Small 1B',
  architecture: 'dense',
  num_parameters_billion: 1.0,
  hidden_size: 2048,
  num_hidden_layers: 16,
  num_attention_heads: 16,
  intermediate_size: 5504,
}

const llama2_7b: Model = {
  id: 'meta-llama/Llama-2-7b-hf',
  name: 'Llama 2 7B',
  architecture: 'dense',
  num_parameters_billion: 7.0,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  intermediate_size: 11008,
}

describe('Integration: Full Calculation Pipeline', () => {
  it('canonical reference: Llama 3 70B GPTQ on H100 80GB', () => {
    // This is the canonical example from RESEARCH.md
    // Expected: ~42 GB total (fits on H100), memory-bound, good throughput
    const vram = calculateInferenceVRAM({
      model: llama3_70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const performance = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'gptq',
      batchSize: 1,
    })

    // Verify VRAM breakdown
    // Model weights: 70B * 0.6 bytes/param = ~39.12 GB
    expect(vram.modelWeights.toNumber()).toBeCloseTo(39.12, 1)

    // KV cache: 2 * 80 layers * 8192 hidden * 4096 seq * 1 batch * 2 bytes * (8/64 GQA)
    // = 2 * 80 * 8192 * 4096 * 1 * 2 * 0.125 / (1024^3) = ~1.25 GB
    expect(vram.kvCache.toNumber()).toBeCloseTo(1.25, 1)

    // Total should fit on H100 80GB
    expect(vram.total.toNumber()).toBeLessThan(80)
    expect(vram.total.toNumber()).toBeGreaterThan(40) // Should be ~42 GB

    // Framework overhead should be 1GB
    expect(vram.frameworkOverhead.toNumber()).toBe(1.0)

    // Verify all fields are Decimal instances
    expect(vram.modelWeights).toBeInstanceOf(Decimal)
    expect(vram.kvCache).toBeInstanceOf(Decimal)
    expect(vram.activations).toBeInstanceOf(Decimal)
    expect(vram.frameworkOverhead).toBeInstanceOf(Decimal)
    expect(vram.total).toBeInstanceOf(Decimal)

    // Performance verification
    // GPTQ on H100 should be fast (>50 tok/s)
    expect(performance.tokensPerSecond.toNumber()).toBeGreaterThan(50)

    // Should be memory-bound (typical for LLM inference)
    expect(performance.bottleneck).toBe('memory')
    expect(performance.isMemoryBound).toBe(true)

    // TTFT should be reasonable (< 100ms for 4K context)
    expect(performance.timeToFirstToken.toNumber()).toBeLessThan(0.1)

    // Verify performance fields are Decimal
    expect(performance.tokensPerSecond).toBeInstanceOf(Decimal)
    expect(performance.timeToFirstToken).toBeInstanceOf(Decimal)
  })

  it('MoE verification: Mixtral 8x7B FP16 exceeds 80GB (total params, not active)', () => {
    // This test verifies the critical MoE behavior:
    // - Weight VRAM uses TOTAL parameters (46.7B), not active (13B)
    // - Activations use active parameters
    const vram = calculateInferenceVRAM({
      model: mixtral_8x7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const performance = estimatePerformance({
      model: mixtral_8x7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      batchSize: 1,
    })

    // Model weights: 46.7B * 2 bytes/param = ~86.97 GB
    // This is TOTAL params (all 8 experts), NOT active (13B = 2 experts)
    expect(vram.modelWeights.toNumber()).toBeCloseTo(86.97, 1)

    // Total should EXCEED 80GB (doesn't fit on single H100)
    expect(vram.total.toNumber()).toBeGreaterThan(80)
    expect(vram.total.toNumber()).toBeLessThan(95) // Should be ~88-90 GB

    // Performance should still calculate correctly
    expect(performance.tokensPerSecond.toNumber()).toBeGreaterThan(0)
    expect(performance.bottleneck).toBe('memory')
  })

  it('small model fits easily on consumer GPU', () => {
    // 1B model should fit comfortably on RTX 4090 (24GB)
    const vram = calculateInferenceVRAM({
      model: small_1b,
      quantization: 'fp16',
      sequenceLength: 512,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const performance = estimatePerformance({
      model: small_1b,
      gpu: rtx4090,
      quantization: 'fp16',
      batchSize: 1,
    })

    // Model weights: 1B * 2 bytes = ~1.86 GB
    expect(vram.modelWeights.toNumber()).toBeCloseTo(1.86, 1)

    // Total should be < 5GB
    expect(vram.total.toNumber()).toBeLessThan(5)

    // Should fit easily on RTX 4090
    expect(vram.total.toNumber()).toBeLessThan(rtx4090.vram_gb)

    // Performance should be good (high tok/s for small model)
    expect(performance.tokensPerSecond.toNumber()).toBeGreaterThan(100)
  })

  it('KV quantization impact: int4 vs fp16 (INFER-05)', () => {
    // Same model/GPU/quant, different KV quantization
    // int4 KV cache should be 4x smaller than fp16
    const vramFP16 = calculateInferenceVRAM({
      model: llama2_7b,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const vramINT4 = calculateInferenceVRAM({
      model: llama2_7b,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'int4',
    })

    // KV cache should be 4x smaller with int4
    const kvRatio = vramFP16.kvCache.div(vramINT4.kvCache)
    expect(kvRatio.toNumber()).toBeCloseTo(4.0, 1)

    // Total VRAM should be meaningfully different
    const totalDiff = vramFP16.total.sub(vramINT4.total).toNumber()
    expect(totalDiff).toBeGreaterThan(0.5) // At least 0.5GB difference

    // Model weights should be identical (only KV cache changes)
    expect(vramFP16.modelWeights.toString()).toBe(vramINT4.modelWeights.toString())
  })

  it('long context: 131K tokens (max per INFER-06)', () => {
    // Verify calculation completes without error for maximum sequence length
    // KV cache should dominate VRAM for very long contexts
    const vram = calculateInferenceVRAM({
      model: llama2_7b,
      quantization: 'fp16',
      sequenceLength: 131072, // 128K context
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const performance = estimatePerformance({
      model: llama2_7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      batchSize: 1,
    })

    // KV cache should be LARGE (potentially larger than model weights)
    expect(vram.kvCache.toNumber()).toBeGreaterThan(10) // > 10GB for 128K context

    // KV cache should dominate total VRAM for long context
    const kvCacheRatio = vram.kvCache.div(vram.total).toNumber()
    expect(kvCacheRatio).toBeGreaterThan(0.3) // KV cache > 30% of total

    // Total should be calculable without error
    expect(vram.total.toNumber()).toBeGreaterThan(0)

    // Performance should still calculate (may be slower due to long context)
    expect(performance.tokensPerSecond.toNumber()).toBeGreaterThan(0)
  })

  it('batch size scaling: activations increase linearly', () => {
    // Activation memory should scale linearly with batch size
    const batch1 = calculateInferenceVRAM({
      model: llama2_7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const batch4 = calculateInferenceVRAM({
      model: llama2_7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 4,
      kvQuantization: 'fp16',
    })

    // Activations should be ~4x larger
    const activationRatio = batch4.activations.div(batch1.activations).toNumber()
    expect(activationRatio).toBeCloseTo(4.0, 1)

    // KV cache should also be ~4x larger
    const kvRatio = batch4.kvCache.div(batch1.kvCache).toNumber()
    expect(kvRatio).toBeCloseTo(4.0, 1)

    // Model weights should be identical (independent of batch)
    expect(batch1.modelWeights.toString()).toBe(batch4.modelWeights.toString())
  })

  it('quantization formats: GPTQ vs FP16 comparison', () => {
    // GPTQ (4-bit + overhead) should be ~3.3x smaller than FP16
    const vramFP16 = calculateInferenceVRAM({
      model: llama3_70b,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const vramGPTQ = calculateInferenceVRAM({
      model: llama3_70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    // Model weights ratio: 2.0 bytes (FP16) / 0.6 bytes (GPTQ) = 3.33x
    const weightRatio = vramFP16.modelWeights.div(vramGPTQ.modelWeights).toNumber()
    expect(weightRatio).toBeCloseTo(3.33, 1)

    // GPTQ should save significant VRAM
    const totalSavings = vramFP16.total.sub(vramGPTQ.total).toNumber()
    expect(totalSavings).toBeGreaterThan(80) // > 80GB saved for 70B model

    // Both should fit/not-fit on same boundaries
    // FP16: ~130GB (doesn't fit on 80GB H100)
    expect(vramFP16.total.toNumber()).toBeGreaterThan(80)
    // GPTQ: ~42GB (fits on 80GB H100)
    expect(vramGPTQ.total.toNumber()).toBeLessThan(80)
  })
})
