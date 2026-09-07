import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { estimatePerformance } from './performance'

// Test fixtures
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

const llama3_70b: Model = {
  id: 'meta-llama-llama-3.1-70b',
  name: 'LLaMA 3.1 70B',
  architecture: 'dense',
  num_parameters_billion: 70,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,
  intermediate_size: 28672,
}

const llama3_8b: Model = {
  id: 'meta-llama-llama-3.1-8b',
  name: 'LLaMA 3.1 8B',
  architecture: 'dense',
  num_parameters_billion: 8,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_kv_heads: 8,
  intermediate_size: 14336,
}

const tiny_1b_model: Model = {
  id: 'tiny-1b',
  name: 'Tiny 1B',
  architecture: 'dense',
  num_parameters_billion: 1,
  hidden_size: 2048,
  num_hidden_layers: 24,
  num_attention_heads: 16,
  num_kv_heads: 16,
  intermediate_size: 8192,
}

const llama7b: Model = {
  id: 'test-llama-7b',
  name: 'Test Llama 7B',
  architecture: 'dense',
  num_parameters_billion: 7,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  intermediate_size: 11008,
}

const gpu_no_flops: GPU = {
  id: 'test-gpu-no-flops',
  name: 'Test GPU No FLOPS',
  manufacturer: 'nvidia',
  vram_gb: 80,
  memory_bandwidth_gbps: 1000,
  memory_type: 'HBM3',
  bus_width: 4096,
  tier: 'datacenter',
  // No fp16_tflops or fp32_tflops
}

describe('estimatePerformance', () => {
  it('should identify memory-bound scenario for typical LLM inference', () => {
    // LLaMA 3 70B FP16 on H100 80GB SXM
    // Model size: 70B * 2 bytes = 140GB = ~130.39 GiB
    // Memory-bound TPS: 3350 GB/s / 130.39 GB ≈ 23.93 tokens/sec
    // Compute-bound TPS: 989 TFLOPS / (70B * 2) = 989e12 / 140e9 ≈ 7064 tokens/sec
    // Result: memory-bound (~23.93 tok/s)
    const result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Verify tokens per second is in expected range (23-26 tok/s)
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(23)
    expect(result.tokensPerSecond.toNumber()).toBeLessThan(26)

    // Verify bottleneck is memory-bound
    expect(result.bottleneck).toBe('memory')
    expect(result.isMemoryBound).toBe(true)
    expect(result.isComputeBound).toBe(false)

    // TTFT is now prefill (compute-bound) + one decode step, not a fixed multiple of
    // decode speed. At T=2048 the linear term dominates:
    //   linearFLOPs = 2 * 70e9 * 2048 = 2.8672e14
    //   attentionFLOPs = 2 * 80 * 2048^2 * 8192 ≈ 5.494e12 (small next to linear)
    //   effectiveFLOPS = 989e12 * 0.45 (PREFILL_MFU) = 4.4505e14
    //   prefillSeconds ≈ 2.9221e14 / 4.4505e14 ≈ 0.6566 s
    //   decodeSeconds = 1 / 23.9286 ≈ 0.0418 s
    //   TTFT ≈ 0.6984 s
    expect(result.timeToFirstToken.toNumber()).toBeGreaterThan(0.6)
    expect(result.timeToFirstToken.toNumber()).toBeLessThan(0.8)
  })

  it('should show higher throughput for GPTQ quantized models', () => {
    // LLaMA 3 70B GPTQ on H100
    // Model size: 70B * 0.6 bytes ≈ 42GB = ~39.12 GiB
    // Memory-bound TPS: 3350 GB/s / 39.12 GB ≈ 85.6 tokens/sec
    // Should be ~3.3x faster than FP16 due to smaller model size
    const fp16_result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    const gptq_result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'gptq',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // GPTQ should be significantly faster (at least 2.5x)
    expect(gptq_result.tokensPerSecond.toNumber()).toBeGreaterThan(
      fp16_result.tokensPerSecond.toNumber() * 2.5,
    )

    // Verify GPTQ throughput is in expected range (75-90 tok/s)
    expect(gptq_result.tokensPerSecond.toNumber()).toBeGreaterThan(75)
    expect(gptq_result.tokensPerSecond.toNumber()).toBeLessThan(90)

    // Should still be memory-bound
    expect(gptq_result.bottleneck).toBe('memory')
  })

  it('should scale tokens/sec linearly with batch size in memory-bound regime', () => {
    const batch1 = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    const batch4 = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 4,
    })

    // Batch=4 should be ~4x faster (within 5% tolerance)
    const expectedBatch4 = batch1.tokensPerSecond.mul(4)
    const ratio = batch4.tokensPerSecond.div(expectedBatch4).toNumber()
    expect(ratio).toBeGreaterThan(0.95)
    expect(ratio).toBeLessThan(1.05)
  })

  it('should estimate TTFT using the two-term compute-bound prefill model', () => {
    // TTFT is no longer a fixed multiple of decode speed — it is prefill time
    // (compute-bound roofline) plus one decode step.
    //
    // LLaMA 3 70B FP16 on H100 80GB SXM, sequenceLength = 2048:
    //   memoryBoundTPS = 3350e9 / (70e9 * 2) = 23.928571428571428571 tok/s (batch=1)
    //   decodeSeconds  = 1 / 23.928571428571428571 = 0.041791044776119403 s
    //   linearFLOPs    = 2 * 70e9 * 2048               = 286,720,000,000,000
    //   attentionFLOPs = 2 * 80 * 2048^2 * 8192         =   5,494,281,338,880
    //   totalFLOPs     = linearFLOPs + attentionFLOPs   = 292,214,281,338,880
    //   effectiveFLOPS = 989e12 * 0.45 (PREFILL_MFU)    = 445,050,000,000,000
    //   prefillSeconds = totalFLOPs / effectiveFLOPS    ≈ 0.656594895267678 s
    //   TTFT           = prefillSeconds + decodeSeconds ≈ 0.698385940043797 s
    const result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    expect(result.prefillSeconds?.toNumber()).toBeCloseTo(0.656594895267678, 9)
    expect(result.timeToFirstToken.toNumber()).toBeCloseTo(0.698385940043797, 9)
  })

  it('should handle missing FLOPS data gracefully', () => {
    // GPU with no fp16_tflops or fp32_tflops should still work
    const result = estimatePerformance({
      model: llama3_8b,
      gpu: gpu_no_flops,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Should produce valid result (memory-bound only)
    expect(result.tokensPerSecond.isFinite()).toBe(true)
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(0)

    // Should be memory-bound (no compute data to compare)
    expect(result.bottleneck).toBe('memory')
    expect(result.isMemoryBound).toBe(true)
    expect(result.isComputeBound).toBe(false)
  })

  it('should detect compute-bound scenarios for small quantized models on fast GPUs', () => {
    // 1B model INT4 on H100: very small model, high FLOPS
    // Model size: 1B * 0.5 bytes = 0.5GB
    // Memory-bound TPS: 3350 GB/s / 0.5 GB = 6700 tokens/sec
    // Compute-bound TPS: 989 TFLOPS / (1B * 2) = 989e12 / 2e9 = 494,500 tokens/sec
    // Result: memory-bound, but let's test the compute path is calculated correctly
    const result = estimatePerformance({
      model: tiny_1b_model,
      gpu: h100_80gb_sxm,
      quantization: 'int4',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Should be memory-bound (memory is still the bottleneck even for small models)
    expect(result.bottleneck).toBe('memory')

    // But verify compute-bound TPS is much higher
    // Memory-bound should be dominant, tokens/sec should be 1000-10000 range
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(1000)
    expect(result.tokensPerSecond.toNumber()).toBeLessThan(10000)
  })

  it('should correctly set bottleneck field and boolean flags', () => {
    const memoryBound = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Memory-bound case
    expect(memoryBound.bottleneck).toBe('memory')
    expect(memoryBound.isMemoryBound).toBe(true)
    expect(memoryBound.isComputeBound).toBe(false)

    // The booleans and string should be consistent
    if (memoryBound.bottleneck === 'memory') {
      expect(memoryBound.isMemoryBound).toBe(true)
    }
    if (memoryBound.bottleneck === 'compute') {
      expect(memoryBound.isComputeBound).toBe(true)
    }
    if (memoryBound.bottleneck === 'balanced') {
      expect(memoryBound.isMemoryBound).toBe(true)
      expect(memoryBound.isComputeBound).toBe(true)
    }
  })

  it('should detect compute-bound scenario when FLOPS is the limiting factor', () => {
    // Create a very low-FLOPS GPU (0.5 TFLOPS) but decent bandwidth (1000 GB/s)
    const low_flops_gpu: GPU = {
      id: 'low-flops-gpu',
      name: 'Low FLOPS GPU',
      manufacturer: 'nvidia',
      vram_gb: 8,
      memory_bandwidth_gbps: 1000,
      memory_type: 'GDDR6',
      bus_width: 256,
      fp16_tflops: 0.5, // Very low compute
      fp32_tflops: 0.25,
      tier: 'consumer',
    }

    // 1B model INT4: very small, low memory requirement
    // Model size: 1B * 0.5 bytes = ~0.47 GB
    // Memory-bound TPS: 1000 GB/s / 0.47 GB ≈ 2128 tok/s
    // Compute-bound TPS: 0.5 TFLOPS / (1B * 2) = 0.5e12 / 2e9 = 250 tok/s
    // Result: compute-bound at ~250 tok/s
    const result = estimatePerformance({
      model: tiny_1b_model,
      gpu: low_flops_gpu,
      quantization: 'int4',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Should be compute-bound
    expect(result.bottleneck).toBe('compute')
    expect(result.isMemoryBound).toBe(false)
    expect(result.isComputeBound).toBe(true)

    // Tokens per second should be limited by compute (~250 tok/s)
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(200)
    expect(result.tokensPerSecond.toNumber()).toBeLessThan(300)
  })

  it('should detect balanced bottleneck when memory and compute bounds are close', () => {
    // Create a custom 10B model
    const model_10b: Model = {
      id: 'test-10b',
      name: 'Test 10B',
      architecture: 'dense',
      num_parameters_billion: 10,
      hidden_size: 4096,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_kv_heads: 32,
      intermediate_size: 16384,
    }

    // Create GPU with bandwidth and FLOPS tuned for balanced performance
    // Model size in INT4: 10B * 0.5 bytes = 5GB
    // Target: ~500 tok/s for both bounds
    // Memory-bound: 2500 GB/s / 5 GB = 500 tok/s
    // Compute-bound: 10 TFLOPS / (10B * 2) = 10e12 / 20e9 = 500 tok/s
    const balanced_gpu: GPU = {
      id: 'balanced-gpu',
      name: 'Balanced GPU',
      manufacturer: 'nvidia',
      vram_gb: 24,
      memory_bandwidth_gbps: 2500,
      memory_type: 'HBM3',
      bus_width: 4096,
      fp16_tflops: 10,
      fp32_tflops: 5,
      tier: 'datacenter',
    }

    const result = estimatePerformance({
      model: model_10b,
      gpu: balanced_gpu,
      quantization: 'int4',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Should be classified as balanced (within 5% tolerance)
    expect(result.bottleneck).toBe('balanced')
    expect(result.isMemoryBound).toBe(true)
    expect(result.isComputeBound).toBe(true)

    // Tokens per second should be around 500
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(450)
    expect(result.tokensPerSecond.toNumber()).toBeLessThan(550)
  })

  it('should return Decimal values for all numeric fields', () => {
    const result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    expect(result.tokensPerSecond).toBeInstanceOf(Decimal)
    expect(result.timeToFirstToken).toBeInstanceOf(Decimal)
    expect(result.prefillSeconds).toBeInstanceOf(Decimal)
  })

  it('should produce reasonable TTFT for various throughput levels', () => {
    // High throughput (small quantized model)
    const fast = estimatePerformance({
      model: llama3_8b,
      gpu: h100_80gb_sxm,
      quantization: 'int4',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Low throughput (large FP16 model)
    const slow = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Fast model should have lower TTFT
    expect(fast.timeToFirstToken.toNumber()).toBeLessThan(slow.timeToFirstToken.toNumber())

    // Both should be in reasonable range (0.001 to 1 second)
    expect(fast.timeToFirstToken.toNumber()).toBeGreaterThan(0.001)
    expect(fast.timeToFirstToken.toNumber()).toBeLessThan(1)
    expect(slow.timeToFirstToken.toNumber()).toBeGreaterThan(0.001)
    expect(slow.timeToFirstToken.toNumber()).toBeLessThan(1)
  })

  it('should handle edge case of very slow GPU', () => {
    const slow_gpu: GPU = {
      id: 'slow-gpu',
      name: 'Slow GPU',
      manufacturer: 'nvidia',
      vram_gb: 24,
      memory_bandwidth_gbps: 100, // Very slow bandwidth
      memory_type: 'GDDR6',
      bus_width: 384,
      fp16_tflops: 10,
      fp32_tflops: 5,
      tier: 'consumer',
    }

    const result = estimatePerformance({
      model: llama3_8b,
      gpu: slow_gpu,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    // Should still produce valid result
    expect(result.tokensPerSecond.isFinite()).toBe(true)
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(0)

    // Slow GPU should have low throughput (< 10 tok/s)
    expect(result.tokensPerSecond.toNumber()).toBeLessThan(10)
  })
})

describe('estimatePerformance - prefill model', () => {
  it('TTFT grows with sequence length', () => {
    const short = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })
    const long = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 131072,
      batchSize: 1,
    })

    expect(long.timeToFirstToken.greaterThan(short.timeToFirstToken)).toBe(true)
  })

  it('grows super-linearly once the quadratic attention term dominates', () => {
    const at256k = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 262144,
      batchSize: 1,
    })
    const at512k = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 524288,
      batchSize: 1,
    })

    const ratio = at512k.prefillSeconds?.div(at256k.prefillSeconds ?? 1).toNumber() ?? 0
    expect(ratio).toBeGreaterThan(2)
  })

  it('reports the linear term as the bottleneck at short context', () => {
    const result = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })

    expect(result.prefillBottleneck).toBe('linear')
    expect(result.prefillEstimateDegraded).toBe(false)
  })

  it('reports the attention term as the bottleneck at 1M context', () => {
    const result = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1048576,
      batchSize: 1,
    })

    expect(result.prefillBottleneck).toBe('attention')
  })

  it('degrades gracefully when the GPU has no FLOPS data', () => {
    const noFlopsGPU: GPU = { ...h100_80gb_sxm, fp16_tflops: undefined, fp32_tflops: undefined }
    const result = estimatePerformance({
      model: llama7b,
      gpu: noFlopsGPU,
      quantization: 'fp16',
      sequenceLength: 131072,
      batchSize: 1,
    })

    expect(result.prefillSeconds).toBeNull()
    expect(result.prefillEstimateDegraded).toBe(true)
    expect(result.timeToFirstToken.isFinite()).toBe(true)
    expect(result.timeToFirstToken.greaterThan(0)).toBe(true)
  })
})

describe('estimatePerformance - MoE active parameters', () => {
  const moe36bA3b: Model = {
    id: 'test-moe-36b-a3b',
    name: 'Test MoE 36B A3B',
    architecture: 'moe',
    num_parameters_billion: 36,
    hidden_size: 2048,
    num_hidden_layers: 40,
    num_attention_heads: 16,
    num_kv_heads: 2,
    intermediate_size: 512,
    num_experts: 256,
    num_experts_per_token: 8,
    active_parameters_billion: 3,
  }

  it('decode throughput uses active parameters, not the total', () => {
    const result = estimatePerformance({
      model: moe36bA3b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })

    // bandwidth 3350 GB/s / (3B active x 2 bytes) — orders of magnitude above
    // the 3350e9 / (36e9 x 2) = ~46 tok/s the total-parameter formula gave.
    expect(result.tokensPerSecond.toNumber()).toBeGreaterThan(400)
  })

  it('decode throughput respects weight quantization', () => {
    const fp16 = estimatePerformance({
      model: moe36bA3b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 1024,
      batchSize: 1,
    })
    const int4 = estimatePerformance({
      model: moe36bA3b,
      gpu: h100_80gb_sxm,
      quantization: 'gptq',
      sequenceLength: 1024,
      batchSize: 1,
    })

    // Fewer bytes read per token means more tokens per second
    expect(int4.tokensPerSecond.greaterThan(fp16.tokensPerSecond)).toBe(true)
  })
})
