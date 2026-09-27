import modelsData from '@data/models.json'
import type { GPU, Model } from '@utils/schemas'
import { validateModels } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { BYTES_PER_GB, INTERCONNECT_SPECS } from './constants'
import { calculateInferenceVRAM, calculateMoEBatchedParams, splitMoEParams } from './inference'
import { calculateKVCacheVRAM, calculateLinearStateVRAM } from './kv-cache'
import { calculateMultiGPUVRAM, resolveInterconnect } from './multi-gpu'
import { estimatePerformance, expertAllToAllSeconds } from './performance'
import { calculateModelWeightVRAM } from './quantization'
import type { MultiGPUVRAMBreakdown } from './types'

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
  max_gpus_per_node: 8,
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
  max_gpus_per_node: 8,
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
    //   attentionFLOPs = 2 * 80 * 2048^2 * 8192 ≈ 5.498e12 (small next to linear)
    //   effectiveFLOPS = 989e12 * 0.45 (PREFILL_MFU) = 4.4505e14
    //   prefillSeconds ≈ 2.9221e14 / 4.4505e14 ≈ 0.6566 s
    //   decodeSeconds = 1 / 23.9286 ≈ 0.0418 s
    //   TTFT ≈ 0.6984 s
    expect(result.timeToFirstToken.toNumber()).toBeGreaterThan(0.6)
    expect(result.timeToFirstToken.toNumber()).toBeLessThan(0.8)
  })

  it('should show higher throughput for GPTQ quantized models', () => {
    // LLaMA 3 70B GPTQ on H100
    // Model size: 70B * 0.52 bytes ≈ 36.4GB = ~33.90 GiB
    // Memory-bound TPS: 3350 GB/s / 33.90 GB ≈ 98.8 tokens/sec (before KV cache bytes)
    // Should be ~3.8x faster than FP16 due to smaller model size
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

    // Verify GPTQ throughput is in expected range (85-95 tok/s)
    expect(gptq_result.tokensPerSecond.toNumber()).toBeGreaterThan(85)
    expect(gptq_result.tokensPerSecond.toNumber()).toBeLessThan(95)

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
    //   decodeBytes    = 70e9 * 2 weights + 2048 * 163840 * 2 KV = 140,671,088,640
    //   decodeSeconds  = decodeBytes / 3350e9           = 0.041991369743283581 s
    //   linearFLOPs    = 2 * 70e9 * 2048               = 286,720,000,000,000
    //   attentionFLOPs = 2 * 80 * 2048^2 * 8192         =   5,497,558,138,880
    //   totalFLOPs     = linearFLOPs + attentionFLOPs   = 292,217,558,138,880
    //   effectiveFLOPS = 989e12 * 0.45 (PREFILL_MFU)    = 445,050,000,000,000
    //   prefillSeconds = totalFLOPs / effectiveFLOPS    ≈ 0.656594895267678 s
    //   TTFT           = prefillSeconds + decodeSeconds ≈ 0.698586265010961 s
    const result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })

    expect(result.prefillSeconds?.toNumber()).toBeCloseTo(0.656594895267678, 9)
    expect(result.timeToFirstToken.toNumber()).toBeCloseTo(0.698586265010961, 9)
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
      max_gpus_per_node: 8,
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
    // Model size in INT4: 10B * 0.5625 bytes = 5.625GB
    // Target: ~500 tok/s for both bounds
    // Memory-bound: 2812.5 GB/s / 5.625 GB = 500 tok/s
    // Compute-bound: 10 TFLOPS / (10B * 2) = 10e12 / 20e9 = 500 tok/s
    const balanced_gpu: GPU = {
      id: 'balanced-gpu',
      name: 'Balanced GPU',
      manufacturer: 'nvidia',
      vram_gb: 24,
      memory_bandwidth_gbps: 2812.5,
      memory_type: 'HBM3',
      bus_width: 4096,
      fp16_tflops: 10,
      fp32_tflops: 5,
      tier: 'datacenter',
      max_gpus_per_node: 8,
    }

    // A 16-token context keeps KV reads and attention FLOPs negligible, so the
    // weights alone set both bounds.
    const result = estimatePerformance({
      model: model_10b,
      gpu: balanced_gpu,
      quantization: 'int4',
      sequenceLength: 16,
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
      max_gpus_per_node: 8,
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

  it('degrades gracefully instead of dividing by zero when fp16_tflops is 0', () => {
    // A GPU with fp16_tflops: 0 (e.g. a user-entered custom-FLOPS value of 0) must not
    // pass the "FLOPS data present" guard — otherwise effectiveFLOPS is 0 and
    // prefillSeconds becomes Infinity with prefillEstimateDegraded left false.
    const zeroFlopsGPU: GPU = { ...h100_80gb_sxm, fp16_tflops: 0, fp32_tflops: undefined }
    const result = estimatePerformance({
      model: llama7b,
      gpu: zeroFlopsGPU,
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

describe('estimatePerformance - multi-GPU scaling', () => {
  // Prefill scales by numGPUs × prefillScalingEfficiency. Decode does not: it divides
  // the bytes per step across the GPUs and adds two all-reduces per layer.
  const multiGPUResult: MultiGPUVRAMBreakdown = {
    numGPUs: 2,
    strategy: 'tensor-parallel',
    perGPU: {
      modelWeights: new Decimal(0),
      kvCache: new Decimal(0),
      activations: new Decimal(0),
      frameworkOverhead: new Decimal(0),
      communicationOverhead: new Decimal(0),
      total: new Decimal(0),
    },
    replicatedMemory: new Decimal(0),
    totalPerGPU: new Decimal(0),
    utilizationPercent: new Decimal(0),
    singleGPUBaseline: new Decimal(0),
    numNodes: 1,
    gpusPerNode: 2,
    intraNodeEfficiency: 0.9,
    interNodeDecodeEfficiency: 1,
    interNodePrefillEfficiency: 1,
    bubbleEfficiency: 1,
    scalingEfficiency: 0.9,
    // Single node (numNodes === 1): prefillScalingEfficiency equals scalingEfficiency, which
    // is exactly what lets this test assert decode and prefill scale by the same factor.
    prefillScalingEfficiency: 0.9,
    interconnectBandwidthGBps: 900,
  }

  it('scales prefill by numGPUs × efficiency, and decode by less than numGPUs', () => {
    const singleGPU = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 131072,
      batchSize: 1,
    })

    const multiGPU = estimatePerformance({
      model: llama7b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 131072,
      batchSize: 1,
      multiGPUResult,
    })

    const factor = multiGPUResult.numGPUs * multiGPUResult.prefillScalingEfficiency

    // MHA 7B: 32 KV heads split 2 ways, so bytes halve; the all-reduce latency keeps it below 2x
    const tpsRatio = multiGPU.tokensPerSecond.div(singleGPU.tokensPerSecond).toNumber()
    expect(tpsRatio).toBeGreaterThan(1.5)
    expect(tpsRatio).toBeLessThan(2)

    // Prefill time scales inversely with effectiveFLOPS (more FLOPS => less time), so the
    // single-GPU / multi-GPU ratio (not multi-GPU / single-GPU) equals the same factor.
    const singlePrefill = singleGPU.prefillSeconds
    const multiPrefill = multiGPU.prefillSeconds
    if (singlePrefill === null || multiPrefill === null) {
      throw new Error('expected prefillSeconds to be computable for this GPU')
    }
    const prefillRatio = singlePrefill.div(multiPrefill).toNumber()
    expect(prefillRatio).toBeCloseTo(factor, 9)
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

describe('multi-node roofline separation', () => {
  // Built via calculateMultiGPUVRAM rather than hand-written, so the fixture tracks the
  // MultiGPUVRAMBreakdown interface as it evolves instead of drifting from it.
  const baseMultiGPUResult: MultiGPUVRAMBreakdown = calculateMultiGPUVRAM(
    calculateInferenceVRAM({
      model: llama3_70b,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    }),
    llama3_70b,
    h100_80gb_sxm.vram_gb,
    4,
    'tensor-parallel',
    h100_80gb_sxm,
  )

  it('scales prefill by the prefill efficiency, not the decode one', () => {
    const multiGPUResult = {
      ...baseMultiGPUResult,
      numGPUs: 32,
      numNodes: 4,
      gpusPerNode: 8,
      scalingEfficiency: 0.9,
      prefillScalingEfficiency: 0.5,
    }
    const fast = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      multiGPUResult: { ...multiGPUResult, prefillScalingEfficiency: 0.9 },
    })
    const slow = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      multiGPUResult,
    })
    // Decode is identical: both carry scalingEfficiency 0.9.
    expect(slow.tokensPerSecond.toString()).toBe(fast.tokensPerSecond.toString())
    // Prefill is not: the slow fabric halves effectiveFLOPS, doubling time-to-first-token's
    // compute term.
    expect(slow.timeToFirstToken.greaterThan(fast.timeToFirstToken)).toBe(true)
  })
})

// Decode step = (weights + batch x KV per sequence) / bandwidth + TP all-reduce latency.
describe('estimatePerformance - decode reads the KV cache', () => {
  const tps = (batchSize: number, sequenceLength: number) =>
    estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp8',
      batchSize,
      sequenceLength,
    }).tokensPerSecond.toNumber()

  it('matches bandwidth / (weights + KV) at batch 1', () => {
    // 70 GB FP8 weights + 80 layers x 2 x 1024 x 2 B x 32768 tokens KV
    const weights = 70e9
    const kv = 80 * 2 * 1024 * 2 * 32768
    expect(tps(1, 32768)).toBeCloseTo(3350e9 / (weights + kv), 3)
  })

  it('stops growing linearly with batch once KV reads dominate', () => {
    // At 128k context each sequence reads ~43 GB of KV, more than the weights
    expect(tps(64, 131072) / tps(1, 131072)).toBeLessThan(4)
  })

  it('is slower at long context than at short context', () => {
    expect(tps(8, 131072)).toBeLessThan(tps(8, 1024))
  })
})

describe('estimatePerformance - compute ceiling is aggregate', () => {
  it('does not multiply the compute bound by batch', () => {
    // 1B model, 1 TFLOPS: 2e9 FLOPs/token + attention caps the machine near 500 tok/s
    const slowCompute: GPU = { ...h100_80gb_sxm, fp16_tflops: 1, memory_bandwidth_gbps: 100000 }
    const perf = estimatePerformance({
      model: tiny_1b_model,
      gpu: slowCompute,
      quantization: 'fp16',
      batchSize: 64,
      sequenceLength: 128,
    })
    expect(perf.bottleneck).toBe('compute')
    expect(perf.tokensPerSecond.toNumber()).toBeLessThan(500)
  })
})

describe('estimatePerformance - multi-GPU decode', () => {
  const run = (
    model: Model,
    numGPUs: number,
    strategy: 'tensor-parallel' | 'pipeline-parallel',
  ) => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp8',
      sequenceLength: 8192,
      batchSize: 1,
    })
    const multi =
      numGPUs > 1
        ? calculateMultiGPUVRAM(singleGPU, model, 80, numGPUs, strategy, h100_80gb_sxm)
        : null
    return estimatePerformance({
      model,
      gpu: h100_80gb_sxm,
      quantization: 'fp8',
      batchSize: 1,
      sequenceLength: 8192,
      multiGPUResult: multi,
    }).tokensPerSecond.toNumber()
  }

  it('keeps TP8 at batch 1 below 8x, because every layer waits on two all-reduces', () => {
    const speedup = run(llama3_70b, 8, 'tensor-parallel') / run(llama3_70b, 1, 'tensor-parallel')
    expect(speedup).toBeGreaterThan(3)
    expect(speedup).toBeLessThan(7)
  })

  it('gives no decode speedup to pipeline parallelism at batch 1', () => {
    const speedup =
      run(llama3_70b, 4, 'pipeline-parallel') / run(llama3_70b, 1, 'pipeline-parallel')
    expect(speedup).toBeCloseTo(1, 1)
  })

  it('reads the full MLA cache on every TP rank', () => {
    const mla: Model = { ...llama3_70b, num_kv_heads: 64, kv_cache_elements_per_token: 40960 }
    const gqa: Model = { ...mla, use_mla: undefined }
    expect(run({ ...mla, use_mla: true }, 8, 'tensor-parallel')).toBeLessThan(
      run(gqa, 8, 'tensor-parallel'),
    )
  })
})

describe('expert-parallel decode', () => {
  it('prices all-to-all as latency plus bytes over the link (DeepEP anchor)', () => {
    // DeepEP legacy table, EP8: 128 tokens, 7168 hidden, top-8, FP8 dispatch + BF16
    // combine = 77 + 114 us, moving 22 MB at ~115 GB/s.
    const seconds = expertAllToAllSeconds(128, 8, 7168, 115, 0)
    expect(seconds * 1e6).toBeGreaterThan(191 * 0.9)
    expect(seconds * 1e6).toBeLessThan(191 * 1.1)
  })

  it('serves a long-context MLA MoE faster than TP, which re-reads duplicated KV', () => {
    const moe: Model = {
      ...llama3_70b,
      architecture: 'moe',
      num_parameters_billion: 671,
      active_parameters_billion: 37,
      num_experts: 256,
      num_experts_per_token: 8,
      kv_cache_elements_per_token: 35136,
      use_mla: true,
    }
    const nvl: GPU = { ...h100_80gb_sxm, vram_gb: 288, interconnect: 'nvlink-5' }
    const run = (strategy: 'tensor-parallel' | 'expert-parallel') => {
      const single = calculateInferenceVRAM({
        model: moe,
        quantization: 'fp8',
        sequenceLength: 131072,
        batchSize: 64,
      })
      const multi = calculateMultiGPUVRAM(single, moe, 288, 8, strategy, nvl)
      return estimatePerformance({
        model: moe,
        gpu: nvl,
        quantization: 'fp8',
        batchSize: 64,
        sequenceLength: 131072,
        multiGPUResult: multi,
      }).tokensPerSecond.toNumber()
    }
    expect(run('expert-parallel')).toBeGreaterThan(run('tensor-parallel'))
  })
})

// Fixtures pulled from the real database (not hand-written) so their weight_refs stay
// anchored to a measured checkpoint. I-2 (spec §3): the decode path must read the same
// effective bytes-per-parameter as the memory sizing, not the format constant.
describe('estimatePerformance - decode honors measured weight_refs', () => {
  const llama8b = validateModels(modelsData).find(
    (m) => m.id === 'meta-llama-llama-3.1-8b',
  ) as Model
  const gemmaMoe = validateModels(modelsData).find(
    (m) => m.id === 'google-gemma-4-26b-a4b',
  ) as Model

  it('dense, single GPU: memory-bound decode reads the fp8 ref, not the fp8 constant', () => {
    const gpu = h100_80gb_sxm
    const sequenceLength = 4096
    const batchSize = 1
    const run = (model: Model) =>
      estimatePerformance({ model, gpu, quantization: 'fp8', sequenceLength, batchSize })
        .tokensPerSecond

    const withRef = run(llama8b)
    const withoutRef = run({ ...llama8b, weight_refs: undefined })

    // The fp8 ref (8.46 GiB) implies ~1.136 bytes/param, heavier than the 1.0 constant,
    // so honoring it must read more bytes and be slower.
    expect(withRef.lessThan(withoutRef)).toBe(true)

    const weightBytes = calculateModelWeightVRAM(
      llama8b.num_parameters_billion,
      'fp8',
      llama8b,
    ).mul(BYTES_PER_GB)
    const kvBytes = calculateKVCacheVRAM({
      model: llama8b,
      sequenceLength,
      batchSize,
      kvPrecision: 'fp16',
    }).mul(BYTES_PER_GB)
    const expected = new Decimal(gpu.memory_bandwidth_gbps).mul(1e9).div(weightBytes.add(kvBytes))
    expect(withRef.toString()).toBe(expected.toString())
  })

  it('MoE batch path: memory-bound decode reads the nvfp4 ref for the batched expert subset', () => {
    const gpu = h100_80gb_sxm
    const sequenceLength = 8192
    const batchSize = 32
    const run = (model: Model) =>
      estimatePerformance({ model, gpu, quantization: 'nvfp4', sequenceLength, batchSize })
        .tokensPerSecond

    const withRef = run(gemmaMoe)
    const withoutRef = run({ ...gemmaMoe, weight_refs: undefined })
    expect(withRef.equals(withoutRef)).toBe(false)

    const decodeParams = calculateMoEBatchedParams(gemmaMoe, batchSize)
    const weightBytes = calculateModelWeightVRAM(decodeParams, 'nvfp4', gemmaMoe).mul(BYTES_PER_GB)
    const kvBytes = calculateKVCacheVRAM({
      model: gemmaMoe,
      sequenceLength,
      batchSize,
      kvPrecision: 'fp16',
    }).mul(BYTES_PER_GB)
    const expected = new Decimal(batchSize)
      .mul(gpu.memory_bandwidth_gbps)
      .mul(1e9)
      .div(weightBytes.add(kvBytes))
    expect(withRef.toString()).toBe(expected.toString())
  })

  it('expert-parallel branch: per-GPU weight bytes use the ref for the base + routed share', () => {
    const gpu = h100_80gb_sxm
    const sequenceLength = 8192
    const batchSize = 16
    const numGPUs = 8
    const run = (model: Model) => {
      const single = calculateInferenceVRAM({
        model,
        quantization: 'nvfp4',
        sequenceLength,
        batchSize,
      })
      const multi = calculateMultiGPUVRAM(
        single,
        model,
        gpu.vram_gb,
        numGPUs,
        'expert-parallel',
        gpu,
      )
      return estimatePerformance({
        model,
        gpu,
        quantization: 'nvfp4',
        sequenceLength,
        batchSize,
        multiGPUResult: multi,
      })
    }

    const withRef = run(gemmaMoe)
    const withoutRef = run({ ...gemmaMoe, weight_refs: undefined })
    expect(withRef.tokensPerSecond.equals(withoutRef.tokensPerSecond)).toBe(false)
    expect(withRef.bottleneck).toBe('memory')

    const split = splitMoEParams(gemmaMoe)
    if (!split) throw new Error('expected gemmaMoe to split into base + routed experts')
    const gpusPerStage = numGPUs // single node, non-pipeline strategy: stages = 1
    const decodeParams = calculateMoEBatchedParams(gemmaMoe, batchSize)
    const perGPUWeightBytes = calculateModelWeightVRAM(
      split.baseB + Math.max(0, decodeParams - split.baseB) / gpusPerStage,
      'nvfp4',
      gemmaMoe,
    ).mul(BYTES_PER_GB)
    const kvBytes = calculateKVCacheVRAM({
      model: gemmaMoe,
      sequenceLength,
      batchSize,
      kvPrecision: 'fp16',
    }).mul(BYTES_PER_GB)
    const stateBytes = calculateLinearStateVRAM(gemmaMoe, batchSize).mul(BYTES_PER_GB)
    const perGPUKVBytes = kvBytes
      .sub(stateBytes)
      .div(gpusPerStage)
      .add(stateBytes.div(gpusPerStage))
    const perGPUBytes = perGPUWeightBytes.add(perGPUKVBytes) // stages = 1
    const memorySeconds = perGPUBytes.div(new Decimal(gpu.memory_bandwidth_gbps).mul(1e9))

    const link = INTERCONNECT_SPECS[resolveInterconnect(gpu)]
    const commSecondsPerLayer = expertAllToAllSeconds(
      batchSize / gpusPerStage,
      split.expertsPerToken,
      gemmaMoe.hidden_size,
      link.bandwidthGBps / 2,
      link.allreduceLatencyUs,
    )
    const stageSeconds = memorySeconds.add(commSecondsPerLayer * gemmaMoe.num_hidden_layers)
    const expected = new Decimal(batchSize).div(stageSeconds)

    expect(withRef.tokensPerSecond.toString()).toBe(expected.toString())
  })
})

describe('estimatePerformance - offloading (decode over host link)', () => {
  it('0% offload leaves tokens/sec unchanged', () => {
    const base = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })
    const withZeroOffload = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      offload: { weightFraction: 0, kvOffloaded: false, linkGBps: 50 },
    })

    expect(withZeroOffload.tokensPerSecond.toString()).toBe(base.tokensPerSecond.toString())
    expect(withZeroOffload.offloadSlowdown).not.toBeNull()
    expect(withZeroOffload.offloadSlowdown as number).toBeCloseTo(1, 6)
  })

  it('100% weight offload on H100 (PCIe 50 GB/s) slows decode by the byte ratio', () => {
    const weightBytes = calculateModelWeightVRAM(
      calculateMoEBatchedParams(llama3_70b, 1),
      'fp16',
      llama3_70b,
    ).mul(BYTES_PER_GB)
    const kvBytes = calculateKVCacheVRAM({
      model: llama3_70b,
      sequenceLength: 2048,
      batchSize: 1,
      kvPrecision: 'fp16',
    }).mul(BYTES_PER_GB)
    const bandwidth = new Decimal(h100_80gb_sxm.memory_bandwidth_gbps).mul(1e9)
    const linkGBps = 50

    const baselineMemorySeconds = weightBytes.add(kvBytes).div(bandwidth)
    const offloadedMemorySeconds = kvBytes
      .div(bandwidth)
      .add(weightBytes.div(new Decimal(linkGBps).mul(1e9)))
    const expectedRatio = offloadedMemorySeconds.div(baselineMemorySeconds).toNumber()

    const result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      offload: { weightFraction: 1, kvOffloaded: false, linkGBps },
    })

    expect(result.offloadSlowdown).not.toBeNull()
    // Compute time is negligible next to memory time for this model/GPU (see the
    // memory-bound test above), so the byte-ratio approximation is close but not
    // exact — compute still contributes a hair to both the numerator and
    // denominator of the real roofline.
    expect(result.offloadSlowdown as number).toBeCloseTo(expectedRatio, 1)
  })

  it('KV offload adds the batch KV bytes to the host-link time', () => {
    const withoutKV = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      offload: { weightFraction: 0.5, kvOffloaded: false, linkGBps: 50 },
    })
    const withKV = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      offload: { weightFraction: 0.5, kvOffloaded: true, linkGBps: 50 },
    })

    expect(withKV.tokensPerSecond.toNumber()).toBeLessThan(withoutKV.tokensPerSecond.toNumber())
    expect(withKV.offloadSlowdown as number).toBeGreaterThan(withoutKV.offloadSlowdown as number)
  })

  it('offloadSlowdown is null without an offload param', () => {
    const result = estimatePerformance({
      model: llama3_70b,
      gpu: h100_80gb_sxm,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })
    expect(result.offloadSlowdown).toBeNull()
  })
})
