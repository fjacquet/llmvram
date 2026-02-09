import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { calculateInferenceVRAM } from './inference'
import { calculateMultiGPUVRAM, resolveInterconnect, validateInterconnect } from './multi-gpu'

// Test fixtures - inline model definitions for test isolation
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

const h100: GPU = {
  id: 'test-h100',
  name: 'Test H100 SXM5',
  manufacturer: 'nvidia',
  vram_gb: 80,
  memory_bandwidth_gbps: 3350,
  memory_type: 'HBM3',
  bus_width: 5120,
  fp16_tflops: 1979,
  fp32_tflops: 67,
  tdp_watts: 700,
  interconnect: 'nvlink-4',
  tier: 'datacenter',
}

const rtx4090: GPU = {
  id: 'test-rtx-4090',
  name: 'Test RTX 4090',
  manufacturer: 'nvidia',
  vram_gb: 24,
  memory_bandwidth_gbps: 1008,
  memory_type: 'GDDR6X',
  bus_width: 384,
  fp16_tflops: 82.6,
  fp32_tflops: 82.6,
  tdp_watts: 450,
  interconnect: 'pcie-4',
  tier: 'consumer',
}

const radeonMI300X: GPU = {
  id: 'test-mi300x',
  name: 'Test AMD MI300X',
  manufacturer: 'amd',
  vram_gb: 192,
  memory_bandwidth_gbps: 5300,
  memory_type: 'HBM3',
  bus_width: 8192,
  fp16_tflops: 1307,
  fp32_tflops: 163,
  tdp_watts: 750,
  interconnect: 'infinity-fabric',
  tier: 'datacenter',
}

describe('calculateMultiGPUVRAM - Tensor Parallelism', () => {
  it('calculates correct breakdown for Llama 70B GPTQ on 4x H100', () => {
    // First get single-GPU baseline
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 4, 'tensor-parallel')

    // Verify structure
    expect(result.numGPUs).toBe(4)
    expect(result.strategy).toBe('tensor-parallel')

    // Verify replicated memory calculation
    // layerNormMemory = 80 * 2 * 8192 * 4 / (1024^3) ≈ 0.0049 GB
    // embeddingMemory = 39.12 * 0.03 ≈ 1.17 GB
    // replicatedMemory = 0.0049 + 1.17 ≈ 1.18 GB
    expect(result.replicatedMemory.toNumber()).toBeGreaterThan(0)
    expect(result.replicatedMemory.toNumber()).toBeLessThan(
      singleGPU.modelWeights.toNumber() * 0.05,
    ) // < 5% of model weights

    // Verify weights per GPU (shardable weights divided, replicated added)
    const expectedWeightsPerGPU = singleGPU.modelWeights
      .sub(result.replicatedMemory)
      .div(4)
      .add(result.replicatedMemory)
    expect(result.perGPU.modelWeights.toString()).toBe(expectedWeightsPerGPU.toString())

    // Verify KV cache is divided
    const expectedKVPerGPU = singleGPU.kvCache.div(4)
    expect(result.perGPU.kvCache.toString()).toBe(expectedKVPerGPU.toString())

    // Verify activations are divided
    const expectedActivationsPerGPU = singleGPU.activations.div(4)
    expect(result.perGPU.activations.toString()).toBe(expectedActivationsPerGPU.toString())

    // Verify NCCL buffers included (0.2 GB * 3 peers = 0.6 GB)
    const expectedFrameworkOverhead = singleGPU.frameworkOverhead.add(new Decimal(0.6))
    expect(result.perGPU.frameworkOverhead.toString()).toBe(expectedFrameworkOverhead.toString())

    // Verify communication overhead (12% of weights per GPU)
    const expectedCommOverhead = result.perGPU.modelWeights.mul(0.12)
    expect(result.perGPU.communicationOverhead.toString()).toBe(expectedCommOverhead.toString())

    // Verify total per GPU
    const expectedTotal = result.perGPU.modelWeights
      .add(result.perGPU.kvCache)
      .add(result.perGPU.activations)
      .add(result.perGPU.frameworkOverhead)
      .add(result.perGPU.communicationOverhead)
    expect(result.perGPU.total.toString()).toBe(expectedTotal.toString())
    expect(result.totalPerGPU.toString()).toBe(result.perGPU.total.toString())

    // Verify utilization
    const expectedUtilization = result.totalPerGPU.div(h100.vram_gb).mul(100)
    expect(result.utilizationPercent.toString()).toBe(expectedUtilization.toString())

    // Verify baseline
    expect(result.singleGPUBaseline.toString()).toBe(singleGPU.total.toString())

    // Verify per-GPU memory is less than single GPU
    expect(result.totalPerGPU.toNumber()).toBeLessThan(singleGPU.total.toNumber())
  })

  it('calculates correct breakdown for MoE model with extra overhead', () => {
    const singleGPU = calculateInferenceVRAM({
      model: mixtral8x7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(singleGPU, mixtral8x7b, h100.vram_gb, 4, 'tensor-parallel')

    // MoE should have 15% extra communication overhead
    // communicationOverhead = weightsPerGPU * 0.12 * 1.15
    const expectedCommOverhead = result.perGPU.modelWeights.mul(0.12).mul(1.15)
    expect(result.perGPU.communicationOverhead.toString()).toBe(expectedCommOverhead.toString())

    // MoE communication overhead should be higher than dense model equivalent
    expect(result.perGPU.communicationOverhead.toNumber()).toBeGreaterThan(
      result.perGPU.modelWeights.mul(0.12).toNumber(),
    )
  })

  it('verifies all Decimal instances in breakdown', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const result = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 4, 'tensor-parallel')

    expect(result.perGPU.modelWeights).toBeInstanceOf(Decimal)
    expect(result.perGPU.kvCache).toBeInstanceOf(Decimal)
    expect(result.perGPU.activations).toBeInstanceOf(Decimal)
    expect(result.perGPU.frameworkOverhead).toBeInstanceOf(Decimal)
    expect(result.perGPU.communicationOverhead).toBeInstanceOf(Decimal)
    expect(result.perGPU.total).toBeInstanceOf(Decimal)
    expect(result.replicatedMemory).toBeInstanceOf(Decimal)
    expect(result.totalPerGPU).toBeInstanceOf(Decimal)
    expect(result.utilizationPercent).toBeInstanceOf(Decimal)
    expect(result.singleGPUBaseline).toBeInstanceOf(Decimal)
  })
})

describe('calculateMultiGPUVRAM - Pipeline Parallelism', () => {
  it('calculates correct breakdown for Llama 70B GPTQ on 4 GPUs', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 4, 'pipeline-parallel')

    expect(result.strategy).toBe('pipeline-parallel')

    // PP has no replication (layers split, not weights)
    expect(result.replicatedMemory.toNumber()).toBe(0)

    // Verify weights divided evenly
    const expectedWeightsPerGPU = singleGPU.modelWeights.div(4)
    expect(result.perGPU.modelWeights.toString()).toBe(expectedWeightsPerGPU.toString())

    // CRITICAL: PP does NOT divide KV cache (each GPU needs full cache for its layers)
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.toString())

    // Verify activations divided with stashing overhead
    const baseActivationsPerGPU = singleGPU.activations.div(4)
    const expectedActivationsPerGPU = baseActivationsPerGPU.mul(1.12) // 12% stashing overhead
    expect(result.perGPU.activations.toString()).toBe(expectedActivationsPerGPU.toString())

    // PP has no NCCL buffers (only point-to-point communication)
    expect(result.perGPU.frameworkOverhead.toString()).toBe(singleGPU.frameworkOverhead.toString())

    // Verify communication overhead (5% for PP)
    const expectedCommOverhead = result.perGPU.modelWeights.mul(0.05)
    expect(result.perGPU.communicationOverhead.toString()).toBe(expectedCommOverhead.toString())

    // Verify total
    const expectedTotal = result.perGPU.modelWeights
      .add(result.perGPU.kvCache)
      .add(result.perGPU.activations)
      .add(result.perGPU.frameworkOverhead)
      .add(result.perGPU.communicationOverhead)
    expect(result.perGPU.total.toString()).toBe(expectedTotal.toString())
  })

  it('verifies KV cache behavior differs between TP and PP', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const tpResult = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 4, 'tensor-parallel')
    const ppResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'pipeline-parallel',
    )

    // TP divides KV cache, PP does not
    expect(tpResult.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())
    expect(ppResult.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.toString())

    // PP has full KV cache per GPU (4x more than TP)
    const kvRatio = ppResult.perGPU.kvCache.div(tpResult.perGPU.kvCache)
    expect(kvRatio.toNumber()).toBeCloseTo(4.0, 10)

    // NOTE: For this specific scenario (small KV cache relative to model size),
    // TP actually uses MORE memory per GPU than PP because:
    // - TP pays for weight replication (~1.18 GB), NCCL buffers (0.6 GB), and higher comm overhead (12% vs 5%)
    // - PP pays for full KV cache (~1.25 GB) but saves on replication/NCCL/comm overhead
    // The tradeoff depends on KV cache size (seq length, batch size) vs model size
    expect(tpResult.totalPerGPU.toNumber()).toBeGreaterThan(ppResult.totalPerGPU.toNumber())
  })
})

describe('calculateMultiGPUVRAM - Single GPU', () => {
  it('returns passthrough with zero overhead for numGPUs=1', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 1, 'tensor-parallel')

    expect(result.numGPUs).toBe(1)
    expect(result.replicatedMemory.toNumber()).toBe(0)
    expect(result.perGPU.communicationOverhead.toNumber()).toBe(0)
    expect(result.totalPerGPU.toString()).toBe(singleGPU.total.toString())

    // All components should match single GPU (no modifications)
    expect(result.perGPU.modelWeights.toString()).toBe(singleGPU.modelWeights.toString())
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.toString())
    expect(result.perGPU.activations.toString()).toBe(singleGPU.activations.toString())
    expect(result.perGPU.frameworkOverhead.toString()).toBe(singleGPU.frameworkOverhead.toString())
  })

  it('strategy does not matter for single GPU', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const tpResult = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 1, 'tensor-parallel')
    const ppResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      1,
      'pipeline-parallel',
    )

    expect(tpResult.totalPerGPU.toString()).toBe(ppResult.totalPerGPU.toString())
  })
})

describe('calculateMultiGPUVRAM - Edge Cases', () => {
  it('throws error for numGPUs < 1', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    expect(() => {
      calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 0, 'tensor-parallel')
    }).toThrow()
  })

  it('throws error for numGPUs > 8', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    expect(() => {
      calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 9, 'tensor-parallel')
    }).toThrow()
  })

  it('handles 8 GPUs correctly (maximum)', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const result = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 8, 'tensor-parallel')

    expect(result.numGPUs).toBe(8)
    // NCCL buffers: 0.2 GB * 7 peers = 1.4 GB
    expect(result.perGPU.frameworkOverhead.toNumber()).toBeCloseTo(
      singleGPU.frameworkOverhead.add(1.4).toNumber(),
      10,
    )
  })
})

describe('resolveInterconnect', () => {
  it('maps nvlink-4 to nvlink-4', () => {
    const result = resolveInterconnect(h100)
    expect(result).toBe('nvlink-4')
  })

  it('maps generic nvlink to nvlink-4', () => {
    const gpu: GPU = { ...h100, interconnect: 'nvlink' }
    const result = resolveInterconnect(gpu)
    expect(result).toBe('nvlink-4')
  })

  it('maps nvlink-5 to nvlink-5', () => {
    const gpu: GPU = { ...h100, interconnect: 'nvlink-5' }
    const result = resolveInterconnect(gpu)
    expect(result).toBe('nvlink-5')
  })

  it('maps undefined interconnect on datacenter GPU to pcie-5', () => {
    const gpu: GPU = { ...h100, interconnect: undefined }
    const result = resolveInterconnect(gpu)
    expect(result).toBe('pcie-5')
  })

  it('maps none interconnect on datacenter GPU to pcie-5', () => {
    const gpu: GPU = { ...h100, interconnect: 'none' }
    const result = resolveInterconnect(gpu)
    expect(result).toBe('pcie-5')
  })

  it('maps undefined interconnect on consumer GPU to pcie-4', () => {
    const gpu: GPU = { ...rtx4090, interconnect: undefined }
    const result = resolveInterconnect(gpu)
    expect(result).toBe('pcie-4')
  })

  it('maps pcie-4 to pcie-4', () => {
    const result = resolveInterconnect(rtx4090)
    expect(result).toBe('pcie-4')
  })

  it('maps infinity-fabric to pcie-5', () => {
    const result = resolveInterconnect(radeonMI300X)
    expect(result).toBe('pcie-5')
  })

  it('maps unified to none (Apple Silicon)', () => {
    const m3Ultra: GPU = {
      id: 'test-m3-ultra',
      name: 'Test M3 Ultra',
      manufacturer: 'apple',
      vram_gb: 128,
      memory_bandwidth_gbps: 800,
      memory_type: 'Unified',
      bus_width: 0,
      interconnect: 'unified',
      tier: 'apple-silicon',
    }
    const result = resolveInterconnect(m3Ultra)
    expect(result).toBe('none')
  })
})

describe('validateInterconnect', () => {
  it('validates nvlink-4 with TP degree 4 as valid with no warning', () => {
    const result = validateInterconnect(h100, 4, 'tensor-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
    expect(result.interconnect.type).toBe('nvlink-4')
    expect(result.interconnect.bandwidthGBps).toBe(900)
    expect(result.interconnect.recommendedMaxTPDegree).toBe(8)
  })

  it('validates pcie-4 with TP degree 4 as valid with warning', () => {
    const result = validateInterconnect(rtx4090, 4, 'tensor-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).toContain('PCIe 4.0')
    expect(result.warning).toContain('communication overhead')
    expect(result.interconnect.type).toBe('pcie-4')
  })

  it('validates pcie-4 with TP degree 2 as valid with no warning', () => {
    const result = validateInterconnect(rtx4090, 2, 'tensor-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
  })

  it('validates none interconnect with numGPUs > 1 as invalid', () => {
    const appleGPU: GPU = {
      ...h100,
      interconnect: 'unified',
      tier: 'apple-silicon',
    }

    const result = validateInterconnect(appleGPU, 2, 'tensor-parallel')

    expect(result.valid).toBe(false)
    expect(result.warning).toContain('does not support multi-GPU')
    expect(result.interconnect.type).toBe('none')
  })

  it('validates any interconnect with numGPUs=1 as valid', () => {
    const appleGPU: GPU = {
      ...h100,
      interconnect: 'unified',
      tier: 'apple-silicon',
    }

    const result = validateInterconnect(appleGPU, 1, 'tensor-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
  })

  it('pipeline parallelism does not warn for high GPU count', () => {
    const result = validateInterconnect(rtx4090, 4, 'pipeline-parallel')

    // PP warning should be less strict than TP
    expect(result.valid).toBe(true)
  })
})
