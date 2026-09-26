import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { INTERCONNECT_SPECS } from './constants'
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
  max_gpus_per_node: 8,
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
  max_gpus_per_node: 8,
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
  max_gpus_per_node: 8,
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

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'tensor-parallel',
      h100,
    )

    // Verify structure
    expect(result.numGPUs).toBe(4)
    expect(result.strategy).toBe('tensor-parallel')

    // Verify replicated memory calculation (layer norms only; embeddings shard)
    // layerNormMemory = 80 * 2 * 8192 * 4 / (1024^3) ≈ 0.0049 GB
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

    // One framework context per rank
    expect(result.perGPU.frameworkOverhead.toString()).toBe(singleGPU.frameworkOverhead.toString())

    // Communication memory is the NCCL buffers (flat 0.25 GB per GPU, not per peer)
    expect(result.perGPU.communicationOverhead.toString()).toBe('0.25')

    // Verify interconnect fields
    expect(result.scalingEfficiency).toBe(0.92)
    expect(result.interconnectBandwidthGBps).toBe(900)

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

  it('charges MoE the same communication memory as dense', () => {
    const singleGPU = calculateInferenceVRAM({
      model: mixtral8x7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(
      singleGPU,
      mixtral8x7b,
      h100.vram_gb,
      4,
      'tensor-parallel',
      h100,
    )

    // Expert routing costs throughput, not memory: only the NCCL buffers count
    expect(result.perGPU.communicationOverhead.toString()).toBe('0.25')
  })

  it('verifies all Decimal instances in breakdown', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'tensor-parallel',
      h100,
    )

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

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'pipeline-parallel',
      h100,
    )

    expect(result.strategy).toBe('pipeline-parallel')

    // PP has no replication (layers split, not weights)
    expect(result.replicatedMemory.toNumber()).toBe(0)

    // Verify weights divided evenly
    const expectedWeightsPerGPU = singleGPU.modelWeights.div(4)
    expect(result.perGPU.modelWeights.toString()).toBe(expectedWeightsPerGPU.toString())

    // PP shards KV cache by layer: each stage holds only its own layers' cache
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())

    // Verify activations divided with stashing overhead
    const baseActivationsPerGPU = singleGPU.activations.div(4)
    const expectedActivationsPerGPU = baseActivationsPerGPU.mul(1.12) // 12% stashing overhead
    expect(result.perGPU.activations.toString()).toBe(expectedActivationsPerGPU.toString())

    // PP has no NCCL buffers (only point-to-point communication)
    expect(result.perGPU.frameworkOverhead.toString()).toBe(singleGPU.frameworkOverhead.toString())

    // Stage sends reuse activation buffers: no extra communication memory
    expect(result.perGPU.communicationOverhead.toString()).toBe('0')

    // Verify total
    const expectedTotal = result.perGPU.modelWeights
      .add(result.perGPU.kvCache)
      .add(result.perGPU.activations)
      .add(result.perGPU.frameworkOverhead)
      .add(result.perGPU.communicationOverhead)
    expect(result.perGPU.total.toString()).toBe(expectedTotal.toString())
  })

  it('matches TP on KV cache but differs on other overheads', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const tpResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'tensor-parallel',
      h100,
    )
    const ppResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'pipeline-parallel',
      h100,
    )

    // TP shards KV cache by head, PP shards it by layer — both divide by numGPUs
    expect(tpResult.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())
    expect(ppResult.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())

    // PP and TP shard the KV cache identically; they differ elsewhere (replication, NCCL, comm %)
    const kvRatio = ppResult.perGPU.kvCache.div(tpResult.perGPU.kvCache)
    expect(kvRatio.toNumber()).toBe(1)

    // TP uses more memory per GPU than PP here: it replicates layer norms and
    // pays NCCL buffers, while PP splits every weight and holds neither.
    expect(tpResult.totalPerGPU.toNumber()).toBeGreaterThan(ppResult.totalPerGPU.toNumber())
  })
})

describe('pipeline parallel KV cache sharding', () => {
  it('divides the KV cache across stages, because layers are split', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'pipeline-parallel',
      h100,
    )
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())
  })

  it('matches tensor parallel on the KV term — both shard it, by different axes', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const tp = calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 4, 'tensor-parallel', h100)
    const pp = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'pipeline-parallel',
      h100,
    )
    expect(pp.perGPU.kvCache.toString()).toBe(tp.perGPU.kvCache.toString())
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

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      1,
      'tensor-parallel',
      h100,
    )

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

    const tpResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      1,
      'tensor-parallel',
      h100,
    )
    const ppResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      1,
      'pipeline-parallel',
      h100,
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
      calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 0, 'tensor-parallel', h100)
    }).toThrow()
  })

  it('throws error for numGPUs > 72', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    expect(() =>
      calculateMultiGPUVRAM(singleGPU, llama70b, h100.vram_gb, 73, 'tensor-parallel', h100),
    ).toThrow(/numGPUs must be between 1 and 72/)
  })

  it('accepts a 72-GPU node, for NVL72-class racks', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 288, 72, 'tensor-parallel', h100)
    expect(result.numGPUs).toBe(72)
    expect(result.totalPerGPU.toNumber()).toBeGreaterThan(0)
    expect(result.totalPerGPU.isFinite()).toBe(true)
  })

  it('handles 8 GPUs correctly (legacy 8-GPU node)', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      8,
      'tensor-parallel',
      h100,
    )

    expect(result.numGPUs).toBe(8)

    // NCCL buffers do not grow with the group. Ring/tree allreduce gives each
    // rank a fixed handful of connections however many ranks there are, so the
    // same figure must appear at 8 and at 72. The previous per-peer model put
    // 1.4 GB here and 14.2 GB at 72 — more than KV cache and activations
    // combined — which is what this asserts against.
    const at72 = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      72,
      'tensor-parallel',
      h100,
    )
    const ncclAt8 = result.perGPU.communicationOverhead
    const ncclAt72 = at72.perGPU.communicationOverhead

    expect(ncclAt8.toString()).toBe(ncclAt72.toString())

    // ...and sits inside the 100-500 MB/GPU band CLAUDE.md documents.
    expect(ncclAt8.toNumber()).toBeGreaterThanOrEqual(0.1)
    expect(ncclAt8.toNumber()).toBeLessThanOrEqual(0.5)
  })

  it('adds no NCCL buffers at all on a single GPU (passthrough, not the TP path)', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
    })

    // One GPU forms no communicator. This is guaranteed by the single-GPU
    // passthrough in calculateMultiGPUVRAM, which returns before the tensor
    // parallel path runs — NOT by a guard on the NCCL term, which is why adding
    // such a guard changes nothing and this test cannot detect its absence.
    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      1,
      'tensor-parallel',
      h100,
    )

    expect(result.perGPU.communicationOverhead.toString()).toBe('0')
  })
})

describe('calculateMultiGPUVRAM - Bandwidth-aware overhead', () => {
  it('scales NVLink-4 better than PCIe-4, without charging it as memory', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const nvlinkResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'tensor-parallel',
      h100, // nvlink-4
    )
    const pcieResult = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      rtx4090.vram_gb,
      4,
      'tensor-parallel',
      rtx4090, // pcie-4
    )

    // NVLink-4: 8% overhead, PCIe-4: 35% overhead
    expect(nvlinkResult.scalingEfficiency).toBe(0.92)
    expect(pcieResult.scalingEfficiency).toBe(0.65)
    expect(nvlinkResult.interconnectBandwidthGBps).toBe(900)
    expect(pcieResult.interconnectBandwidthGBps).toBe(64)

    // Memory is the same NCCL buffers on both
    expect(nvlinkResult.perGPU.communicationOverhead.toString()).toBe(
      pcieResult.perGPU.communicationOverhead.toString(),
    )
  })

  it('returns scalingEfficiency=1.0 for single GPU passthrough', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const result = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      1,
      'tensor-parallel',
      h100,
    )

    expect(result.scalingEfficiency).toBe(1.0)
    expect(result.interconnectBandwidthGBps).toBe(0)
  })

  it('PP always returns flat 95% scaling efficiency regardless of interconnect', () => {
    const singleGPU = calculateInferenceVRAM({
      model: llama70b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
      kvQuantization: 'fp16',
    })

    const ppH100 = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      h100.vram_gb,
      4,
      'pipeline-parallel',
      h100,
    )
    const ppRTX = calculateMultiGPUVRAM(
      singleGPU,
      llama70b,
      rtx4090.vram_gb,
      4,
      'pipeline-parallel',
      rtx4090,
    )

    // PP efficiency is always 0.95 (flat, regardless of interconnect)
    expect(ppH100.scalingEfficiency).toBeCloseTo(0.95, 10)
    expect(ppRTX.scalingEfficiency).toBeCloseTo(0.95, 10)
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

  it('maps infinity-fabric to its own type', () => {
    const result = resolveInterconnect(radeonMI300X)
    expect(result).toBe('infinity-fabric')
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
      max_gpus_per_node: 1,
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
    expect(result.warning).toContain('PCIe 4')
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
      max_gpus_per_node: 1,
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
      max_gpus_per_node: 1,
    }

    const result = validateInterconnect(appleGPU, 1, 'tensor-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
  })

  it('pipeline parallelism stays valid (not invalid) for high GPU count', () => {
    const result = validateInterconnect(rtx4090, 4, 'pipeline-parallel')

    // PP never invalidates the config the way 'none' interconnect does — it
    // only warns, and only past the interconnect's recommended max degree.
    expect(result.valid).toBe(true)
  })

  it('warns for pipeline parallelism beyond the recommended max, with PP-specific wording', () => {
    // GB300 NVL72-class part: nvlink-5, recommendedMaxTPDegree 8, run at 72-way PP.
    // This regime was unreachable before max_gpus_per_node replaced the flat
    // 8-GPU guard — nothing on screen used to qualify a 72-way pipeline.
    const nvl72Like: GPU = { ...h100, interconnect: 'nvlink-5', max_gpus_per_node: 72 }
    const result = validateInterconnect(nvl72Like, 72, 'pipeline-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).not.toBeNull()
    expect(result.warning).toContain('72 GPUs')
    // PP-specific: bubble/stage-imbalance overhead, not the TP sentence about
    // link-bandwidth communication overhead (asserted verbatim elsewhere).
    expect(result.warning).not.toContain('communication overhead')
  })

  it('does not warn for pipeline parallelism within the recommended max', () => {
    const result = validateInterconnect(h100, 8, 'pipeline-parallel')

    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
  })
})

describe('Infinity Fabric interconnect', () => {
  const mi300x: GPU = {
    id: 'amd-mi300x',
    name: 'AMD MI300X',
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
    max_gpus_per_node: 8,
  }

  it('resolves to its own type, not pcie-5', () => {
    expect(resolveInterconnect(mi300x)).toBe('infinity-fabric')
  })

  it('carries AMD bidirectional bandwidth and 8-way TP support', () => {
    const spec = INTERCONNECT_SPECS['infinity-fabric']
    expect(spec.bandwidthGBps).toBe(1075)
    expect(spec.recommendedMaxTPDegree).toBe(8)
    expect(spec.tpScalingEfficiency).toBe(0.93)
  })

  it('sits between NVLink-4 and NVLink-5 in scaling efficiency', () => {
    expect(INTERCONNECT_SPECS['infinity-fabric'].tpScalingEfficiency).toBeGreaterThan(
      INTERCONNECT_SPECS['nvlink-4'].tpScalingEfficiency,
    )
    expect(INTERCONNECT_SPECS['infinity-fabric'].tpScalingEfficiency).toBeLessThan(
      INTERCONNECT_SPECS['nvlink-5'].tpScalingEfficiency,
    )
  })

  it('does not warn at 8-way tensor parallel', () => {
    const result = validateInterconnect(mi300x, 8, 'tensor-parallel')
    expect(result.valid).toBe(true)
    expect(result.warning).toBeNull()
  })

  it('names the interconnect in a warning rather than printing the enum value', () => {
    const pcie4GPU: GPU = { ...mi300x, interconnect: 'pcie-4', name: 'Test PCIe4' }
    const result = validateInterconnect(pcie4GPU, 8, 'tensor-parallel')
    expect(result.warning).toContain('PCIe 4')
  })
})

describe('node dimension defaults', () => {
  const singleGPU = calculateInferenceVRAM({
    model: llama70b,
    quantization: 'gptq',
    sequenceLength: 4096,
    batchSize: 1,
    kvQuantization: 'fp16',
  })

  it('reports a single node with all GPUs in it', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100)
    expect(result.numNodes).toBe(1)
    expect(result.gpusPerNode).toBe(4)
  })

  it('leaves every inter-node term neutral', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100)
    expect(result.interNodeDecodeEfficiency).toBe(1)
    expect(result.interNodePrefillEfficiency).toBe(1)
    expect(result.bubbleEfficiency).toBe(1)
  })

  it('makes prefill and decode efficiency identical within one node', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 4, 'tensor-parallel', h100)
    expect(result.prefillScalingEfficiency).toBe(result.scalingEfficiency)
    expect(result.intraNodeEfficiency).toBe(result.scalingEfficiency)
  })

  it('holds for the single-GPU passthrough too', () => {
    const result = calculateMultiGPUVRAM(singleGPU, llama70b, 80, 1, 'tensor-parallel', h100)
    expect(result.numNodes).toBe(1)
    expect(result.gpusPerNode).toBe(1)
    expect(result.scalingEfficiency).toBe(1)
    expect(result.prefillScalingEfficiency).toBe(1)
  })
})

// vLLM is the authority for serving memory. Sources (docs.vllm.ai, stable):
// - ModelConfig.get_num_kv_heads: `if self.use_mla: return 1`, else
//   `max(1, total_num_kv_heads // tensor_parallel_size)` (heads replicated past TP > H)
// - lmcache_mp_connector: "Tensor parallel does not change the KV caches for MLA models"
// - VocabParallelEmbedding / ParallelLMHead: the vocabulary is split across TP ranks
describe('tensor parallel memory follows vLLM', () => {
  const tp = (model: Model, numGPUs: number) => {
    const singleGPU = calculateInferenceVRAM({
      model,
      quantization: 'fp16',
      sequenceLength: 32768,
      batchSize: 1,
    })
    const nvl72Like: GPU = { ...h100, interconnect: 'nvlink-5', max_gpus_per_node: 72 }
    return {
      singleGPU,
      result: calculateMultiGPUVRAM(singleGPU, model, 288, numGPUs, 'tensor-parallel', nvl72Like),
    }
  }

  it('splits GQA KV by kv heads, not below one head per GPU', () => {
    const fourKVHeads: Model = { ...llama70b, num_kv_heads: 4 }
    const { singleGPU, result } = tp(fourKVHeads, 8)
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(4).toString())
  })

  it('splits KV fully while TP stays within the kv head count', () => {
    const { singleGPU, result } = tp(llama70b, 8)
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.div(8).toString())
  })

  it('duplicates MLA KV on every TP rank', () => {
    const mla: Model = { ...mixtral8x7b, num_kv_heads: 32, use_mla: true }
    const { singleGPU, result } = tp(mla, 8)
    expect(result.perGPU.kvCache.toString()).toBe(singleGPU.kvCache.toString())
  })

  it('shards embeddings: only layer norms are replicated', () => {
    const { result } = tp(llama70b, 8)
    // 80 layers x 2 norms x 8192 x 4 bytes
    const layerNorms = new Decimal(80 * 2 * 8192 * 4).div(1024 ** 3)
    expect(result.replicatedMemory.toString()).toBe(layerNorms.toString())
  })

  it('charges each GPU one framework context plus NCCL buffers, whatever the TP degree', () => {
    const { result } = tp(llama70b, 72)
    expect(result.perGPU.frameworkOverhead.toNumber()).toBe(1.0)
    expect(result.perGPU.communicationOverhead.toNumber()).toBe(0.25)
  })

  it('does not count interconnect efficiency as memory', () => {
    const pcie: GPU = { ...rtx4090, vram_gb: 80 }
    const singleGPU = calculateInferenceVRAM({
      model: mixtral8x7b,
      quantization: 'fp16',
      sequenceLength: 2048,
      batchSize: 1,
    })
    const result = calculateMultiGPUVRAM(singleGPU, mixtral8x7b, 80, 4, 'tensor-parallel', pcie)
    expect(result.perGPU.communicationOverhead.toNumber()).toBe(0.25)
    expect(result.scalingEfficiency).toBeLessThan(1)
  })
})
