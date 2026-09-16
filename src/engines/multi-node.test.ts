import { FABRIC_SPECS } from '@engines/fabric'
import { calculateInferenceVRAM } from '@engines/inference'
import { calculateMultiGPUVRAM } from '@engines/multi-gpu'
import { calculateMultiNodeVRAM } from '@engines/multi-node'
import type { GPU, Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'

// Test fixture — matches the shape used in multi-gpu.test.ts, which is kept in
// sync with the current Zod schema (ModelSchema has no vocab_size or
// max_position_embeddings field).
const llama70b: Model = {
  id: 'test-llama-3-70b',
  name: 'Test Llama 3 70B',
  architecture: 'dense',
  num_parameters_billion: 70,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,
  intermediate_size: 28672,
}

const mi355x: GPU = {
  id: 'amd-mi355x',
  name: 'AMD Instinct MI355X',
  manufacturer: 'amd',
  vram_gb: 288,
  memory_bandwidth_gbps: 8000,
  memory_type: 'HBM3E',
  bus_width: 8192,
  fp16_tflops: 2516,
  fp32_tflops: 157,
  tdp_watts: 1400,
  interconnect: 'infinity-fabric',
  tier: 'datacenter',
}

const singleGPU = calculateInferenceVRAM({
  model: llama70b,
  quantization: 'fp16',
  sequenceLength: 4096,
  batchSize: 1,
})

const base = {
  singleGPU,
  model: llama70b,
  gpuVramGB: 288,
  intraNodeStrategy: 'tensor-parallel' as const,
  gpu: mi355x,
  fabric: FABRIC_SPECS['ethernet-800g'],
  batchSize: 1,
}

describe('calculateMultiNodeVRAM', () => {
  it('is an exact passthrough at one node — the regression guard', () => {
    const multiNode = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 1 })
    const multiGPU = calculateMultiGPUVRAM(singleGPU, llama70b, 288, 8, 'tensor-parallel', mi355x)
    expect(multiNode.totalPerGPU.toString()).toBe(multiGPU.totalPerGPU.toString())
    expect(multiNode.scalingEfficiency).toBe(multiGPU.scalingEfficiency)
    expect(multiNode.prefillScalingEfficiency).toBe(multiGPU.prefillScalingEfficiency)
  })

  it('reports the topology', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.numNodes).toBe(4)
    expect(result.gpusPerNode).toBe(8)
    expect(result.numGPUs).toBe(32)
  })

  it('divides weights by the total GPU count across both levels', () => {
    const oneNode = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 1 })
    const fourNodes = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(fourNodes.perGPU.modelWeights.lessThan(oneNode.perGPU.modelWeights)).toBe(true)
  })

  it('shards the KV cache across nodes as well as within them', () => {
    const oneNode = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 1 })
    const fourNodes = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    // Four nodes split the layers four ways before the intra-node TP split.
    expect(fourNodes.perGPU.kvCache.toNumber()).toBeCloseTo(
      oneNode.perGPU.kvCache.toNumber() / 4,
      6,
    )
  })

  it('separates decode from prefill efficiency once nodes > 1', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.interNodeDecodeEfficiency).toBeGreaterThan(result.interNodePrefillEfficiency)
    expect(result.scalingEfficiency).toBeGreaterThan(result.prefillScalingEfficiency)
  })

  it('charges a pipeline bubble at batch 1 across nodes', () => {
    const result = calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 4 })
    expect(result.bubbleEfficiency).toBeCloseTo(0.571, 3)
  })

  it('rewards a faster fabric', () => {
    const slow = calculateMultiNodeVRAM({
      ...base,
      gpusPerNode: 8,
      numNodes: 4,
      fabric: FABRIC_SPECS['ethernet-100g'],
    })
    const fast = calculateMultiNodeVRAM({
      ...base,
      gpusPerNode: 8,
      numNodes: 4,
      fabric: FABRIC_SPECS['ethernet-1600g'],
    })
    expect(fast.prefillScalingEfficiency).toBeGreaterThan(slow.prefillScalingEfficiency)
  })

  it('rejects a node count below 1', () => {
    expect(() => calculateMultiNodeVRAM({ ...base, gpusPerNode: 8, numNodes: 0 })).toThrow(
      /numNodes/,
    )
  })

  it('still rejects more than 8 GPUs in one node', () => {
    expect(() => calculateMultiNodeVRAM({ ...base, gpusPerNode: 9, numNodes: 2 })).toThrow(
      /numGPUs must be between 1 and 8/,
    )
  })

  it('computes total from the divided parts, not the undivided single-GPU total (regression for the spread defect)', () => {
    const oneNodeOneGPU = calculateMultiNodeVRAM({ ...base, gpusPerNode: 1, numNodes: 1 })
    const fourNodesOneGPU = calculateMultiNodeVRAM({ ...base, gpusPerNode: 1, numNodes: 4 })

    // 1 GPU per server across 4 servers must actually shrink relative to a
    // single node of 1 GPU — the ...singleGPU spread in stageBreakdown must
    // not leak the undivided `total` through calculateMultiGPUVRAM's
    // numGPUs === 1 passthrough branch.
    expect(fourNodesOneGPU.totalPerGPU.lessThan(oneNodeOneGPU.totalPerGPU)).toBe(true)
    expect(fourNodesOneGPU.perGPU.total.lessThan(oneNodeOneGPU.perGPU.total)).toBe(true)
  })
})
