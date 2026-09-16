import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { PP_ACTIVATION_STASHING_OVERHEAD } from './constants'
import {
  fabricDecodeEfficiency,
  fabricPrefillEfficiency,
  perNodeFabricGBps,
  pipelineBubbleEfficiency,
} from './fabric'
import { calculateMultiGPUVRAM } from './multi-gpu'
import type {
  FabricSpec,
  InferenceVRAMBreakdown,
  MultiGPUVRAMBreakdown,
  ShardingStrategy,
} from './types'

/**
 * Calculate VRAM for inference spread across multiple identical servers
 *
 * Two-level topology: tensor parallelism inside a node, pipeline parallelism
 * across nodes. Tensor parallelism across a node boundary is deliberately not
 * offered — its per-layer allreduce over a fabric an order of magnitude slower
 * than NVLink or Infinity Fabric is not a configuration worth presenting as
 * viable.
 *
 * Composes over calculateMultiGPUVRAM rather than replacing it: this function
 * derives a per-stage breakdown, then hands the intra-node split to the
 * existing code. That keeps calculateMultiGPUVRAM's 1-8 GPU guard correct —
 * under composition it is exactly the per-node bound.
 *
 * @throws Error if numNodes < 1, or if gpusPerNode is outside 1-8 (raised by
 *         calculateMultiGPUVRAM)
 *
 * @example
 * ```ts
 * calculateMultiNodeVRAM({
 *   singleGPU, model: llama405b, gpuVramGB: 288,
 *   gpusPerNode: 8, numNodes: 4,
 *   intraNodeStrategy: 'tensor-parallel', gpu: mi355x,
 *   fabric: FABRIC_SPECS['ethernet-800g'], batchSize: 1,
 * })
 * ```
 */
export function calculateMultiNodeVRAM(params: {
  singleGPU: InferenceVRAMBreakdown
  model: Model
  gpuVramGB: number
  gpusPerNode: number
  numNodes: number
  intraNodeStrategy: ShardingStrategy
  gpu: GPU
  fabric: FabricSpec
  batchSize: number
}): MultiGPUVRAMBreakdown {
  const {
    singleGPU,
    model,
    gpuVramGB,
    gpusPerNode,
    numNodes,
    intraNodeStrategy,
    gpu,
    fabric,
    batchSize,
  } = params

  if (!Number.isInteger(numNodes) || numNodes < 1) {
    throw new Error(`numNodes must be an integer >= 1, got ${numNodes}`)
  }

  // Single node: delegate unchanged. This is the regression guard for the whole
  // feature — one node must produce byte-identical results to before it existed.
  if (numNodes === 1) {
    return calculateMultiGPUVRAM(singleGPU, model, gpuVramGB, gpusPerNode, intraNodeStrategy, gpu)
  }

  // Pipeline parallelism across nodes: each node owns a contiguous slice of
  // layers, so weights, KV cache and activations all divide by the node count.
  // Framework overhead does not — it is per-process (PyTorch + CUDA/ROCm
  // context) and every rank pays it in full.
  //
  // `total` is NOT carried through the ...singleGPU spread: that would be the
  // undivided single-GPU total. It must be recomputed from the post-division
  // parts, or a gpusPerNode: 1 configuration would silently pass the whole
  // single-GPU total through calculateMultiGPUVRAM's numGPUs === 1 branch,
  // which returns every perGPU field verbatim from the breakdown it is given.
  // PP_ACTIVATION_STASHING_OVERHEAD is a flat modelling fudge factor for
  // stashing activations across a pipeline stage boundary, not a per-level
  // physical quantity — it must be applied exactly once, not once per
  // pipeline level. calculateMultiGPUVRAM's pipeline-parallel path
  // (calculatePipelineParallelVRAM) applies this same constant itself, but
  // only when it actually runs the PP branch: at gpusPerNode === 1 it takes
  // the numGPUs === 1 early-return passthrough instead (see multi-gpu.ts),
  // which applies no stashing multiplier at all regardless of strategy. So
  // the inter-node stage here must apply it whenever the intra-node level
  // will NOT — i.e. every case except "pipeline-parallel with more than one
  // GPU per node". Skipping this check would compound the constant to
  // 1.12^2 for that one case.
  const stashing =
    intraNodeStrategy === 'pipeline-parallel' && gpusPerNode > 1
      ? new Decimal(1)
      : new Decimal(1).add(PP_ACTIVATION_STASHING_OVERHEAD)

  const modelWeights = singleGPU.modelWeights.div(numNodes)
  const kvCache = singleGPU.kvCache.div(numNodes)
  const activations = singleGPU.activations.div(numNodes).mul(stashing)
  const frameworkOverhead = singleGPU.frameworkOverhead

  const stageBreakdown: InferenceVRAMBreakdown = {
    modelWeights,
    kvCache,
    activations,
    frameworkOverhead,
    total: modelWeights.add(kvCache).add(activations).add(frameworkOverhead),
  }

  const inner = calculateMultiGPUVRAM(
    stageBreakdown,
    model,
    gpuVramGB,
    gpusPerNode,
    intraNodeStrategy,
    gpu,
  )

  const perNodeGBps = perNodeFabricGBps(fabric.portGBps, gpusPerNode)
  const interNodePrefillEfficiency = fabricPrefillEfficiency(perNodeGBps, fabric.classFactor)
  const interNodeDecodeEfficiency = fabricDecodeEfficiency(perNodeGBps)
  const bubbleEfficiency = pipelineBubbleEfficiency(batchSize, numNodes)
  const intraNodeEfficiency = inner.intraNodeEfficiency

  return {
    ...inner,
    numGPUs: gpusPerNode * numNodes,
    numNodes,
    gpusPerNode,
    intraNodeEfficiency,
    interNodeDecodeEfficiency,
    interNodePrefillEfficiency,
    bubbleEfficiency,
    scalingEfficiency: intraNodeEfficiency * interNodeDecodeEfficiency * bubbleEfficiency,
    prefillScalingEfficiency: intraNodeEfficiency * interNodePrefillEfficiency * bubbleEfficiency,
    singleGPUBaseline: singleGPU.total,
  }
}
