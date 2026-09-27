import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { BYTES_PER_GB, INTERCONNECT_SPECS, PREFILL_MFU } from './constants'
import { calculateMoEActiveParams, calculateMoEBatchedParams, splitMoEParams } from './inference'
import { calculateKVCacheVRAM, calculateLinearStateVRAM } from './kv-cache'
import { kvCacheTPShards, resolveInterconnect } from './multi-gpu'
import { calculateModelWeightVRAM } from './quantization'
import type {
  KVCachePrecision,
  MultiGPUVRAMBreakdown,
  PerformanceEstimate,
  QuantizationFormat,
} from './types'

/**
 * Decode-time offloading: a fraction of the model weights and/or the whole KV
 * cache live off-GPU and are read over `linkGBps` every decode step, instead
 * of once from HBM. See `hostLinkGBps` (engines/offloading.ts) for how
 * `linkGBps` is resolved from the offload target and GPU.
 */
export interface OffloadDecodeParams {
  /** Fraction (0-1) of model weights offloaded off-GPU */
  weightFraction: number
  /** Whether the entire KV cache (and linear/SSM state) is offloaded */
  kvOffloaded: boolean
  /** Per-GPU read bandwidth of the host link, decimal GB/s */
  linkGBps: number
}

/**
 * Performance estimation parameters
 */
export interface PerformanceParams {
  /** Model to estimate performance for */
  model: Model
  /** GPU hardware specs */
  gpu: GPU
  /** Model weight quantization format */
  quantization: QuantizationFormat
  /** Number of concurrent sequences (batch size) */
  batchSize: number
  /** Context length in tokens — drives prefill time, and the KV each decode step reads */
  sequenceLength: number
  /** KV cache precision (defaults to fp16) — sets the bytes each decode step reads */
  kvQuantization?: KVCachePrecision
  /** Optional multi-GPU result; sets each GPU's share of the bytes and FLOPs per step */
  multiGPUResult?: MultiGPUVRAMBreakdown | null
  /** CPU/RAM or NVMe offloading; null/undefined when offloading is disabled */
  offload?: OffloadDecodeParams | null
}

/**
 * Expert-parallel all-to-all for one MoE layer: dispatch each token's hidden state to
 * its top-k experts in FP8 (1 byte), combine the results back in BF16 (2 bytes), each
 * paying one small-message latency. Bandwidth-bound at real batch sizes: DeepEP's EP8
 * table (128 tokens, 7168 hidden, top-8) moves 22 MB in 77 + 114 us.
 *
 * @param uniGBps - one-direction link bandwidth per GPU in GB/s
 */
export function expertAllToAllSeconds(
  tokensPerGPU: number,
  expertsPerToken: number,
  hidden: number,
  uniGBps: number,
  latencyUs: number,
): number {
  const bytes = tokensPerGPU * expertsPerToken * hidden * (1 + 2)
  return (2 * latencyUs) / 1e6 + bytes / (uniGBps * 1e9)
}

/**
 * How a multi-GPU layout divides one decode step.
 *
 * Pipeline stages run one after another; the GPUs inside a stage split its layers
 * by tensor parallelism or, for MoE, by expert parallelism. Nodes are always
 * pipeline stages (see multi-node.ts), and intra-node pipeline parallelism adds
 * gpusPerNode stages per node.
 */
function decodeLayout(model: Model, multi: MultiGPUVRAMBreakdown | null | undefined) {
  if (!multi || multi.numGPUs <= 1) {
    return { strategy: null, stages: 1, gpusPerStage: 1, kvShards: 1, interNodeEfficiency: 1 }
  }
  const intraPP = multi.strategy === 'pipeline-parallel'
  const gpusPerStage = intraPP ? 1 : multi.gpusPerNode
  let kvShards = 1
  if (multi.strategy === 'expert-parallel') kvShards = gpusPerStage
  else if (gpusPerStage > 1) kvShards = kvCacheTPShards(model, gpusPerStage)
  return {
    strategy: gpusPerStage > 1 ? multi.strategy : null,
    stages: multi.numNodes * (intraPP ? multi.gpusPerNode : 1),
    gpusPerStage,
    kvShards,
    interNodeEfficiency: multi.interNodeDecodeEfficiency,
  }
}

/**
 * Estimate inference performance using roofline model
 *
 * One decode step produces one token for each of batchSize sequences. Its time is
 * the slower of two bounds (hence "roofline"), plus communication:
 * - **Memory bandwidth** (typical for LLM inference): the step reads the weights once
 *   and every sequence's KV cache. Time = (weights + batch * KV) / bandwidth.
 * - **Compute throughput** (large batches, small models): ~2 FLOPs per active
 *   parameter plus attention over the context, per token. Time = batch * FLOPs / FLOPS.
 * - **Multi-GPU**: each GPU reads and computes only its share; tensor parallelism
 *   adds two all-reduce latencies per layer, and pipeline stages overlap only as far
 *   as the batch fills them.
 *
 * Aggregate tokens/sec = batchSize / step time.
 *
 * **TTFT (Time To First Token)** is modeled as a compute-bound prefill pass over the
 * prompt, plus one decode step. Prefill FLOPs have two terms: a linear
 * `2 * activeParams * T` term (same per-token cost as decode, precision-independent)
 * and a quadratic causal-attention term `2 * layers * T^2 * hidden`. Dividing by
 * `gpuFLOPS * PREFILL_MFU` gives prefill seconds; when the GPU exposes no FLOPS data,
 * TTFT falls back to the previous heuristic and the estimate is marked degraded.
 *
 * @param params - Model, GPU, quantization, batch size, and prompt sequence length
 * @returns Performance estimate with tokens/sec, TTFT, prefill breakdown, and bottleneck analysis
 *
 * @example
 * ```ts
 * // LLaMA 3 70B FP16 on H100 80GB SXM
 * const perf = estimatePerformance({
 *   model: llama3_70b,
 *   gpu: h100_80gb_sxm,
 *   quantization: 'fp16',
 *   batchSize: 1,
 *   sequenceLength: 2048,
 * })
 * // perf.tokensPerSecond ≈ 23.8 (memory-bound: 140 GB weights + 0.7 GB KV per step)
 * // perf.bottleneck === 'memory'
 * // perf.prefillBottleneck === 'linear'
 * ```
 */
export function estimatePerformance(params: PerformanceParams): PerformanceEstimate {
  const {
    model,
    gpu,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization = 'fp16',
    multiGPUResult,
  } = params

  // 1. Bytes read per decode step: the weights once, plus every sequence's KV cache.
  //    MoE decode touches only the experts the step's tokens route to, so the weights
  //    use active params — but they must still route through the quantization helper,
  //    because bytes depend on precision. Never hardcode `x 2` here. At batch > 1 the
  //    sequences route independently and the union of touched experts grows, so the
  //    memory side uses the batched figure rather than the batch-1 one.
  //    KV is read at the full context (sequenceLength): the end of a generation, the
  //    conservative point.
  const activeParams = calculateMoEActiveParams(model)
  const decodeParams = calculateMoEBatchedParams(model, batchSize)
  const weightBytes = calculateModelWeightVRAM(decodeParams, quantization, model).mul(BYTES_PER_GB)
  const kvBytes = calculateKVCacheVRAM({
    model,
    sequenceLength,
    batchSize,
    kvPrecision: kvQuantization,
  }).mul(BYTES_PER_GB)

  // 2. Per-GPU share of one step. Each pipeline stage holds 1/stages of the layers.
  //    Inside a stage, tensor parallelism splits the weights gpusPerStage ways and the
  //    KV only kvShards ways (MLA duplicates it, GQA stops at one head per GPU).
  //    Expert parallelism reads the replicated base in full and 1/N of the routed
  //    experts the batch touches; each GPU reads only its own sessions' KV.
  const layout = decodeLayout(model, multiGPUResult)
  const split = splitMoEParams(model)
  const perGPUWeightBytes =
    layout.strategy === 'expert-parallel' && split
      ? calculateModelWeightVRAM(
          split.baseB + Math.max(0, decodeParams - split.baseB) / layout.gpusPerStage,
          quantization,
          model,
        ).mul(BYTES_PER_GB)
      : weightBytes.div(layout.gpusPerStage)
  // Linear-attention state is part of kvBytes but splits across every GPU of the
  // stage (vLLM divides its heads by tp_world_size), even where MLA KV is duplicated.
  const stateBytes = calculateLinearStateVRAM(model, batchSize).mul(BYTES_PER_GB)
  const perGPUKVBytes = kvBytes
    .sub(stateBytes)
    .div(layout.kvShards)
    .add(stateBytes.div(layout.gpusPerStage))

  // Offloading: a fraction of the weights and/or the whole KV cache live off-GPU
  // and are read over the host link every step instead of from HBM (hostLinkGBps
  // resolves linkGBps from the target and GPU). Only the on-GPU share still
  // counts as an HBM read, so it's subtracted here rather than read twice.
  //
  // The offloaded share is taken out of perGPUWeightBytes/perGPUKVBytes — the
  // SAME per-GPU figures the on-device HBM math below uses — not out of the
  // global weightBytes/kvBytes divided by gpusPerStage. Those two disagree
  // whenever a GPU's actual share isn't a plain 1/gpusPerStage split: MLA
  // duplicates the full KV on every TP rank (kvShards can be 1 while
  // gpusPerStage is 8, so dividing by gpusPerStage undercounts the per-GPU
  // link read by kvShards/gpusPerStage), and expert parallelism replicates
  // the base weights across every GPU rather than sharding them (dividing by
  // gpusPerStage there undercounts the base's contribution too).
  const offload = params.offload ?? null
  const offloadedPerGPUWeightBytes = offload
    ? perGPUWeightBytes.mul(offload.weightFraction)
    : new Decimal(0)
  const offloadedPerGPUKVBytes = offload?.kvOffloaded ? perGPUKVBytes : new Decimal(0)
  const onDevicePerGPUWeightBytes = perGPUWeightBytes.sub(offloadedPerGPUWeightBytes)
  const onDevicePerGPUKVBytes = perGPUKVBytes.sub(offloadedPerGPUKVBytes)
  // Serial with the HBM read (conservative; matches vLLM cpu_offload_gb streaming).
  const offloadSecondsPerStep = offload
    ? offloadedPerGPUWeightBytes
        .add(offloadedPerGPUKVBytes)
        .div(layout.stages)
        .div(new Decimal(offload.linkGBps).mul(1e9))
    : new Decimal(0)

  const perGPUBytes = onDevicePerGPUWeightBytes.add(onDevicePerGPUKVBytes).div(layout.stages)
  const bandwidthBytesPerSec = new Decimal(gpu.memory_bandwidth_gbps).mul(1e9)
  const memorySeconds = perGPUBytes.div(bandwidthBytesPerSec).add(offloadSecondsPerStep)

  // 3. Compute per step. FLOPs are precision-independent: ~2 per active parameter
  //    (batch-1 figure — each token computes only its own experts) plus causal
  //    attention over the context, 4 * layers * context * hidden per token (the
  //    per-token slope of the prefill attention term below). The machine's FLOPS
  //    are shared by the batch: this is an aggregate ceiling, never multiplied by it.
  const flopsPerToken = new Decimal(activeParams)
    .mul(2e9)
    .add(new Decimal(4).mul(model.num_hidden_layers).mul(sequenceLength).mul(model.hidden_size))

  // Handle missing/non-positive FLOPS: a GPU with no usable FLOPS figure (undefined,
  // zero, or negative — e.g. a user-entered custom-FLOPS value of 0) can never be
  // compute-bound; treat it the same as "no FLOPS data" rather than dividing by zero.
  // Prefer FP16 FLOPS (more relevant for inference), fallback to FP32.
  const decodeGpuTFLOPS = gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0
  const computeSeconds =
    decodeGpuTFLOPS > 0
      ? flopsPerToken
          .mul(batchSize)
          .div(layout.gpusPerStage * layout.stages)
          .div(new Decimal(decodeGpuTFLOPS).mul(1e12))
      : new Decimal(0)

  // 4. Roofline per stage, plus communication per layer: two all-reduces for tensor
  //    parallelism (after attention and after the MLP), one dispatch + combine
  //    all-to-all for expert parallelism. A step flows through the stages in
  //    turn; with batchSize sequences in flight the pipeline overlaps them, less
  //    the bubble B / (B + stages - 1). A decode token cannot be split into
  //    micro-batches (unlike a prompt, see fabric.ts pipelineBubbleEfficiency), so
  //    at batch 1 pipeline parallelism gives no decode speedup.
  const link = INTERCONNECT_SPECS[resolveInterconnect(gpu)]
  const layersPerStage = model.num_hidden_layers / layout.stages
  let commSecondsPerLayer = 0
  if (layout.strategy === 'tensor-parallel') {
    commSecondsPerLayer = (2 * link.allreduceLatencyUs) / 1e6
  } else if (layout.strategy === 'expert-parallel' && split) {
    // bandwidthGBps is bidirectional per GPU; one direction carries each transfer
    commSecondsPerLayer = expertAllToAllSeconds(
      batchSize / layout.gpusPerStage,
      split.expertsPerToken,
      model.hidden_size,
      link.bandwidthGBps / 2,
      link.allreduceLatencyUs,
    )
  }
  const stageSeconds = Decimal.max(memorySeconds, computeSeconds).add(
    commSecondsPerLayer * layersPerStage,
  )
  const tokensPerSecond = new Decimal(batchSize)
    .div(stageSeconds)
    .mul(new Decimal(batchSize).div(batchSize + layout.stages - 1))
    .mul(layout.interNodeEfficiency)

  // Offload slowdown: this step's time versus the same step with nothing
  // offloaded (full weights + KV in HBM, no host-link read). Compute and
  // communication are unaffected by offloading, so this isolates exactly the
  // cost the host link adds — computed here so callers never call this
  // function twice to get it.
  let offloadSlowdown: number | null = null
  if (offload) {
    const baselinePerGPUBytes = perGPUWeightBytes.add(perGPUKVBytes).div(layout.stages)
    const baselineMemorySeconds = baselinePerGPUBytes.div(bandwidthBytesPerSec)
    const baselineStageSeconds = Decimal.max(baselineMemorySeconds, computeSeconds).add(
      commSecondsPerLayer * layersPerStage,
    )
    offloadSlowdown = stageSeconds.div(baselineStageSeconds).toNumber()
  }

  const memoryBoundTPS = new Decimal(batchSize).div(memorySeconds)
  const computeBoundTPS =
    decodeGpuTFLOPS > 0 ? new Decimal(batchSize).div(computeSeconds) : new Decimal(Infinity)

  // 5. Bottleneck analysis (5% tolerance to avoid flip-flopping at boundary)
  const tolerance = 0.95
  let bottleneck: 'memory' | 'compute' | 'balanced'
  let isMemoryBound: boolean
  let isComputeBound: boolean

  if (memoryBoundTPS.lessThan(computeBoundTPS.mul(tolerance))) {
    bottleneck = 'memory'
    isMemoryBound = true
    isComputeBound = false
  } else if (computeBoundTPS.lessThan(memoryBoundTPS.mul(tolerance))) {
    bottleneck = 'compute'
    isMemoryBound = false
    isComputeBound = true
  } else {
    bottleneck = 'balanced'
    isMemoryBound = true
    isComputeBound = true
  }

  // 6. Prefill model. Prefill is compute-bound — a different roofline regime from the
  //    bandwidth-bound decode above. Two terms:
  //      linear:    2 * activeParams * T          (precision-independent FLOPs)
  //      attention: 2 * layers * T^2 * hidden      (1/2 * 4 * T^2 * D * L, causal)
  //    Batch is NOT applied: TTFT is a per-request latency for one sequence of T tokens.
  const promptTokens = new Decimal(sequenceLength)
  const linearFLOPs = new Decimal(activeParams).mul(2e9).mul(promptTokens)
  const attentionFLOPs = new Decimal(2)
    .mul(model.num_hidden_layers)
    .mul(promptTokens.pow(2))
    .mul(model.hidden_size)

  const prefillBottleneck: 'linear' | 'attention' = attentionFLOPs.greaterThan(linearFLOPs)
    ? 'attention'
    : 'linear'

  let prefillSeconds: Decimal | null = null
  let prefillEstimateDegraded = false
  let timeToFirstToken: Decimal

  // Same FLOPS figure and same guard as the decode roofline above — one source, so a
  // change to the selection policy cannot desynchronise the two rooflines.
  if (decodeGpuTFLOPS > 0) {
    let effectiveFLOPS = new Decimal(decodeGpuTFLOPS).mul(1e12).mul(PREFILL_MFU)

    if (multiGPUResult && multiGPUResult.numGPUs > 1) {
      effectiveFLOPS = effectiveFLOPS
        .mul(multiGPUResult.numGPUs)
        .mul(multiGPUResult.prefillScalingEfficiency)
    }

    prefillSeconds = linearFLOPs.add(attentionFLOPs).div(effectiveFLOPS)
    timeToFirstToken = prefillSeconds.add(new Decimal(1).div(tokensPerSecond))
  } else {
    // No usable FLOPS data (missing, zero, or negative): prefill time is not
    // computable. Fall back to the previous heuristic rather than returning
    // Infinity or NaN, and mark the estimate degraded.
    prefillEstimateDegraded = true
    timeToFirstToken = new Decimal(1).div(tokensPerSecond.mul(0.5))
  }

  return {
    tokensPerSecond,
    timeToFirstToken,
    prefillSeconds,
    prefillBottleneck,
    prefillEstimateDegraded,
    isMemoryBound,
    isComputeBound,
    bottleneck,
    offloadSlowdown,
  }
}
