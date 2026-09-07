import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { BYTES_PER_GB, PREFILL_MFU } from './constants'
import { calculateMoEActiveParams, calculateMoEBatchedParams } from './inference'
import { calculateModelWeightVRAM } from './quantization'
import type { MultiGPUVRAMBreakdown, PerformanceEstimate, QuantizationFormat } from './types'

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
  /** Prompt length in tokens — drives prefill time and therefore TTFT */
  sequenceLength: number
  /** Optional multi-GPU result; when provided, tokens/sec is scaled by numGPUs × scalingEfficiency */
  multiGPUResult?: MultiGPUVRAMBreakdown | null
}

/**
 * Estimate inference performance using roofline model
 *
 * The roofline model determines whether performance is limited by:
 * - **Memory bandwidth** (typical for LLM inference): Each decode token requires reading
 *   all model weights from memory once. Throughput = bandwidth / model_size.
 * - **Compute throughput** (rare, small models on fast GPUs): Forward pass requires
 *   ~2 FLOPs per parameter. Throughput = FLOPS / (2 * params).
 *
 * Performance is the minimum of these two bounds (hence "roofline").
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
 * // perf.tokensPerSecond ≈ 23.9 (memory-bound)
 * // perf.bottleneck === 'memory'
 * // perf.prefillBottleneck === 'linear'
 * ```
 */
export function estimatePerformance(params: PerformanceParams): PerformanceEstimate {
  const { model, gpu, quantization, sequenceLength, batchSize, multiGPUResult } = params

  // 1. Bytes read per decode step. MoE decode touches only the experts the step's tokens
  //    route to, so this uses active params — but it must still route through the
  //    quantization helper, because bytes depend on precision. Never hardcode `x 2` here.
  //    At batch > 1 the sequences route independently and the union of touched experts
  //    grows, so the memory side uses the batched figure rather than the batch-1 one.
  const activeParams = calculateMoEActiveParams(model)
  const decodeParams = calculateMoEBatchedParams(model, batchSize)
  const modelSizeGB = calculateModelWeightVRAM(decodeParams, quantization)
  const modelSizeBytes = modelSizeGB.mul(BYTES_PER_GB)

  // 2. Memory-bound tokens/sec (dominant for LLM inference)
  const bandwidthBytesPerSec = new Decimal(gpu.memory_bandwidth_gbps).mul(1e9)
  const memoryBoundTPS = bandwidthBytesPerSec.div(modelSizeBytes).mul(batchSize)

  // 3. Compute-bound tokens/sec. FLOPs are precision-independent: ~2 FLOPs per
  //    active parameter (one multiply, one add). This stays on the batch-1 figure:
  //    each token computes only its own experts, however many others the step has
  //    resident. Only the memory term above widens with batch.
  const flopsPerToken = new Decimal(activeParams).mul(2e9)

  let computeBoundTPS: Decimal

  // Handle missing/non-positive FLOPS: a GPU with no usable FLOPS figure (undefined,
  // zero, or negative — e.g. a user-entered custom-FLOPS value of 0) can never be
  // compute-bound; treat it the same as "no FLOPS data" rather than dividing by zero.
  const decodeGpuTFLOPS = gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0
  if (decodeGpuTFLOPS > 0) {
    // Prefer FP16 FLOPS (more relevant for inference), fallback to FP32
    const gpuFLOPS = new Decimal(decodeGpuTFLOPS).mul(1e12)
    computeBoundTPS = gpuFLOPS.div(flopsPerToken).mul(batchSize)
  } else {
    // No usable FLOPS data: set to Infinity (memory-bound only)
    computeBoundTPS = new Decimal(Infinity)
  }

  // 4. Roofline decision: performance is min of memory-bound and compute-bound
  let tokensPerSecond = Decimal.min(memoryBoundTPS, computeBoundTPS)

  // 4b. Apply multi-GPU scaling: effective TPS = single-GPU TPS × numGPUs × scalingEfficiency
  if (multiGPUResult && multiGPUResult.numGPUs > 1) {
    tokensPerSecond = tokensPerSecond
      .mul(multiGPUResult.numGPUs)
      .mul(multiGPUResult.scalingEfficiency)
  }

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
        .mul(multiGPUResult.scalingEfficiency)
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
  }
}
