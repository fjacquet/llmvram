import type { GPU, Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { BYTES_PER_GB } from './constants'
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
 * **TTFT (Time To First Token)** is estimated as 2x slower than decode because prefill
 * processes the entire prompt at once with quadratic attention, while decode is linear.
 *
 * @param params - Model, GPU, quantization, and batch size
 * @returns Performance estimate with tokens/sec, TTFT, and bottleneck analysis
 *
 * @example
 * ```ts
 * // LLaMA 3 70B FP16 on H100 80GB SXM
 * const perf = estimatePerformance({
 *   model: llama3_70b,
 *   gpu: h100_80gb_sxm,
 *   quantization: 'fp16',
 *   batchSize: 1
 * })
 * // perf.tokensPerSecond ≈ 23.9 (memory-bound)
 * // perf.bottleneck === 'memory'
 * // perf.timeToFirstToken ≈ 0.083 seconds
 * ```
 */
export function estimatePerformance(params: PerformanceParams): PerformanceEstimate {
  const { model, gpu, quantization, batchSize, multiGPUResult } = params

  // 1. Calculate model size in bytes (for memory-bound estimate)
  const modelSizeGB = calculateModelWeightVRAM(model.num_parameters_billion, quantization)
  const modelSizeBytes = modelSizeGB.mul(BYTES_PER_GB)

  // 2. Memory-bound tokens/sec (dominant for LLM inference)
  // Each decode token requires reading all model weights once
  // Throughput = (bandwidth in bytes/sec) / (model size in bytes) * batch
  const bandwidthBytesPerSec = new Decimal(gpu.memory_bandwidth_gbps).mul(1e9)
  const memoryBoundTPS = bandwidthBytesPerSec.div(modelSizeBytes).mul(batchSize)

  // 3. Compute-bound tokens/sec
  // Forward pass requires ~2 FLOPs per parameter (1 multiply + 1 add)
  // Throughput = (GPU FLOPS) / (FLOPs per token) * batch
  const flopsPerToken = new Decimal(model.num_parameters_billion).mul(2e9)

  let computeBoundTPS: Decimal

  // Handle missing FLOPS: If both fp16 and fp32 are undefined, never compute-bound
  if (gpu.fp16_tflops !== undefined || gpu.fp32_tflops !== undefined) {
    // Prefer FP16 FLOPS (more relevant for inference), fallback to FP32
    const gpuTFLOPS = gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0
    const gpuFLOPS = new Decimal(gpuTFLOPS).mul(1e12)
    computeBoundTPS = gpuFLOPS.div(flopsPerToken).mul(batchSize)
  } else {
    // No FLOPS data: set to Infinity (memory-bound only)
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

  // 6. TTFT estimation (prefill is 2x slower than decode)
  // TTFT = 1 / (decode_speed * 0.5)
  const timeToFirstToken = new Decimal(1).div(tokensPerSecond.mul(0.5))

  return {
    tokensPerSecond,
    timeToFirstToken,
    isMemoryBound,
    isComputeBound,
    bottleneck,
  }
}
