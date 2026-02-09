import type Decimal from 'decimal.js'
import { z } from 'zod'

/**
 * All supported quantization formats for model weights
 *
 * Float formats: fp32, fp16, bf16
 * NVIDIA FP formats: nvfp4 (E2M1), nvfp6 (E3M2) — Blackwell/Hopper
 * Integer formats: int8, int4, nf4
 * Compressed formats with overhead: gptq, awq (4-bit with 1.2x overhead)
 * GGUF variants: q2_k through q8_0 (empirical bits-per-parameter from llama.cpp)
 *
 * Reference: .planning/research/PITFALLS.md - Quantization overhead section
 */
export type QuantizationFormat =
  | 'fp32'
  | 'fp16'
  | 'bf16'
  | 'nvfp6'
  | 'nvfp4'
  | 'int8'
  | 'int4'
  | 'nf4'
  | 'gptq'
  | 'awq'
  | 'gguf-q8_0'
  | 'gguf-q6_k'
  | 'gguf-q5_k_s'
  | 'gguf-q5_k_m'
  | 'gguf-q5_0'
  | 'gguf-q4_k_s'
  | 'gguf-q4_k_m'
  | 'gguf-q4_0'
  | 'gguf-q3_k_l'
  | 'gguf-q3_k_m'
  | 'gguf-q3_k_s'
  | 'gguf-q2_k'

/**
 * KV cache quantization precision formats
 *
 * Supported by modern inference frameworks (vLLM, TensorRT-LLM, etc.)
 * See: NVIDIA TensorRT-LLM docs on KV cache quantization
 */
export type KVCachePrecision = 'fp16' | 'fp8' | 'int8' | 'int4'

/**
 * VRAM breakdown for inference workload
 *
 * All values in GB (gigabytes) using Decimal.js for precision
 */
export interface InferenceVRAMBreakdown {
  /** Model weight storage (post-quantization) */
  modelWeights: Decimal
  /** KV cache memory (depends on sequence length, batch size, architecture) */
  kvCache: Decimal
  /** Activation memory (forward pass intermediate tensors) */
  activations: Decimal
  /** Framework overhead (PyTorch runtime, CUDA context, etc.) */
  frameworkOverhead: Decimal
  /** Total VRAM requirement (sum of all above) */
  total: Decimal
}

/**
 * Performance estimate for inference workload
 *
 * Based on hardware specs (memory bandwidth, compute throughput) and model size
 */
export interface PerformanceEstimate {
  /** Throughput during decoding phase (tokens/sec) */
  tokensPerSecond: Decimal
  /** Prefill latency (time to process prompt) in milliseconds */
  timeToFirstToken: Decimal
  /** True if performance limited by TFLOPS (small batch, short sequence) */
  isComputeBound: boolean
  /** True if performance limited by memory bandwidth (large batch, long sequence) */
  isMemoryBound: boolean
  /** Primary bottleneck classification */
  bottleneck: 'compute' | 'memory' | 'balanced'
}

/**
 * Input parameters for VRAM calculation
 *
 * Validates ranges from research (INFER-06):
 * - Sequence length: 512 to 131072 (128K context)
 * - Batch size: 1 to 64 (typical inference range)
 */
export const CalculationInputSchema = z.object({
  /** Maximum sequence length (prompt + generation) */
  sequenceLength: z.number().int().min(512).max(131072),
  /** Number of concurrent sequences */
  batchSize: z.number().int().min(1).max(64),
  /** Model weight quantization format */
  quantization: z.enum([
    'fp32',
    'fp16',
    'bf16',
    'nvfp6',
    'nvfp4',
    'int8',
    'int4',
    'nf4',
    'gptq',
    'awq',
    'gguf-q8_0',
    'gguf-q6_k',
    'gguf-q5_k_s',
    'gguf-q5_k_m',
    'gguf-q5_0',
    'gguf-q4_k_s',
    'gguf-q4_k_m',
    'gguf-q4_0',
    'gguf-q3_k_l',
    'gguf-q3_k_m',
    'gguf-q3_k_s',
    'gguf-q2_k',
  ]),
  /** KV cache quantization (defaults to fp16) */
  kvQuantization: z.enum(['fp16', 'fp8', 'int8', 'int4']).default('fp16'),
})

/**
 * Type-safe calculation input (inferred from Zod schema)
 */
export type CalculationInput = z.infer<typeof CalculationInputSchema>

/**
 * Multi-GPU sharding strategies
 *
 * - tensor-parallel: Shard model weights, KV cache, and activations across GPUs
 *   (requires high-bandwidth interconnect like NVLink)
 * - pipeline-parallel: Distribute layers across GPUs, each GPU processes different
 *   stages of the pipeline (lower bandwidth requirements, higher latency)
 */
export type ShardingStrategy = 'tensor-parallel' | 'pipeline-parallel'

/**
 * GPU interconnect types with different bandwidth characteristics
 *
 * - nvlink-4: 4th gen NVLink (900 GB/s bidirectional)
 * - nvlink-5: 5th gen NVLink (1800 GB/s bidirectional)
 * - pcie-4: PCIe 4.0 x16 (64 GB/s bidirectional)
 * - pcie-5: PCIe 5.0 x16 (128 GB/s bidirectional)
 * - none: No multi-GPU support (single GPU only)
 */
export type InterconnectType = 'nvlink-4' | 'nvlink-5' | 'pcie-4' | 'pcie-5' | 'none'

/**
 * Interconnect specification with bandwidth and recommended limits
 */
export interface InterconnectSpec {
  type: InterconnectType
  bandwidthGBps: number
  recommendedMaxTPDegree: number
}

/**
 * Multi-GPU VRAM breakdown showing per-GPU memory distribution
 *
 * All values in GB (gigabytes) using Decimal.js for precision
 */
export interface MultiGPUVRAMBreakdown {
  /** Number of GPUs in the configuration */
  numGPUs: number
  /** Sharding strategy used */
  strategy: ShardingStrategy
  /** Per-GPU memory breakdown */
  perGPU: {
    /** Model weights allocated to this GPU */
    modelWeights: Decimal
    /** KV cache allocated to this GPU */
    kvCache: Decimal
    /** Activation memory for this GPU */
    activations: Decimal
    /** Framework overhead for this GPU */
    frameworkOverhead: Decimal
    /** Communication overhead (NCCL buffers, gradient sync) */
    communicationOverhead: Decimal
    /** Total memory required per GPU */
    total: Decimal
  }
  /** Memory replicated across all GPUs (embeddings, layer norms for TP) */
  replicatedMemory: Decimal
  /** Total VRAM required per GPU (includes all components) */
  totalPerGPU: Decimal
  /** GPU utilization percentage (totalPerGPU / gpuVramGB * 100) */
  utilizationPercent: Decimal
  /** Single-GPU baseline for comparison */
  singleGPUBaseline: Decimal
}

/**
 * Interconnect validation result
 */
export interface InterconnectValidation {
  /** True if configuration is valid */
  valid: boolean
  /** Warning message if configuration is suboptimal (null if no warning) */
  warning: string | null
  /** Resolved interconnect specification */
  interconnect: InterconnectSpec
}

/**
 * Offload target types
 *
 * - cpu-ram: Offload to system memory via PCIe (2-5x slower for small amounts, 15-50x for large)
 * - nvme: Offload to NVMe SSD storage (10-100x slower, for extreme memory pressure)
 */
export type OffloadTarget = 'cpu-ram' | 'nvme'

/**
 * Offload mode determines how to calculate offload amount
 *
 * - percentage: Offload a percentage of model weights (0-100%)
 * - layers: Offload a specific number of layers
 */
export type OffloadMode = 'percentage' | 'layers'

/**
 * Configuration for CPU/RAM or NVMe offloading
 *
 * Enables reducing GPU VRAM usage by offloading model weights and/or KV cache
 * to system memory or NVMe storage at the cost of inference speed.
 */
export interface OffloadingConfig {
  /** Whether offloading is enabled */
  enabled: boolean
  /** Target for offloaded memory */
  target: OffloadTarget
  /** Mode for calculating offload amount */
  mode: OffloadMode
  /** 0-100 percentage of model weights to offload */
  offloadPercentage: number
  /** Number of layers to offload (alternative to percentage) */
  offloadLayers: number
  /** Whether to offload KV cache to CPU/RAM */
  kvCacheOffload: boolean
}

/**
 * VRAM breakdown after applying offloading
 *
 * Shows memory remaining on GPU vs offloaded to CPU/RAM or NVMe,
 * with performance impact estimate.
 */
export interface OffloadedVRAMBreakdown {
  /** VRAM remaining on GPU after offloading */
  onDevice: InferenceVRAMBreakdown
  /** Memory offloaded to CPU/RAM or NVMe */
  offloaded: {
    /** Model weights offloaded */
    modelWeights: Decimal
    /** KV cache offloaded */
    kvCache: Decimal
    /** Total offloaded memory */
    total: Decimal
  }
  /** Estimated performance impact description */
  performanceImpact: string
  /** Estimated slowdown multiplier (e.g. 2.0 = 2x slower) */
  slowdownFactor: number
}
