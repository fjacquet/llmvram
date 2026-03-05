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
  /** Fraction of linear scaling achieved in tensor parallelism (0–1). Accounts for all-reduce overhead. */
  tpScalingEfficiency: number
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
  /** TP scaling efficiency for this interconnect (0–1 fraction, e.g. 0.92 for NVLink-4) */
  scalingEfficiency: number
  /** Interconnect bandwidth in GB/s (0 for single GPU) */
  interconnectBandwidthGBps: number
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

/**
 * Optimizer types for fine-tuning
 *
 * - adamw: AdamW optimizer (standard, 2 FP32 states = 8 bytes/param)
 * - sgd-momentum: SGD with momentum (1 FP32 state = 4 bytes/param)
 * - adamw-8bit: 8-bit quantized AdamW (2 8-bit states = 2 bytes/param)
 * - adafactor: Memory-efficient factored approximation (4 bytes/param)
 *
 * CRITICAL: Optimizer states are ALWAYS stored in FP32 for numerical stability,
 * even during mixed precision training. See PITFALLS.md #2.
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export type OptimizerType = 'adamw' | 'sgd-momentum' | 'adamw-8bit' | 'adafactor'

/**
 * Fine-tuning methods with different VRAM profiles
 *
 * - full: Full fine-tuning (all parameters trainable)
 * - lora: Low-Rank Adaptation (adapters only, base frozen)
 * - qlora: Quantized LoRA (4-bit base + FP16 adapters)
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export type FineTuningMethod = 'full' | 'lora' | 'qlora'

/**
 * Training precision for weights and gradients
 *
 * - fp32: Full precision training (4 bytes/param)
 * - fp16: Half precision training (2 bytes/param, requires FP32 master weights)
 * - bf16: BFloat16 training (2 bytes/param, requires FP32 master weights)
 *
 * Note: Mixed precision (fp16/bf16) stores training weights in FP16/BF16 but
 * maintains FP32 master weights for numerical stability.
 */
export type TrainingPrecision = 'fp32' | 'fp16' | 'bf16'

/**
 * VRAM breakdown for training workload
 *
 * All values in GB (gigabytes) using Decimal.js for precision.
 *
 * Components follow the training memory formula:
 * Total = ModelWeights + MasterWeights + Gradients + OptimizerStates + Activations + Overhead
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export interface TrainingVRAMBreakdown {
  /** Model weight storage at training precision (or quantized for LoRA/QLoRA frozen base) */
  modelWeights: Decimal
  /**
   * FP32 master copy for mixed precision training
   *
   * Zero if training in fp32. Required for fp16/bf16 training to maintain
   * numerical stability during weight updates.
   */
  masterWeights: Decimal
  /**
   * Gradient storage (same precision as training weights)
   *
   * Only allocated for trainable parameters. For LoRA, this is only
   * for adapter parameters (~1% of full model).
   */
  gradients: Decimal
  /**
   * Optimizer state storage
   *
   * CRITICAL: ALWAYS stored in FP32, even during mixed precision training.
   * Size depends on optimizer type:
   * - AdamW: 8 bytes/param (2 FP32 states: momentum + variance)
   * - SGD-momentum: 4 bytes/param (1 FP32 state)
   * - AdamW-8bit: 2 bytes/param (quantized)
   * - Adafactor: 4 bytes/param (factored approximation)
   *
   * For LoRA/QLoRA, only applies to adapter parameters, NOT frozen base.
   * See PITFALLS.md #2 for why this is the most common estimation error.
   */
  optimizerStates: Decimal
  /** Training activation memory (batch-dependent, proportional to batch_size * seq_len * hidden_size) */
  activations: Decimal
  /** Framework overhead (PyTorch + CUDA context + autograd graph) */
  frameworkOverhead: Decimal
  /** Total VRAM requirement (sum of all components) */
  total: Decimal
  /** Count of trainable parameters in billions */
  trainableParameters: Decimal
  /** Count of total parameters in billions */
  totalParameters: Decimal
  /** Fine-tuning method used */
  method: FineTuningMethod
}

/**
 * VRAM breakdown for LoRA/QLoRA training
 *
 * Extends TrainingVRAMBreakdown with LoRA-specific fields showing
 * the split between frozen base model and trainable adapters.
 */
export interface LoRAVRAMBreakdown extends TrainingVRAMBreakdown {
  /** Frozen base model weights (INT4 for QLoRA, original precision for LoRA) */
  baseWeights: Decimal
  /** LoRA adapter weights in FP16 (q_proj, k_proj, v_proj, etc.) */
  adapterWeights: Decimal
  /** Adapter parameter count in billions */
  adapterParameters: Decimal
}

/**
 * Training memory optimization configuration
 *
 * Optional parameters for memory optimization techniques that reduce
 * activation memory during training at the cost of additional compute.
 *
 * - gradientCheckpointing: Recompute activations during backward pass (60% memory reduction)
 * - flashAttention: Fused attention kernel (15-70% reduction based on sequence length)
 *
 * Both techniques can be combined for multiplicative savings.
 *
 * Reference: .planning/phases/08-memory-optimization-features/08-RESEARCH.md
 */
export interface TrainingOptimizationConfig {
  /** Enable gradient checkpointing (activation checkpointing) */
  gradientCheckpointing?: boolean
  /** Enable Flash Attention memory-efficient attention */
  flashAttention?: boolean
}

/**
 * DeepSpeed ZeRO memory calculation result
 *
 * Shows per-GPU memory breakdown after applying ZeRO partitioning,
 * with reduction factor showing total memory savings vs single-GPU baseline.
 *
 * CRITICAL: ZeRO stages partition different components:
 * - ZeRO-1: Only optimizer states partitioned (2x reduction)
 * - ZeRO-2: Optimizer states + gradients partitioned (4x reduction)
 * - ZeRO-3: All training state partitioned (8x reduction with gather overhead)
 *
 * Reference: .planning/phases/10-framework-presets-multi-gpu-training/10-RESEARCH.md
 */
export interface ZeROResult {
  /** Per-GPU memory breakdown after ZeRO partitioning */
  perGPU: {
    /** Model weights (replicated for ZeRO-1/2, partitioned for ZeRO-3) */
    modelWeights: Decimal
    /** FP32 master weights (replicated for ZeRO-1/2, partitioned for ZeRO-3) */
    masterWeights: Decimal
    /** Gradients (replicated for ZeRO-1, partitioned for ZeRO-2/3) */
    gradients: Decimal
    /** Optimizer states (partitioned for all ZeRO stages) */
    optimizerStates: Decimal
    /** Activations (always partitioned) */
    activations: Decimal
    /** Framework overhead (15% extra for ZeRO-3 parameter gather) */
    frameworkOverhead: Decimal
    /** Total memory per GPU */
    total: Decimal
  }
  /** Memory reduction factor (singleGPU.total / perGPU.total) */
  reductionFactor: Decimal
  /** Number of GPUs in configuration */
  numGPUs: number
  /** ZeRO stage applied */
  zeroStage: import('./frameworks').ZeROStage
}
