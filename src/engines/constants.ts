import Decimal from 'decimal.js'
import type {
  InterconnectSpec,
  InterconnectType,
  KVCachePrecision,
  OptimizerType,
  QuantizationFormat,
  TrainingPrecision,
} from './types'

/**
 * Framework overhead for PyTorch + CUDA runtime
 *
 * Conservative 1GB estimate covering:
 * - PyTorch JIT compilation and runtime (200-400MB)
 * - CUDA context and driver (300-500MB)
 * - Kernel caching and workspace buffers (100-300MB)
 *
 * Reference: INFER-04 in .planning/phases/02-inference-engine/02-RESEARCH.md
 */
export const FRAMEWORK_OVERHEAD_GB = new Decimal(1.0)

/**
 * Bytes per parameter for each quantization format
 *
 * Float formats: Standard precision
 * - fp32: 4 bytes, fp16/bf16: 2 bytes
 *
 * NVIDIA FP formats (Blackwell/Hopper):
 * - nvfp6: 6-bit FP (E3M2) → 0.75 bytes/param
 * - nvfp4: 4-bit FP (E2M1) with FP8 scaling → ~4.5 bpp effective → 0.5625 bytes/param
 *
 * Integer formats: Native precision
 * - int8: 1 byte, int4/nf4: 0.5 bytes
 *
 * Compressed formats with overhead (NOT pure bits-per-weight):
 * - gptq: 0.6 bytes (4-bit + 1.2x overhead for codebooks/scales)
 * - awq: 0.6 bytes (4-bit + 1.2x overhead for activation-aware quantization)
 *
 * GGUF formats (empirical bits-per-parameter from llama.cpp block sizes):
 * Computed as: type_size_bytes * 8 / block_size
 * - q2_k:   block 256, size 84 → 2.625 bpp → 0.328 bytes
 * - q3_k_s: block 256, size 110 → 3.4375 bpp → 0.430 bytes
 * - q3_k_m: mixed quantization → ~3.9 bpp → 0.489 bytes
 * - q3_k_l: mixed quantization → ~4.13 bpp → 0.516 bytes
 * - q4_0:   block 32, size 18 → 4.5 bpp → 0.5625 bytes
 * - q4_k_s: block 256, size 144 → 4.5 bpp → 0.5625 bytes
 * - q4_k_m: mixed quantization → ~4.8 bpp → 0.6 bytes
 * - q5_0:   block 32, size 22 → 5.5 bpp → 0.6875 bytes
 * - q5_k_s: block 256, size 176 → 5.5 bpp → 0.6875 bytes
 * - q5_k_m: mixed quantization → ~5.69 bpp → 0.711 bytes
 * - q6_k:   block 256, size 210 → 6.5625 bpp → 0.820 bytes
 * - q8_0:   block 32, size 34 → 8.5 bpp → 1.0625 bytes
 *
 * Sources:
 * - GPTQ/AWQ overhead: localaimaster.com/gptq-vs-awq-quantization-methods (INFER-01)
 * - GGUF block sizes: llama.cpp ggml-common.h (INFER-02)
 * - NVIDIA FP formats: TensorRT-LLM documentation
 */
export const BYTES_PER_PARAMETER: Record<QuantizationFormat, Decimal> = {
  // Float formats
  fp32: new Decimal(4.0),
  fp16: new Decimal(2.0),
  bf16: new Decimal(2.0),

  // NVIDIA FP formats
  nvfp6: new Decimal(0.75), // 6-bit FP (E3M2)
  nvfp4: new Decimal(0.5625), // 4-bit FP (E2M1) + FP8 scales → ~4.5 bpp

  // Integer formats
  int8: new Decimal(1.0),
  int4: new Decimal(0.5),
  nf4: new Decimal(0.5),

  // Compressed formats (4-bit base + 1.2x overhead)
  gptq: new Decimal(0.6), // 0.5 * 1.2
  awq: new Decimal(0.6), // 0.5 * 1.2

  // GGUF formats (empirical bpp / 8, from llama.cpp block sizes)
  'gguf-q8_0': new Decimal(1.0625), // 8.5 bpp
  'gguf-q6_k': new Decimal(0.82), // 6.5625 bpp
  'gguf-q5_k_s': new Decimal(0.6875), // 5.5 bpp
  'gguf-q5_k_m': new Decimal(0.711), // 5.69 bpp
  'gguf-q5_0': new Decimal(0.6875), // 5.5 bpp
  'gguf-q4_k_s': new Decimal(0.5625), // 4.5 bpp
  'gguf-q4_k_m': new Decimal(0.6), // 4.8 bpp
  'gguf-q4_0': new Decimal(0.5625), // 4.5 bpp
  'gguf-q3_k_l': new Decimal(0.516), // 4.13 bpp
  'gguf-q3_k_m': new Decimal(0.489), // 3.9 bpp
  'gguf-q3_k_s': new Decimal('0.430'), // 3.44 bpp
  'gguf-q2_k': new Decimal(0.328), // 2.625 bpp
}

/**
 * Bytes per element for KV cache quantization precision
 *
 * KV cache stores key/value tensors separately from model weights.
 * Modern frameworks support per-layer KV quantization for memory efficiency.
 *
 * Formats:
 * - fp16: 2 bytes (default, lossless for most models)
 * - fp8: 1 byte (E4M3/E5M2 formats, minimal quality loss)
 * - int8: 1 byte (requires calibration)
 * - int4: 0.5 bytes (aggressive, may degrade quality)
 *
 * Reference: NVIDIA TensorRT-LLM KV cache quantization docs (INFER-05)
 */
export const KV_PRECISION_BYTES: Record<KVCachePrecision, Decimal> = {
  fp16: new Decimal(2.0),
  fp8: new Decimal(1.0),
  int8: new Decimal(1.0),
  int4: new Decimal(0.5),
}

/**
 * Bytes per gigabyte (1024^3)
 *
 * Used for converting parameter counts to GB.
 * Binary units (GiB) match GPU VRAM reporting (NVIDIA-SMI, etc.)
 */
export const BYTES_PER_GB = new Decimal(1024).pow(3)

/**
 * NCCL buffer memory per peer GPU (200MB per peer)
 *
 * NCCL (NVIDIA Collective Communications Library) allocates buffers for
 * inter-GPU communication in tensor parallelism. Each GPU needs buffers
 * for each peer GPU in the tensor parallel group.
 *
 * For N GPUs: NCCL overhead = 0.2 GB * (N-1) peers
 *
 * Reference: NVIDIA NCCL documentation on buffer allocation
 */
export const NCCL_BUFFER_PER_PEER_GB = new Decimal(0.2)

/**
 * Embedding weight fraction of total model weights (~3%)
 *
 * In tensor parallelism, embeddings must be replicated across all GPUs
 * (not sharded) for efficient lookup. This constant estimates the fraction
 * of model weights devoted to embeddings.
 *
 * Used in conjunction with layer norm memory for replicated memory calculation.
 */
export const EMBEDDING_WEIGHT_FRACTION = new Decimal(0.03)

/**
 * Tensor parallelism communication overhead (12%)
 *
 * TP requires all-reduce operations for gradient synchronization and
 * activation passing. This adds ~12% memory overhead for communication buffers.
 */
export const TP_COMMUNICATION_OVERHEAD = new Decimal(0.12)

/**
 * Pipeline parallelism communication overhead (5%)
 *
 * PP has lower communication overhead than TP since it only passes activations
 * between pipeline stages (no all-reduce). ~5% overhead for activation buffers.
 */
export const PP_COMMUNICATION_OVERHEAD = new Decimal(0.05)

/**
 * Pipeline parallelism activation stashing overhead (12%)
 *
 * PP requires stashing activations from forward pass for backward pass across
 * pipeline stages. This adds ~12% to the base activation memory.
 */
export const PP_ACTIVATION_STASHING_OVERHEAD = new Decimal(0.12)

/**
 * MoE multi-GPU communication overhead (15% extra)
 *
 * MoE models have additional expert routing communication overhead in multi-GPU
 * setups. This multiplier is applied on top of base TP/PP overhead.
 *
 * For TP with MoE: communicationOverhead = weightsPerGPU * 0.12 * 1.15
 */
export const MOE_MULTI_GPU_OVERHEAD = new Decimal(0.15)

/**
 * Interconnect specifications for different GPU interconnect types
 *
 * Bandwidth values are bidirectional (total read+write).
 * recommendedMaxTPDegree is the practical limit for tensor parallelism
 * before communication overhead dominates.
 */
export const INTERCONNECT_SPECS: Record<InterconnectType, InterconnectSpec> = {
  'nvlink-4': {
    type: 'nvlink-4',
    bandwidthGBps: 900,
    recommendedMaxTPDegree: 8,
    tpScalingEfficiency: 0.92, // 900 GB/s — excellent scaling
  },
  'nvlink-5': {
    type: 'nvlink-5',
    bandwidthGBps: 1800,
    recommendedMaxTPDegree: 8,
    tpScalingEfficiency: 0.97, // 1800 GB/s — near-linear scaling
  },
  'pcie-4': {
    type: 'pcie-4',
    bandwidthGBps: 64,
    recommendedMaxTPDegree: 2,
    tpScalingEfficiency: 0.65, // 64 GB/s — significant communication drag
  },
  'pcie-5': {
    type: 'pcie-5',
    bandwidthGBps: 128,
    recommendedMaxTPDegree: 4,
    tpScalingEfficiency: 0.78, // 128 GB/s — noticeable drag
  },
  none: {
    type: 'none',
    bandwidthGBps: 0,
    recommendedMaxTPDegree: 1,
    tpScalingEfficiency: 0.0, // no multi-GPU support
  },
}

/**
 * Optimizer state memory per trainable parameter (in bytes)
 *
 * CRITICAL: Optimizer states are ALWAYS stored in FP32 for numerical stability,
 * even during mixed precision (FP16/BF16) training. This is the most commonly
 * underestimated component in training VRAM calculations.
 *
 * Memory breakdown by optimizer:
 * - AdamW: 8 bytes/param (2 FP32 states: momentum + variance)
 * - SGD-momentum: 4 bytes/param (1 FP32 state: momentum)
 * - AdamW-8bit: 2 bytes/param (2 8-bit quantized states)
 * - Adafactor: 4 bytes/param (factored approximation, memory-efficient)
 *
 * For LoRA/QLoRA: Only applies to adapter parameters, NOT frozen base weights.
 * Example: 70B model with 1% LoRA adapters → 0.7B trainable params → 5.6GB optimizer states (AdamW)
 *
 * Reference: .planning/research/PITFALLS.md #2 (30-60% underestimation from missing this)
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export const OPTIMIZER_STATE_BYTES: Record<OptimizerType, Decimal> = {
  adamw: new Decimal(8), // 2 FP32 states (momentum + variance) = 2 * 4 bytes
  'sgd-momentum': new Decimal(4), // 1 FP32 state (momentum) = 1 * 4 bytes
  'adamw-8bit': new Decimal(2), // 2 8-bit quantized states = 2 * 1 byte
  adafactor: new Decimal(4), // Factored approximation ~4 bytes/param
}

/**
 * Training framework overhead (PyTorch + CUDA + autograd)
 *
 * Training requires more overhead than inference (1.5GB vs 1.0GB) due to:
 * - Autograd graph storage for backward pass
 * - Gradient accumulation buffers
 * - Optimizer state management structures
 * - CUDA workspace memory for training kernels
 *
 * Conservative 1.5GB estimate based on PyTorch training profiling.
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export const TRAINING_FRAMEWORK_OVERHEAD_GB = new Decimal(1.5)

/**
 * Total targetable modules per transformer layer
 *
 * Standard transformer layer has 7 linear modules that can receive LoRA adapters:
 * - Attention: q_proj, k_proj, v_proj, o_proj (4 modules)
 * - MLP: gate_proj, up_proj, down_proj (3 modules)
 *
 * Used to calculate actual adapter count from targetModulesPercent:
 * adapters_per_layer = (TOTAL_TARGETABLE_MODULES_PER_LAYER * targetModulesPercent / 100)
 *
 * Example: 30% targeting → 2.1 → round(2.1) = 2 modules per layer
 *
 * Reference: .planning/phases/06-fine-tuning-calculation-engines/06-RESEARCH.md
 */
export const TOTAL_TARGETABLE_MODULES_PER_LAYER = 7

/**
 * Gradient storage bytes per parameter by training precision
 *
 * Gradients are stored at the same precision as training weights:
 * - FP32 training → FP32 gradients (4 bytes)
 * - FP16 training → FP16 gradients (2 bytes)
 * - BF16 training → BF16 gradients (2 bytes)
 *
 * Only allocated for trainable parameters. For LoRA, gradients only
 * apply to adapter parameters (~1% of full model).
 */
export const GRADIENT_BYTES: Record<TrainingPrecision, Decimal> = {
  fp32: new Decimal(4),
  fp16: new Decimal(2),
  bf16: new Decimal(2),
}

/**
 * Training weight storage bytes per parameter by precision
 *
 * Same values as GRADIENT_BYTES, but semantically distinct for code clarity.
 * Training weights (not frozen base) are stored at training precision.
 */
export const WEIGHT_BYTES: Record<TrainingPrecision, Decimal> = {
  fp32: new Decimal(4),
  fp16: new Decimal(2),
  bf16: new Decimal(2),
}

/**
 * FP32 master weight storage for mixed precision training
 *
 * Mixed precision training (FP16/BF16) maintains an FP32 "master copy" of
 * weights for numerical stability during optimizer updates. The training
 * forward/backward passes use FP16/BF16, but weight updates use FP32.
 *
 * Master weights = 4 bytes/param for FP16/BF16 training, 0 for FP32 training.
 *
 * For LoRA: Only applies to trainable adapter parameters, not frozen base.
 */
export const MASTER_WEIGHT_BYTES = new Decimal(4)

/**
 * Gradient checkpointing activation retention factor (40%)
 *
 * Gradient checkpointing (activation checkpointing) trades compute for memory
 * by storing only a subset of activations during forward pass and recomputing
 * the rest during backward pass. This typically retains 40% of activation memory
 * (60% reduction) with ~20-30% compute overhead.
 *
 * Formula: optimized_activations = base_activations * 0.4
 *
 * Reference: .planning/phases/08-memory-optimization-features/08-RESEARCH.md
 */
export const CHECKPOINTING_RETENTION_FACTOR = new Decimal(0.4)

/**
 * Flash Attention retention factors by sequence length
 *
 * Flash Attention reduces memory by fusing attention operations and avoiding
 * materialization of large QK^T attention matrices. Memory savings scale with
 * sequence length due to quadratic attention memory O(N^2).
 *
 * Retention factors (fraction of activation memory kept):
 * - Short (<2048 tokens): 0.85 (15% reduction) - minimal benefit
 * - Medium (2048-8191 tokens): 0.5 (50% reduction) - moderate benefit
 * - Long (>=8192 tokens): 0.3 (70% reduction) - large benefit
 *
 * Formula: optimized_activations = base_activations * retention_factor
 *
 * Reference: .planning/phases/08-memory-optimization-features/08-RESEARCH.md
 */
export const FLASH_ATTENTION_SHORT_RETENTION = new Decimal(0.85)
export const FLASH_ATTENTION_MEDIUM_RETENTION = new Decimal(0.5)
export const FLASH_ATTENTION_LONG_RETENTION = new Decimal(0.3)

/**
 * Flash Attention sequence length thresholds
 *
 * Thresholds for determining Flash Attention retention factor:
 * - < 2048: Short sequence (minimal benefit)
 * - 2048-8191: Medium sequence (moderate benefit)
 * - >= 8192: Long sequence (large benefit)
 */
export const FLASH_ATTENTION_SHORT_THRESHOLD = 2048
export const FLASH_ATTENTION_LONG_THRESHOLD = 8192
