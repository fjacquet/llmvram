import Decimal from 'decimal.js'
import type {
  InterconnectSpec,
  InterconnectType,
  KVCachePrecision,
  QuantizationFormat,
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
 * Integer formats: Native precision
 * - int8: 1 byte, int4/nf4: 0.5 bytes
 *
 * Compressed formats with overhead (NOT pure bits-per-weight):
 * - gptq: 0.6 bytes (4-bit + 1.2x overhead for codebooks/scales)
 * - awq: 0.6 bytes (4-bit + 1.2x overhead for activation-aware quantization)
 *
 * GGUF formats (empirical bits-per-parameter, NOT theoretical):
 * - q4_0: 4.5 bpp → 0.5625 bytes
 * - q4_k_m: 4.8 bpp → 0.6 bytes (includes K-quantization metadata)
 * - q5_k_m: 5.6 bpp → 0.7 bytes
 * - q6_k: 6.5 bpp → 0.8125 bytes
 * - q8_0: 8.5 bpp → 1.0625 bytes
 *
 * Sources:
 * - GPTQ/AWQ overhead: localaimaster.com/gptq-vs-awq-quantization-methods (INFER-01)
 * - GGUF empirical bpp: Artefact2 gist github.com/Artefact2/d7dc977a6c7288ac784cab80e3f3a700 (INFER-02)
 */
export const BYTES_PER_PARAMETER: Record<QuantizationFormat, Decimal> = {
  // Float formats
  fp32: new Decimal(4.0),
  fp16: new Decimal(2.0),
  bf16: new Decimal(2.0),

  // Integer formats
  int8: new Decimal(1.0),
  int4: new Decimal(0.5),
  nf4: new Decimal(0.5),

  // Compressed formats (4-bit base + 1.2x overhead)
  gptq: new Decimal(0.6), // 0.5 * 1.2
  awq: new Decimal(0.6), // 0.5 * 1.2

  // GGUF formats (empirical bpp / 8)
  'gguf-q4_0': new Decimal(0.5625), // 4.5 / 8
  'gguf-q4_k_m': new Decimal(0.6), // 4.8 / 8
  'gguf-q5_k_m': new Decimal(0.7), // 5.6 / 8
  'gguf-q6_k': new Decimal(0.8125), // 6.5 / 8
  'gguf-q8_0': new Decimal(1.0625), // 8.5 / 8
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
  },
  'nvlink-5': {
    type: 'nvlink-5',
    bandwidthGBps: 1800,
    recommendedMaxTPDegree: 8,
  },
  'pcie-4': {
    type: 'pcie-4',
    bandwidthGBps: 64,
    recommendedMaxTPDegree: 2,
  },
  'pcie-5': {
    type: 'pcie-5',
    bandwidthGBps: 128,
    recommendedMaxTPDegree: 4,
  },
  none: {
    type: 'none',
    bandwidthGBps: 0,
    recommendedMaxTPDegree: 1,
  },
}
