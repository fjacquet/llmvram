import type Decimal from 'decimal.js'
import { z } from 'zod'

/**
 * All 13 supported quantization formats for model weights
 *
 * Float formats: fp32, fp16, bf16
 * Integer formats: int8, int4, nf4
 * Compressed formats with overhead: gptq, awq (4-bit with 1.2x overhead)
 * GGUF variants: q4_0, q4_k_m, q5_k_m, q6_k, q8_0 (empirical bits-per-parameter)
 *
 * Reference: .planning/research/PITFALLS.md - Quantization overhead section
 */
export type QuantizationFormat =
  | 'fp32'
  | 'fp16'
  | 'bf16'
  | 'int8'
  | 'int4'
  | 'nf4'
  | 'gptq'
  | 'awq'
  | 'gguf-q4_0'
  | 'gguf-q4_k_m'
  | 'gguf-q5_k_m'
  | 'gguf-q6_k'
  | 'gguf-q8_0'

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
    'int8',
    'int4',
    'nf4',
    'gptq',
    'awq',
    'gguf-q4_0',
    'gguf-q4_k_m',
    'gguf-q5_k_m',
    'gguf-q6_k',
    'gguf-q8_0',
  ]),
  /** KV cache quantization (defaults to fp16) */
  kvQuantization: z.enum(['fp16', 'fp8', 'int8', 'int4']).default('fp16'),
})

/**
 * Type-safe calculation input (inferred from Zod schema)
 */
export type CalculationInput = z.infer<typeof CalculationInputSchema>
