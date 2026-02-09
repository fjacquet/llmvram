import Decimal from 'decimal.js'
import { BYTES_PER_GB, BYTES_PER_PARAMETER } from './constants'
import type { QuantizationFormat } from './types'

/**
 * Get bytes per parameter for a quantization format
 *
 * Returns the effective bytes-per-parameter including overhead for compressed formats.
 * GPTQ/AWQ include 1.2x overhead for codebooks and scales (not pure 4-bit).
 * GGUF formats use empirical bits-per-parameter from real-world measurements.
 *
 * @param format - Quantization format (one of 13 supported formats)
 * @returns Bytes per parameter as Decimal (e.g., 0.6 for GPTQ, 4.0 for FP32)
 *
 * @example
 * ```ts
 * getBytesPerParameter('fp16') // Decimal(2.0)
 * getBytesPerParameter('gptq') // Decimal(0.6) - includes 1.2x overhead
 * getBytesPerParameter('gguf-q4_k_m') // Decimal(0.6) - 4.8 bpp empirical
 * ```
 */
export function getBytesPerParameter(format: QuantizationFormat): Decimal {
  return BYTES_PER_PARAMETER[format]
}

/**
 * Calculate model weight VRAM requirement
 *
 * Computes VRAM in GB for storing model weights after quantization.
 * Uses Decimal.js for all arithmetic to avoid floating-point precision errors.
 *
 * For MoE models: Pass TOTAL parameters (e.g., 46.7B for Mixtral 8x7B), NOT active
 * parameters (13B). All expert weights must fit in VRAM even though only some are
 * active per token.
 *
 * @param numParametersBillion - Model size in billions of parameters (e.g., 7.0 for 7B, 70.0 for 70B)
 * @param format - Quantization format (determines bytes per parameter)
 * @returns VRAM requirement in GB as Decimal
 *
 * @example
 * ```ts
 * // 7B model in FP16
 * calculateModelWeightVRAM(7.0, 'fp16') // ~13.04 GB
 *
 * // 70B model in GPTQ (4-bit + overhead)
 * calculateModelWeightVRAM(70.0, 'gptq') // ~39.12 GB
 *
 * // Mixtral 8x7B in FP16 (TOTAL params, not active)
 * calculateModelWeightVRAM(46.7, 'fp16') // ~86.97 GB
 * ```
 */
export function calculateModelWeightVRAM(
  numParametersBillion: number,
  format: QuantizationFormat,
): Decimal {
  const bytesPerParam = getBytesPerParameter(format)
  const totalParams = new Decimal(numParametersBillion).mul(1e9)
  const totalBytes = totalParams.mul(bytesPerParam)
  return totalBytes.div(BYTES_PER_GB)
}
