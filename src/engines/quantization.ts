import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { BYTES_PER_GB, BYTES_PER_PARAMETER } from './constants'
import type { QuantizationFormat } from './types'

/**
 * Get bytes per parameter for a quantization format
 *
 * Returns the effective bytes-per-parameter including overhead for compressed formats.
 * GPTQ/AWQ include a 16-bit scale and 4-bit zero point per group of 128 (not pure 4-bit).
 * GGUF formats use empirical bits-per-parameter from real-world measurements.
 *
 * @param format - Quantization format (one of 13 supported formats)
 * @returns Bytes per parameter as Decimal (e.g., 0.52 for GPTQ, 4.0 for FP32)
 *
 * @example
 * ```ts
 * getBytesPerParameter('fp16') // Decimal(2.0)
 * getBytesPerParameter('gptq') // Decimal(0.52) - 4-bit + group-128 scale/zero
 * getBytesPerParameter('gguf-q4_k_m') // Decimal(0.6) - 4.8 bpp empirical
 * ```
 */
export function getBytesPerParameter(format: QuantizationFormat): Decimal {
  return BYTES_PER_PARAMETER[format]
}

/**
 * Bytes per parameter for a model in a format: the measured checkpoint when the model
 * has a weight_refs entry for it, else the format constant.
 *
 * Real checkpoints keep some tensors 16-bit and which ones depends on the recipe, so
 * a measured size (spec 2026-09-27) beats any single constant.
 */
export function effectiveBytesPerParameter(format: QuantizationFormat, model?: Model): Decimal {
  const ref = model?.weight_refs?.[format]
  if (ref && model) {
    return new Decimal(ref.gib)
      .mul(BYTES_PER_GB)
      .div(new Decimal(model.num_parameters_billion).mul(1e9))
  }
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
 * active per token. The decode path passes the active (batched) subset, which
 * scales by the same effective bytes per parameter.
 *
 * @param numParametersBillion - Parameters to size, in billions
 * @param format - Quantization format (determines bytes per parameter)
 * @param model - When given and it has a weight_refs entry for `format`, the measured
 *   checkpoint sets the bytes per parameter
 * @returns VRAM requirement in GB (GiB) as Decimal
 *
 * @example
 * ```ts
 * // 7B model in FP16
 * calculateModelWeightVRAM(7.0, 'fp16') // ~13.04 GB
 *
 * // 70B model in GPTQ (4-bit + group-128 scale/zero)
 * calculateModelWeightVRAM(70.0, 'gptq') // ~33.90 GB (0.52 bytes/param)
 *
 * // Mixtral 8x7B in FP16 (TOTAL params, not active)
 * calculateModelWeightVRAM(46.7, 'fp16') // ~86.97 GB
 * ```
 */
export function calculateModelWeightVRAM(
  numParametersBillion: number,
  format: QuantizationFormat,
  model?: Model,
): Decimal {
  const bytesPerParam = effectiveBytesPerParameter(format, model)
  const totalParams = new Decimal(numParametersBillion).mul(1e9)
  const totalBytes = totalParams.mul(bytesPerParam)
  return totalBytes.div(BYTES_PER_GB)
}

/** The repo a weight figure was measured from, or null when it is an estimate. */
export function weightSource(model: Model, format: QuantizationFormat): string | null {
  return model.weight_refs?.[format]?.repo ?? null
}
