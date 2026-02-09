/**
 * Calculation engine barrel export
 *
 * Provides clean import surface for all engine functions and types.
 * Engines are pure functions designed for testability and Web Worker offloading.
 */

// Constants
export { BYTES_PER_PARAMETER, FRAMEWORK_OVERHEAD_GB, KV_PRECISION_BYTES } from './constants'
// Inference engine
export { calculateInferenceVRAM } from './inference'
// KV cache engine
export { calculateKVCacheVRAM } from './kv-cache'

// Performance engine
export { estimatePerformance } from './performance'
// Quantization engine
export { calculateModelWeightVRAM, getBytesPerParameter } from './quantization'
// Types
export type {
  CalculationInput,
  InferenceVRAMBreakdown,
  KVCachePrecision,
  PerformanceEstimate,
  QuantizationFormat,
} from './types'
export { CalculationInputSchema } from './types'
