/**
 * Calculation engine barrel export
 *
 * Provides clean import surface for all engine functions and types.
 * Engines are pure functions designed for testability and Web Worker offloading.
 */

// Constants
export {
  BYTES_PER_PARAMETER,
  FRAMEWORK_OVERHEAD_GB,
  KV_PRECISION_BYTES,
  OPTIMIZER_STATE_BYTES,
  TRAINING_FRAMEWORK_OVERHEAD_GB,
} from './constants'

// DeepSpeed ZeRO engine
export { calculateCPUOffloadMemory, calculateZeROMemoryPerGPU } from './deepspeed'

// Framework presets
export {
  type CPUOffloadConfig,
  FRAMEWORK_PRESETS,
  type FrameworkPreset,
  type FrameworkPresetConfig,
  type ZeROStage,
} from './frameworks'

// Inference engine
export { calculateInferenceVRAM } from './inference'

// KV cache engine
export { calculateKVCacheVRAM } from './kv-cache'

// LoRA/QLoRA training engine
export {
  calculateLoRAAdapterParams,
  calculateLoRAFineTuningVRAM,
  calculateQLoRAFineTuningVRAM,
} from './lora'

// Optimization engine
export {
  applyFlashAttention,
  applyGradientCheckpointing,
  calculateEffectiveBatchSize,
} from './optimizations'

// Performance engine
export { estimatePerformance } from './performance'

// Quantization engine
export { calculateModelWeightVRAM, getBytesPerParameter } from './quantization'

// Training engine
export {
  calculateFullFineTuningVRAM,
  calculateOptimizerStateMemory,
  calculateTrainingActivationMemory,
} from './training'

// Types
export type {
  CalculationInput,
  FineTuningMethod,
  InferenceVRAMBreakdown,
  KVCachePrecision,
  LoRAVRAMBreakdown,
  OptimizerType,
  PerformanceEstimate,
  QuantizationFormat,
  TrainingOptimizationConfig,
  TrainingPrecision,
  TrainingVRAMBreakdown,
  ZeROResult,
} from './types'
export { CalculationInputSchema } from './types'
