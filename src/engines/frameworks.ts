import type { OptimizerType } from './types'

/**
 * Framework presets for training and inference optimizations
 *
 * Maps framework names to their optimal configurations including
 * DeepSpeed ZeRO stages, Unsloth optimizations, and inference frameworks.
 */

/**
 * Supported framework presets
 *
 * Training frameworks:
 * - none: Standard training without framework-specific optimizations
 * - deepspeed-zero1: DeepSpeed ZeRO Stage 1 (partition optimizer states only)
 * - deepspeed-zero2: DeepSpeed ZeRO Stage 2 (partition optimizer states + gradients)
 * - deepspeed-zero3: DeepSpeed ZeRO Stage 3 (partition all training state)
 * - unsloth: Unsloth optimizations (gradient checkpointing + flash attention + 8-bit optimizer)
 *
 * Inference frameworks:
 * - vllm: vLLM inference engine
 * - tgi: HuggingFace Text Generation Inference
 */
export type FrameworkPreset =
  | 'none'
  | 'deepspeed-zero1'
  | 'deepspeed-zero2'
  | 'deepspeed-zero3'
  | 'unsloth'
  | 'vllm'
  | 'tgi'

/**
 * DeepSpeed ZeRO optimization stages
 *
 * - zero-1: Partition optimizer states across GPUs (2x memory reduction)
 * - zero-2: Partition optimizer states + gradients (4x memory reduction)
 * - zero-3: Partition all training state including model parameters (8x memory reduction)
 *
 * CRITICAL: ZeRO stages are NOT simple divide-by-N. Each stage has specific
 * partitioning rules for which components are sharded vs replicated.
 *
 * Reference: .planning/phases/10-framework-presets-multi-gpu-training/10-RESEARCH.md
 */
export type ZeROStage = 'zero-1' | 'zero-2' | 'zero-3'

/**
 * CPU offload configuration for DeepSpeed ZeRO
 *
 * Enables moving optimizer states and/or model parameters from GPU to CPU RAM
 * to reduce GPU memory usage at the cost of training speed.
 */
export interface CPUOffloadConfig {
  /** Offload optimizer states to CPU RAM */
  offloadOptimizer: boolean
  /** Offload model parameters to CPU RAM */
  offloadParameters: boolean
}

/**
 * Framework preset configuration
 *
 * Defines the mode (training vs inference) and automatic optimizations
 * applied when a framework preset is selected.
 */
export interface FrameworkPresetConfig {
  /** Display name for UI */
  name: string
  /** Whether this preset is for training or inference */
  mode: 'training' | 'inference'
  /** DeepSpeed ZeRO stage (null if not a DeepSpeed preset) */
  zeroStage: ZeROStage | null
  /** Optimizations automatically enabled by this preset */
  autoOptimizations: {
    /** Enable gradient checkpointing */
    gradientCheckpointing?: boolean
    /** Enable Flash Attention */
    flashAttention?: boolean
    /** Use specific optimizer */
    optimizer?: OptimizerType
  }
  /** Description for tooltips/help text */
  description: string
}

/**
 * Framework preset configurations
 *
 * Maps each framework preset to its configuration including mode,
 * ZeRO stage (for DeepSpeed), and automatic optimizations.
 */
export const FRAMEWORK_PRESETS: Record<FrameworkPreset, FrameworkPresetConfig> = {
  none: {
    name: 'Standard Training',
    mode: 'training',
    zeroStage: null,
    autoOptimizations: {},
    description: 'Standard single-GPU training without framework-specific optimizations',
  },
  'deepspeed-zero1': {
    name: 'DeepSpeed ZeRO-1',
    mode: 'training',
    zeroStage: 'zero-1',
    autoOptimizations: {
      gradientCheckpointing: true,
      flashAttention: true,
    },
    description: 'Partition optimizer states across GPUs (2x memory reduction)',
  },
  'deepspeed-zero2': {
    name: 'DeepSpeed ZeRO-2',
    mode: 'training',
    zeroStage: 'zero-2',
    autoOptimizations: {
      gradientCheckpointing: true,
      flashAttention: true,
    },
    description: 'Partition optimizer states and gradients (4x memory reduction)',
  },
  'deepspeed-zero3': {
    name: 'DeepSpeed ZeRO-3',
    mode: 'training',
    zeroStage: 'zero-3',
    autoOptimizations: {
      gradientCheckpointing: true,
      flashAttention: true,
    },
    description: 'Partition all training state including model weights (8x memory reduction)',
  },
  unsloth: {
    name: 'Unsloth',
    mode: 'training',
    zeroStage: null,
    autoOptimizations: {
      gradientCheckpointing: true,
      flashAttention: true,
      optimizer: 'adamw-8bit',
    },
    description: 'Unsloth optimizations for efficient fine-tuning',
  },
  vllm: {
    name: 'vLLM',
    mode: 'inference',
    zeroStage: null,
    autoOptimizations: {},
    description: 'vLLM inference engine with PagedAttention',
  },
  tgi: {
    name: 'Text Generation Inference',
    mode: 'inference',
    zeroStage: null,
    autoOptimizations: {},
    description: 'HuggingFace Text Generation Inference optimized serving',
  },
}
