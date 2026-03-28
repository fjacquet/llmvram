import { z } from 'zod'

// GPU Schema based on research (dbgpu fields)
export const GPUSchema = z.object({
  id: z.string().min(1),
  name: z.string(),
  manufacturer: z.enum(['nvidia', 'amd', 'apple', 'intel']),
  vram_gb: z.number().positive(),
  memory_bandwidth_gbps: z.number().positive(),
  memory_type: z.string(),
  bus_width: z.number().int().nonnegative(), // 0 for unified memory (Apple Silicon)

  // Performance (optional for inference speed estimation)
  fp16_tflops: z.number().positive().optional(),
  fp32_tflops: z.number().positive().optional(),

  // Power and interconnect
  tdp_watts: z.number().positive().optional(),
  interconnect: z
    .enum([
      'none',
      'nvlink',
      'nvlink-4',
      'nvlink-5',
      'pcie-4',
      'pcie-5',
      'infinity-fabric',
      'unified',
    ])
    .optional(),
  interconnect_options: z
    .array(
      z.enum([
        'none',
        'nvlink',
        'nvlink-4',
        'nvlink-5',
        'pcie-4',
        'pcie-5',
        'infinity-fabric',
        'unified',
      ]),
    )
    .optional(),

  // Classification
  tier: z.enum(['datacenter', 'consumer', 'apple-silicon']),

  // Metadata (optional, for linking to spec sheets)
  spec_url: z.string().url().optional(),
})

// Model Schema based on HuggingFace config.json fields
export const ModelSchema = z.object({
  id: z.string().min(1),
  name: z.string(),
  architecture: z.enum(['dense', 'moe']),
  num_parameters_billion: z.number().positive(),
  hidden_size: z.number().int().positive(),
  num_hidden_layers: z.number().int().positive(),
  num_attention_heads: z.number().int().positive(),

  // GQA field (optional - fallback to num_attention_heads if missing)
  num_kv_heads: z.number().int().positive().optional(),

  intermediate_size: z.number().int().positive(),

  // MoE fields (optional, only for MoE architectures)
  num_experts: z.number().int().positive().optional(),
  num_experts_per_token: z.number().int().positive().optional(),

  // Metadata fields (optional, for display and linking)
  context_length: z.number().int().positive().optional(),
  license: z.string().optional(),
  hf_url: z.string().url().optional(),
})

// Export inferred types
export type GPU = z.infer<typeof GPUSchema>
export type Model = z.infer<typeof ModelSchema>

// Validation helper functions
export function validateGPU(data: unknown): GPU {
  return GPUSchema.parse(data)
}

export function validateModel(data: unknown): Model {
  return ModelSchema.parse(data)
}

export function validateGPUs(data: unknown): GPU[] {
  return z.array(GPUSchema).parse(data)
}

export function validateModels(data: unknown): Model[] {
  return z.array(ModelSchema).parse(data)
}

// Training configuration schema for fine-tuning VRAM calculation
export const TrainingInputSchema = z.object({
  /** Fine-tuning method */
  method: z.enum(['full', 'lora', 'qlora']),

  /** Optimizer type — affects optimizer state memory */
  optimizer: z.enum(['adamw', 'sgd-momentum', 'adamw-8bit', 'adafactor']),

  /** Training precision — affects weight and gradient memory */
  trainingPrecision: z.enum(['fp32', 'fp16', 'bf16']),

  /** Training batch size (micro-batch per GPU) */
  batchSize: z.number().int().min(1).max(128),

  /** Sequence length for training */
  sequenceLength: z.number().int().min(512).max(131072),

  /** LoRA rank — controls adapter capacity (only used for lora/qlora methods) */
  loraRank: z.number().int().min(4).max(256).default(16),

  /** LoRA alpha — scaling factor (alpha/rank), does NOT affect VRAM */
  loraAlpha: z.number().int().min(1).max(512).default(32),

  /** Percentage of linear modules per layer that get LoRA adapters (10-100%) */
  targetModulesPercent: z.number().int().min(10).max(100).default(30),

  /** Gradient accumulation steps — accumulate gradients over multiple micro-batches */
  gradientAccumulationSteps: z.number().int().min(1).max(128).default(1),

  /** Gradient checkpointing — reduce activation memory by ~60% at cost of ~20-25% training time */
  gradientCheckpointing: z.boolean().default(false),

  /** Flash Attention — reduce attention activation memory by 15-70% (sequence-dependent) */
  flashAttention: z.boolean().default(false),
})

export type TrainingInput = z.infer<typeof TrainingInputSchema>

export function validateTrainingInput(data: unknown): TrainingInput {
  return TrainingInputSchema.parse(data)
}
