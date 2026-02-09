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

  // Classification
  tier: z.enum(['datacenter', 'consumer', 'apple-silicon']),
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
