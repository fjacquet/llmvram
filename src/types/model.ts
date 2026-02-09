import type { Model as ModelBase } from '@utils/schemas'

// Re-export the Zod-inferred type
export type Model = ModelBase

// Type for custom model input (minimal fields required)
export interface CustomModelInput {
  name: string
  num_parameters_billion: number
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_kv_heads?: number
  intermediate_size?: number
}

// Helper to create custom model with sensible defaults
export function createCustomModel(input: CustomModelInput): Model {
  // Default architecture ratios based on common LLM patterns
  const hiddenSize = input.hidden_size || 4096
  const numLayers = input.num_hidden_layers || 32
  const numHeads = input.num_attention_heads || 32

  return {
    id: `custom-${Date.now()}`,
    name: input.name,
    architecture: 'dense',
    num_parameters_billion: input.num_parameters_billion,
    hidden_size: hiddenSize,
    num_hidden_layers: numLayers,
    num_attention_heads: numHeads,
    num_kv_heads: input.num_kv_heads || Math.max(1, Math.floor(numHeads / 4)), // GQA default
    intermediate_size: input.intermediate_size || hiddenSize * 4, // Typical FFN expansion
  }
}
