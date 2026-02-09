import { writeFile } from 'node:fs/promises'
import { type Model, validateModels } from '../src/utils/schemas'

// Model IDs to fetch (matches the 30+ models from Plan 03)
const MODEL_IDS = [
  // LLaMA 2 (3 models)
  'meta-llama/Llama-2-7b-hf',
  'meta-llama/Llama-2-13b-hf',
  'meta-llama/Llama-2-70b-hf',

  // LLaMA 3.1 (4 models)
  'meta-llama/Llama-3.1-8B',
  'meta-llama/Llama-3.1-70B',
  'meta-llama/Llama-3.1-405B',
  'meta-llama/Llama-3.2-3B',

  // Mistral (2 models)
  'mistralai/Mistral-7B-v0.1',
  'mistralai/Mistral-7B-v0.3',

  // Mixtral MoE (3 models)
  'mistralai/Mixtral-8x7B-v0.1',
  'mistralai/Mixtral-8x22B-v0.1',
  'mistralai/Mixtral-8x7B-Instruct-v0.1',

  // Qwen (4 models)
  'Qwen/Qwen2.5-7B',
  'Qwen/Qwen2.5-14B',
  'Qwen/Qwen2.5-32B',
  'Qwen/Qwen2.5-72B',

  // Phi (3 models)
  'microsoft/Phi-3-mini-4k-instruct',
  'microsoft/Phi-3-small-8k-instruct',
  'microsoft/Phi-3-medium-4k-instruct',

  // DeepSeek (3 models)
  'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
  'deepseek-ai/DeepSeek-V2-Lite',
  'deepseek-ai/deepseek-coder-33b-instruct',

  // Gemma (3 models)
  'google/gemma-2b',
  'google/gemma-7b',
  'google/gemma-2-9b',

  // Command-R (2 models)
  'CohereForAI/c4ai-command-r-v01',
  'CohereForAI/c4ai-command-r-plus',

  // Additional (3 models)
  '01-ai/Yi-34B',
  'tiiuae/falcon-40b',
  'mosaicml/mpt-30b',
]

interface HFConfig {
  model_type: string
  hidden_size: number
  num_hidden_layers: number
  num_attention_heads: number
  num_key_value_heads?: number
  intermediate_size?: number
  num_local_experts?: number
  num_experts_per_tok?: number
  // Many other fields we don't need
  [key: string]: unknown
}

async function fetchModelConfig(modelId: string): Promise<Model> {
  const url = `https://huggingface.co/${modelId}/raw/main/config.json`
  console.log(`Fetching ${modelId}...`)

  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${modelId}: ${response.statusText}`)
  }

  const config: HFConfig = await response.json()

  // Determine architecture (MoE if has num_local_experts)
  const architecture = config.num_local_experts ? 'moe' : 'dense'

  // Estimate parameter count based on architecture
  // This is a rough estimate - ideally we'd parse model.safetensors.index.json
  const numParams = estimateParameterCount(config)

  // Create model entry
  // Use intermediate_size from config, or estimate as 4*hidden_size if missing
  const intermediateSize = config.intermediate_size ?? config.hidden_size * 4

  const model: Model = {
    id: modelId.replace('/', '-').toLowerCase(),
    name: modelId.split('/')[1] || modelId,
    architecture,
    num_parameters_billion: numParams,
    hidden_size: config.hidden_size,
    num_hidden_layers: config.num_hidden_layers,
    num_attention_heads: config.num_attention_heads,
    num_kv_heads: config.num_key_value_heads,
    intermediate_size: intermediateSize,
  }

  // Add MoE fields if present
  if (architecture === 'moe') {
    model.num_experts = config.num_local_experts
    model.num_experts_per_token = config.num_experts_per_tok
  }

  return model
}

function estimateParameterCount(config: HFConfig): number {
  // Rough parameter count estimation based on architecture
  // For production, parse model.safetensors.index.json for exact count
  const h = config.hidden_size
  const l = config.num_hidden_layers
  // Use intermediate_size if present, otherwise estimate as 4*hidden_size
  const i = config.intermediate_size ?? h * 4

  // Embedding + layers + output
  // Very rough: (vocab * h) + l * (4*h^2 + 3*h*i) + (vocab * h)
  // Simplified for estimation
  const perLayerParams = 4 * h * h + 3 * h * i

  if (config.num_local_experts) {
    // MoE: shared layers + expert layers
    const expertsPerLayer = config.num_local_experts
    const totalParams = (l * perLayerParams * expertsPerLayer) / 1e9
    return Math.round(totalParams * 10) / 10 // Round to 1 decimal
  }

  const totalParams = (l * perLayerParams) / 1e9
  return Math.round(totalParams * 10) / 10
}

async function main() {
  console.log(`Fetching ${MODEL_IDS.length} model configurations from HuggingFace...`)
  console.log(
    `Note: Many models are gated and require authentication. This script fetches public models only.\n`,
  )

  const models: Model[] = []
  const errors: string[] = []

  for (const modelId of MODEL_IDS) {
    try {
      const model = await fetchModelConfig(modelId)
      models.push(model)
    } catch (error) {
      const errorMsg = `Failed to fetch ${modelId}: ${error}`
      console.error(errorMsg)
      errors.push(errorMsg)
    }
  }

  console.log(`\nSuccessfully fetched ${models.length} models`)
  if (errors.length > 0) {
    console.error(`\nFailed to fetch ${errors.length} models (likely gated):`)
    for (const err of errors) {
      console.error(`  - ${err}`)
    }
    console.error(`\nFor gated models, manually add specs from HuggingFace model cards.`)
  }

  // Validate all models against schema
  console.log('\nValidating models against Zod schema...')
  try {
    validateModels(models)
    console.log('✓ All models valid')
  } catch (error) {
    console.error('✗ Validation failed:', error)
    process.exit(1)
  }

  // Write to temporary file for review
  const outputPath = 'src/data/models-fetched.json'
  await writeFile(outputPath, JSON.stringify(models, null, 2))
  console.log(`\n✓ Wrote ${models.length} models to ${outputPath}`)
  console.log(`\nIMPORTANT: Review ${outputPath} and manually merge with models.json as needed.`)
  console.log(`The curated models.json includes gated models that cannot be auto-fetched.`)
}

main().catch(console.error)
