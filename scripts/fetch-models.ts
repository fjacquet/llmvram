import { writeFile } from 'node:fs/promises'
import { type Model, validateModels } from '../src/utils/schemas'

// Model IDs to fetch — current-generation curated roster (2026-07-03 refresh).
// NOTE: multimodal models (Gemma 4, Qwen3.6, MiniMax M3, Mistral 3) expose the
// transformer fields under config.json `text_config`, which this script does not read;
// and several models are gated or custom-code. Treat fetched output as a starting point
// and hand-curate num_parameters_billion / MoE fields against models.json.
const MODEL_IDS = [
  // Gemma 4 (Google)
  'google/gemma-4-31B-it',
  'google/gemma-4-12B-it',
  'google/gemma-4-26B-A4B-it',

  // Qwen3.6 (Alibaba)
  'Qwen/Qwen3.6-27B',
  'Qwen/Qwen3.6-35B-A3B',

  // DeepSeek V4
  'deepseek-ai/DeepSeek-V4-Flash',
  'deepseek-ai/DeepSeek-V4-Pro',

  // GLM (Z.ai)
  'zai-org/GLM-5.2',

  // Kimi (Moonshot)
  'moonshotai/Kimi-K2-Thinking',
  'moonshotai/Kimi-Linear-48B-A3B-Instruct',

  // Mistral
  'mistralai/Mistral-Medium-3.5-128B',

  // MiniMax
  'MiniMaxAI/MiniMax-M3',

  // NVIDIA Nemotron
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',
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
