import { z } from 'zod'

/**
 * Maximum expressible sequence length, in tokens
 *
 * Set by the largest context window in the model database (Llama 4 Scout,
 * 10,485,760). This is the bound on what the calculator can *express*, not on what
 * is advisable — the UI marks each model's native context separately and warns
 * rather than clamping, because RoPE/YaRN extension beyond native context is a real
 * workload.
 */
export const MAX_SEQUENCE_LENGTH = 10_485_760

/**
 * Every weight quantization format, the single source for the QuantizationFormat
 * type (src/engines/types.ts) and every Zod enum over formats.
 */
export const QUANTIZATION_FORMATS = [
  'fp32',
  'fp16',
  'bf16',
  'fp8',
  'mxfp4',
  'nvfp6',
  'nvfp4',
  'int8',
  'int4',
  'nf4',
  'gptq',
  'awq',
  'gguf-q8_0',
  'gguf-q6_k',
  'gguf-q5_k_s',
  'gguf-q5_k_m',
  'gguf-q5_0',
  'gguf-q4_k_s',
  'gguf-q4_k_m',
  'gguf-q4_0',
  'gguf-q3_k_l',
  'gguf-q3_k_m',
  'gguf-q3_k_s',
  'gguf-q2_k',
] as const

// GPU Schema based on research (dbgpu fields)
export const GPUSchema = z.object({
  id: z.string().min(1),
  name: z.string(),
  manufacturer: z.enum(['nvidia', 'amd', 'apple', 'intel']),
  vram_gb: z.number().positive(),
  memory_bandwidth_gbps: z.number().positive(),
  memory_type: z.string(),
  bus_width: z.number().int().nonnegative(), // 0 for unified memory (Apple Silicon)

  /**
   * Largest GPU count that can exist inside one node for this part.
   *
   * Derivation: min(coherent interconnect limit, largest shipping chassis slot
   * count). For parts with no coherent domain — anything on PCIe — the chassis
   * bound alone.
   *
   * This is a HARD bound: configurations above it cannot be built. It is not
   * the same as INTERCONNECT_SPECS.recommendedMaxTPDegree, which is a SOFT
   * warning about configurations that are buildable but scale badly. Eight
   * RTX PRO 6000 in a Dell XE7745 is both buildable (8) and a poor tensor-
   * parallel target (4).
   *
   * Required, not optional: a row added later must state its own bound rather
   * than inherit a default that happens to be wrong.
   */
  max_gpus_per_node: z.number().int().positive(),

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

  /**
   * CPU and GPU share one memory pool (Apple Silicon, GB10 DGX Spark). The only
   * source for "no separate host memory" (config-rules R6): never infer it from
   * `interconnect === 'unified'` or from the tier.
   */
  unified_memory: z.boolean().optional(),

  // Classification
  tier: z.enum(['datacenter', 'consumer', 'apple-silicon']),

  // Metadata (optional, for linking to spec sheets)
  spec_url: z.string().url().optional(),
})

/**
 * User-specified scale-out fabric
 *
 * port_gbps is unidirectional GB/s per port. NIC count is not asked for: it is
 * derived as one NIC per GPU, the standard AI-node build.
 */
export const CustomFabricSchema = z.object({
  name: z.string().min(1),
  port_gbps: z.number().positive().max(10_000),
})

export type CustomFabricInput = z.infer<typeof CustomFabricSchema>

// Model Schema based on HuggingFace config.json fields
const ModelFields = z.object({
  id: z.string().min(1),
  name: z.string(),
  architecture: z.enum(['dense', 'moe']),
  num_parameters_billion: z.number().positive(),
  hidden_size: z.number().int().positive(),
  num_hidden_layers: z.number().int().positive(),
  num_attention_heads: z.number().int().positive(),

  // GQA field (optional - fallback to num_attention_heads if missing)
  num_kv_heads: z.number().int().positive().optional(),

  // Exotic-attention override: total cached elements per token across ALL layers
  // (MLA latent dims, hybrid attention-layers-only, explicit head_dim). When present,
  // the KV-cache engine uses it directly instead of the GQA formula.
  kv_cache_elements_per_token: z.number().int().positive().optional(),

  // Sliding-window / chunked-local layers (Gemma 3/4, gpt-oss, Llama 4): cached elements
  // per token across those layers, and the window they are capped at. vLLM allocates
  // them at min(window, context) tokens. kv_cache_elements_per_token then counts only
  // the full-attention layers.
  kv_sliding_elements_per_token: z.number().int().positive().optional(),
  kv_sliding_window: z.number().int().positive().optional(),

  // Multi-head latent attention (config.json carries kv_lora_rank). vLLM caches one latent
  // per token and duplicates it on every tensor-parallel rank, so TP does not split it.
  use_mla: z.boolean().optional(),

  // Linear-attention / SSM layers (gated delta net, KDA, mamba2, short conv): bytes of conv +
  // recurrent state one session holds at TP=1, as vLLM allocates it. Constant in context
  // length; its dtype comes from the model config (fp32 SSM state for Qwen3.5 and NemotronH).
  linear_state_bytes_per_session: z.number().int().positive().optional(),

  // Measured weight-file size per format from a published checkpoint (GiB = bytes / 1024^3).
  // Recipes differ (what stays 16-bit depends on who quantized it), so each format carries
  // its own reference repo. Absent formats use BYTES_PER_PARAMETER.
  weight_refs: z
    .partialRecord(
      z.enum(QUANTIZATION_FORMATS),
      z.object({
        repo: z.string().min(1),
        gib: z.number().positive(),
        high_precision: z
          .object({ params_b: z.number().positive(), gib: z.number().positive() })
          .optional(),
      }),
    )
    .optional(),

  intermediate_size: z.number().int().positive(),

  // MoE fields (optional, only for MoE architectures)
  num_experts: z.number().int().positive().optional(),
  num_experts_per_token: z.number().int().positive().optional(),

  // Active parameters per token for MoE models (e.g. 3 for a 36B "A3B" model).
  // Used for decode throughput and prefill FLOPs only — weight VRAM always uses
  // num_parameters_billion, because every expert must be resident.
  // Absent means "derive it"; see calculateMoEActiveParams tier 2.
  active_parameters_billion: z.number().positive().optional(),

  // Metadata fields (optional, for display and linking)
  context_length: z.number().int().positive().optional(),
  license: z.string().optional(),
  hf_url: z.string().url().optional(),
})

export const ModelSchema = ModelFields.refine(
  (m) => (m.kv_sliding_elements_per_token === undefined) === (m.kv_sliding_window === undefined),
  {
    error: 'kv_sliding_elements_per_token and kv_sliding_window must be set together',
    path: ['kv_sliding_window'],
  },
)

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
  sequenceLength: z.number().int().min(512).max(MAX_SEQUENCE_LENGTH),

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
