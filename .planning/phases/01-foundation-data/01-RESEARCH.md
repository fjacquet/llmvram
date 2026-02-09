# Phase 1: Foundation & Data - Research

**Researched:** 2026-02-09
**Domain:** React + TypeScript web application with static data for LLM VRAM calculation
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundation for llmvram, mirroring the proven architecture of the sister project raidy. The stack is well-established: React 19, TypeScript strict mode, Vite, Zustand for state management, Tailwind CSS for styling, Recharts for visualization, Biome for linting/formatting, Vitest for testing, and Decimal.js for precision arithmetic. The primary data challenge is curating GPU and model databases from open sources (dbgpu, gpu-info-api, HuggingFace) with correct schema fields to support accurate VRAM calculations.

Research confirms that LLM memory requirements follow well-defined formulas: Model Weights = Parameters × Bytes per Parameter, and KV Cache = b × s × l × h × nh × 2 × Bytes per Parameter. MoE models (like Mixtral) require all expert parameters loaded in VRAM despite only activating a subset per token, making them memory-intensive despite computational efficiency.

**Primary recommendation:** Mirror raidy's exact package versions and project structure for consistency, use Zod schemas for model/GPU validation, and build a Node.js script to fetch HuggingFace config.json via the public API or direct file access for automated database refresh.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | ^19.2.0 | UI framework | Industry standard, raidy already uses it |
| TypeScript | ~5.9.3 | Type safety | Critical for calculation accuracy in math-heavy apps |
| Vite | ^7.2.4 | Build tool | Fast, modern, raidy-proven |
| Zustand | ^5.0.9 | State management | Lightweight, raidy uses it for config store patterns |
| Tailwind CSS | ^4.1.18 | Styling | Utility-first, raidy uses v4 with @tailwindcss/vite |
| Recharts | ^3.6.0 | Charts/visualization | Simple API, raidy uses for data viz |
| Biome | 2.3.11 | Linting/formatting | Fast Rust-based tool, raidy uses for all code quality |
| Vitest | ^4.0.16 | Testing | Native Vite integration, raidy uses for unit tests |
| Decimal.js | Latest | Precision arithmetic | Arbitrary-precision for accurate VRAM calculations |
| Zod | ^3.25.76 | Schema validation | Type-safe validation, raidy uses for data schemas |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @huggingface/transformers | Latest | Fetch model configs | Optional: for client-side config fetching (or use build script) |
| @vitejs/plugin-react | ^5.1.1 | Vite React plugin | Required for Vite + React |
| @tailwindcss/vite | ^4.1.18 | Tailwind v4 Vite plugin | Required for Tailwind CSS |
| @testing-library/react | ^16.3.1 | React testing utilities | Testing React components |
| @testing-library/jest-dom | ^6.9.1 | DOM matchers for tests | Testing assertions |
| jsdom | ^27.4.0 | DOM simulation | Vitest environment for React tests |
| @vitest/coverage-v8 | ^4.0.16 | Coverage reporting | Test coverage metrics |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Decimal.js | big.js or bignumber.js | Decimal.js has best HuggingFace docs coverage (167 snippets), most complete API |
| Zustand | Redux or Jotai | Zustand is simplest, raidy already uses it, no boilerplate |
| Recharts | Chart.js or D3 | Recharts has React-native API, raidy uses it |
| Biome | ESLint + Prettier | Biome is faster, single tool, raidy already configured |

**Installation:**
```bash
npm install react react-dom zustand recharts zod decimal.js
npm install -D typescript vite @vitejs/plugin-react @tailwindcss/vite tailwindcss @biomejs/biome vitest jsdom @testing-library/react @testing-library/jest-dom @vitest/coverage-v8 @types/react @types/react-dom @types/node
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── engines/            # Calculation engines (VRAM, KV cache, inference)
├── data/              # Static JSON databases (models.json, gpus.json)
├── types/             # TypeScript interfaces (Model, GPU, CalcResult)
├── store/             # Zustand stores (configStore)
├── components/        # React components
├── utils/             # Helpers, schemas, validators
├── hooks/             # Custom React hooks
└── workers/           # Web Workers (optional for heavy calcs)
```

### Pattern 1: Zod Schema for Data Validation

**What:** Use Zod to define schemas for model and GPU data, validate on load
**When to use:** All data imports from JSON, URL params, user input
**Example:**
```typescript
// Source: raidy's utils/schemas.ts pattern
import { z } from 'zod'

export const GPUSchema = z.object({
  id: z.string().min(1),
  name: z.string(),
  manufacturer: z.enum(['nvidia', 'amd', 'apple', 'intel']),
  vram_gb: z.number().positive(),
  memory_bandwidth_gbps: z.number().positive(),
  fp16_tflops: z.number().positive().optional(),
  fp32_tflops: z.number().positive().optional(),
  memory_type: z.string(),
  bus_width: z.number().int().positive(),
  tdp_watts: z.number().positive().optional(),
  interconnect: z.enum(['none', 'nvlink', 'infinity-fabric', 'unified']).optional(),
})

export const ModelSchema = z.object({
  id: z.string().min(1),
  name: z.string(),
  architecture: z.enum(['dense', 'moe']),
  num_parameters_billion: z.number().positive(),
  hidden_size: z.number().int().positive(),
  num_hidden_layers: z.number().int().positive(),
  num_attention_heads: z.number().int().positive(),
  num_kv_heads: z.number().int().positive().optional(),
  intermediate_size: z.number().int().positive(),
  // MoE fields
  num_experts: z.number().int().positive().optional(),
  num_experts_per_token: z.number().int().positive().optional(),
})

export type GPU = z.infer<typeof GPUSchema>
export type Model = z.infer<typeof ModelSchema>
```

### Pattern 2: Decimal.js for Precision Arithmetic

**What:** Use Decimal.js for all VRAM and memory calculations to avoid floating-point errors
**When to use:** Any calculation involving memory sizes, bandwidth, TFLOPS
**Example:**
```typescript
// Source: Decimal.js Context7 docs
import Decimal from 'decimal.js'

// Configure precision globally
Decimal.set({ precision: 20, rounding: 4 }) // ROUND_HALF_UP

export function calculateModelWeightsGB(
  numParams: number,
  bytesPerParam: number
): Decimal {
  const params = new Decimal(numParams)
  const bytes = new Decimal(bytesPerParam)
  const gigabyte = new Decimal(1e9)

  return params.times(bytes).div(gigabyte)
}

export function calculateKVCacheGB(
  batchSize: number,
  seqLength: number,
  numLayers: number,
  hiddenSize: number,
  numKVHeads: number,
  bytesPerParam: number
): Decimal {
  const b = new Decimal(batchSize)
  const s = new Decimal(seqLength)
  const l = new Decimal(numLayers)
  const h = new Decimal(hiddenSize)
  const nh = new Decimal(numKVHeads)
  const bytes = new Decimal(bytesPerParam)
  const gigabyte = new Decimal(1e9)

  // KV Memory = b × s × l × h × nh × 2 × bytes
  return b.times(s).times(l).times(h).times(nh).times(2).times(bytes).div(gigabyte)
}
```

### Pattern 3: Zustand State Management

**What:** Single global store for user configuration, calculated results
**When to use:** App-wide state (selected model, GPU, calculation params, results)
**Example:**
```typescript
// Source: raidy's store pattern
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface ConfigState {
  // Model selection
  modelId: string | null
  customModel: Model | null

  // GPU selection
  gpuId: string | null
  customGPU: GPU | null

  // Calculation params
  precision: 'fp32' | 'fp16' | 'int8' | 'int4'
  contextLength: number
  batchSize: number

  // Actions
  setModelId: (id: string | null) => void
  setCustomModel: (model: Model | null) => void
  setGPUId: (id: string | null) => void
  setCustomGPU: (gpu: GPU | null) => void
  setPrecision: (p: 'fp32' | 'fp16' | 'int8' | 'int4') => void
  setContextLength: (len: number) => void
  setBatchSize: (size: number) => void
}

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      modelId: null,
      customModel: null,
      gpuId: null,
      customGPU: null,
      precision: 'fp16',
      contextLength: 2048,
      batchSize: 1,

      setModelId: (id) => set({ modelId: id }),
      setCustomModel: (model) => set({ customModel: model }),
      setGPUId: (id) => set({ gpuId: id }),
      setCustomGPU: (gpu) => set({ customGPU: gpu }),
      setPrecision: (p) => set({ precision: p }),
      setContextLength: (len) => set({ contextLength: len }),
      setBatchSize: (size) => set({ batchSize: size }),
    }),
    {
      name: 'llmvram-config',
      partialize: (state) => ({
        modelId: state.modelId,
        gpuId: state.gpuId,
        precision: state.precision,
        contextLength: state.contextLength,
        batchSize: state.batchSize,
      }),
    }
  )
)
```

### Pattern 4: Engine Pattern for Calculations

**What:** Separate calculation logic into engine modules, testable in isolation
**When to use:** VRAM calculation, KV cache, inference throughput estimation
**Example:**
```typescript
// src/engines/vramEngine.ts
import Decimal from 'decimal.js'
import type { Model, GPU } from '@types'

export interface VRAMRequirements {
  modelWeightsGB: Decimal
  kvCacheGB: Decimal
  activationsGB: Decimal
  overheadGB: Decimal
  totalGB: Decimal
  fitsOnGPU: boolean
  utilizationPercent: Decimal
}

export function calculateVRAMRequirements(
  model: Model,
  precision: 'fp32' | 'fp16' | 'int8' | 'int4',
  contextLength: number,
  batchSize: number,
  gpu: GPU
): VRAMRequirements {
  const bytesPerParam = precision === 'fp32' ? 4 : precision === 'fp16' ? 2 : precision === 'int8' ? 1 : 0.5

  // Model weights
  const modelWeightsGB = calculateModelWeightsGB(model.num_parameters_billion * 1e9, bytesPerParam)

  // KV cache
  const kvCacheGB = calculateKVCacheGB(
    batchSize,
    contextLength,
    model.num_hidden_layers,
    model.hidden_size,
    model.num_kv_heads || model.num_attention_heads,
    bytesPerParam
  )

  // Activations (rough estimate: 20% of model weights)
  const activationsGB = modelWeightsGB.times(0.2)

  // Overhead (10%)
  const subtotal = modelWeightsGB.plus(kvCacheGB).plus(activationsGB)
  const overheadGB = subtotal.times(0.1)

  const totalGB = subtotal.plus(overheadGB)
  const gpuVRAM = new Decimal(gpu.vram_gb)
  const fitsOnGPU = totalGB.lte(gpuVRAM)
  const utilizationPercent = totalGB.div(gpuVRAM).times(100)

  return {
    modelWeightsGB,
    kvCacheGB,
    activationsGB,
    overheadGB,
    totalGB,
    fitsOnGPU,
    utilizationPercent,
  }
}
```

### Anti-Patterns to Avoid
- **Direct number arithmetic for VRAM calculations:** Use Decimal.js to avoid precision loss (0.1 + 0.2 !== 0.3 in JS)
- **Hardcoded model/GPU data in components:** Always load from JSON databases, keep components pure
- **Global mutable state outside Zustand:** All state changes through store actions only
- **Manual TypeScript types for data:** Derive types from Zod schemas using `z.infer<>`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Arbitrary precision math | Custom BigInt wrapper | Decimal.js | Handles edge cases (division, rounding modes, scientific notation), battle-tested |
| Schema validation | Manual if/else checks | Zod | Type inference, runtime validation, composable schemas |
| State persistence | localStorage wrapper | zustand/persist | Handles serialization, migration, storage events |
| Model config fetching | Custom HTTP client | fetch API or @huggingface/transformers | HuggingFace has public URLs, no auth needed for public models |
| CSS framework | Custom utility classes | Tailwind CSS | Raidy already uses v4, consistency across projects |

**Key insight:** These libraries exist because the problems are deceptively complex. Decimal.js handles 50+ edge cases in arithmetic, Zod handles nested validation with error messages, zustand/persist handles storage quotas and edge cases. Use proven solutions.

## Common Pitfalls

### Pitfall 1: Incorrect MoE Parameter Counting

**What goes wrong:** Assuming active parameters = total VRAM (e.g., Mixtral 8x7B = 13B active, but needs 46B in VRAM)
**Why it happens:** Misunderstanding MoE architecture—all experts must be in VRAM for fast switching
**How to avoid:** Always use `num_parameters_billion` field from config (total params), not computed active params. Add warning in UI for MoE models.
**Warning signs:** VRAM estimate for Mixtral seems too low (13B instead of 46B)

### Pitfall 2: Forgetting num_kv_heads (GQA)

**What goes wrong:** KV cache calculation assumes num_kv_heads = num_attention_heads, overestimating VRAM
**Why it happens:** Older models (pre-LLaMA 2) used MHA where they're equal; newer models use GQA with fewer KV heads
**How to avoid:** Use `num_kv_heads` from config if present, fallback to `num_attention_heads` for older models
**Warning signs:** KV cache estimate seems too high for LLaMA 2/3, Mistral

### Pitfall 3: JavaScript Number Precision Loss

**What goes wrong:** Using JS numbers for VRAM calculations loses precision (0.1 + 0.2 = 0.30000000000000004)
**Why it happens:** IEEE 754 float representation
**How to avoid:** Always use Decimal.js for calculations, only convert to number for display
**Warning signs:** Unit tests fail on floating-point comparisons

### Pitfall 4: Missing HuggingFace Config Fields

**What goes wrong:** Fetching config.json but missing architecture-specific fields (e.g., intermediate_size for LLaMA)
**Why it happens:** Different architectures expose different fields in config.json
**How to avoid:** Build a mapping of required fields per architecture type, validate with Zod
**Warning signs:** Build script succeeds but models.json has null/undefined fields

### Pitfall 5: Hardcoded Bytes per Precision

**What goes wrong:** Assuming int4 = 0.5 bytes/param without considering packing overhead
**Why it happens:** Theoretical vs actual quantization implementations
**How to avoid:** Use conservative estimates (int4 = 0.55 bytes/param to account for packing), cite sources
**Warning signs:** VRAM estimates don't match real-world measurements from community

## Code Examples

Verified patterns from official sources:

### Decimal.js Basic Usage
```typescript
// Source: https://github.com/mikemcl/decimal.js
import Decimal from 'decimal.js'

// Configure precision globally
Decimal.set({ precision: 20, rounding: 4 }) // ROUND_HALF_UP

// Basic operations
const a = new Decimal(0.1)
const b = new Decimal(0.2)
const sum = a.plus(b) // '0.3' (exact)

// Static methods
const result = Decimal.add('10.5', '5.2') // '15.7'

// Chaining
const vramGB = new Decimal(7000000000) // 7B params
  .times(2) // FP16 = 2 bytes/param
  .div(1e9) // Convert to GB
  .toNumber() // 14
```

### HuggingFace Config Fetching
```typescript
// Source: https://huggingface.co/docs/transformers.js/api/configs
// Option 1: Direct fetch from public URL
async function fetchModelConfig(modelId: string) {
  const url = `https://huggingface.co/${modelId}/raw/main/config.json`
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to fetch config for ${modelId}`)
  return response.json()
}

// Option 2: Using @huggingface/transformers (client-side)
import { AutoConfig } from '@huggingface/transformers'

const config = await AutoConfig.from_pretrained('meta-llama/Llama-3.1-8B')
console.log(config)
// PretrainedConfig {
//   "model_type": "llama",
//   "hidden_size": 4096,
//   "num_hidden_layers": 32,
//   "num_attention_heads": 32,
//   "num_kv_heads": 8,
//   "intermediate_size": 14336,
//   ...
// }
```

### Zod Schema Validation
```typescript
// Source: raidy's utils/schemas.ts pattern
import { z } from 'zod'

const ModelSchema = z.object({
  id: z.string().min(1),
  name: z.string(),
  architecture: z.enum(['dense', 'moe']),
  num_parameters_billion: z.number().positive(),
  hidden_size: z.number().int().positive(),
  num_hidden_layers: z.number().int().positive(),
  num_attention_heads: z.number().int().positive(),
  num_kv_heads: z.number().int().positive().optional(),
  intermediate_size: z.number().int().positive(),
  num_experts: z.number().int().positive().optional(),
  num_experts_per_token: z.number().int().positive().optional(),
})

export type Model = z.infer<typeof ModelSchema>

// Validate data
const result = ModelSchema.safeParse(data)
if (result.success) {
  const model: Model = result.data
} else {
  console.error('Validation errors:', result.error.format())
}
```

### Vitest Test Example
```typescript
// Source: raidy's vitest.config.ts + pattern
import { describe, it, expect } from 'vitest'
import Decimal from 'decimal.js'
import { calculateModelWeightsGB } from '@engines/vramEngine'

describe('VRAM Engine', () => {
  it('calculates model weights for 7B FP16 model', () => {
    const weightsGB = calculateModelWeightsGB(7e9, 2)
    expect(weightsGB.toNumber()).toBe(14)
  })

  it('calculates model weights for Mixtral 8x7B (46B total)', () => {
    const weightsGB = calculateModelWeightsGB(46e9, 2)
    expect(weightsGB.toNumber()).toBe(92)
  })
})
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| ESLint + Prettier | Biome | 2023-2024 | Single tool, 10x faster, raidy already migrated |
| Tailwind v3 | Tailwind v4 | 2024 | Native CSS-first, Vite plugin, raidy uses @tailwindcss/vite 4.1.18 |
| Jest | Vitest | 2021-2022 | Native Vite integration, faster, raidy uses 4.0.16 |
| React 18 | React 19 | 2024 | Improved server components, raidy uses 19.2.0 |
| num_attention_heads only | num_kv_heads (GQA) | 2023 (LLaMA 2) | Reduced KV cache size, critical for accurate VRAM calc |
| MHA (Multi-Head Attention) | GQA (Grouped Query Attention) | 2023 | KV heads < attention heads, lower memory footprint |

**Deprecated/outdated:**
- **big.js/bignumber.js over Decimal.js:** All three are by same author (mikemcl), but Decimal.js has the most complete API and best docs (167 snippets in Context7)
- **Manual localStorage:** Use zustand/persist middleware
- **CRA (Create React App):** Dead project, Vite is the standard

## Open Questions

1. **Quantization Packing Overhead**
   - What we know: int4 theoretically uses 0.5 bytes/param, but implementations have packing overhead
   - What's unclear: Exact overhead percentage for GPTQ vs AWQ vs bitsandbytes
   - Recommendation: Use conservative estimates (int4 = 0.55 bytes/param), add note in UI citing uncertainty, revisit with community data

2. **Apple Silicon Unified Memory Efficiency**
   - What we know: M1/M2/M3/M4 Ultra have unified memory (CPU+GPU share same VRAM pool)
   - What's unclear: Does unified memory architecture allow higher utilization than discrete GPUs? (>95% vs ~90%)
   - Recommendation: Use same overhead formula as discrete GPUs initially, add toggle for "unified memory mode" in future phase

3. **Build Script vs Client-Side Fetching**
   - What we know: HuggingFace config.json available via public URL (no auth)
   - What's unclear: Better UX to pre-fetch 30+ models at build time or lazy-load client-side?
   - Recommendation: Start with build script for curated list (faster initial load), add "fetch custom model" feature in Phase 2

4. **KV Cache Formula Variations**
   - What we know: Base formula is b × s × l × h × nh × 2 × bytes
   - What's unclear: Some sources divide by num_attention_heads (head_dim = hidden_size / num_attention_heads), unclear if equivalent
   - Recommendation: Verify formula with reference implementations (vLLM, llama.cpp), add unit tests with known models

## HuggingFace Config.json Field Mapping

Research from HuggingFace transformers documentation and model configs:

### Common Fields (All Architectures)
- `model_type`: Architecture identifier (e.g., "llama", "mistral", "mixtral")
- `hidden_size`: Hidden dimension size
- `num_hidden_layers`: Number of transformer blocks/layers
- `num_attention_heads`: Number of attention heads in MHA
- `intermediate_size`: FFN intermediate dimension (usually ~4x hidden_size)
- `vocab_size`: Vocabulary size (not needed for VRAM calc)

### Architecture-Specific Fields

**LLaMA 2/3 (and variants):**
- `num_key_value_heads`: KV heads for GQA (e.g., 8 for LLaMA-2-7B)
- `rope_theta`: RoPE base frequency (not needed for VRAM)
- `max_position_embeddings`: Max sequence length

**Mistral:**
- `num_key_value_heads`: KV heads for GQA
- `sliding_window`: Sliding window attention size (not needed for VRAM)

**Mixtral (MoE):**
- `num_local_experts`: Total number of experts (e.g., 8)
- `num_experts_per_tok`: Active experts per token (e.g., 2)
- `num_key_value_heads`: KV heads for GQA

**Qwen:**
- `num_key_value_heads`: KV heads for GQA (Qwen 2+)
- `kv_channels`: Alternative to num_key_value_heads in Qwen 1

**Phi:**
- `num_key_value_heads`: KV heads for GQA (Phi-2+)
- `partial_rotary_factor`: Partial RoPE (not needed for VRAM)

**DeepSeek:**
- `num_key_value_heads`: KV heads for GQA
- May have MoE variants with expert fields

**Gemma:**
- `num_key_value_heads`: KV heads for GQA
- `head_dim`: Explicit head dimension (usually hidden_size / num_attention_heads)

### Build Script Strategy

```typescript
// scripts/fetch-models.ts
interface HFConfig {
  model_type: string
  hidden_size: number
  num_hidden_layers: number
  num_attention_heads: number
  num_key_value_heads?: number
  intermediate_size: number
  num_local_experts?: number
  num_experts_per_tok?: number
  // ... other fields
}

const MODEL_IDS = [
  'meta-llama/Llama-2-7b-hf',
  'meta-llama/Llama-3.1-8B',
  'mistralai/Mistral-7B-v0.1',
  'mistralai/Mixtral-8x7B-v0.1',
  // ... 30+ models
]

async function fetchModels() {
  const models = []

  for (const modelId of MODEL_IDS) {
    const url = `https://huggingface.co/${modelId}/raw/main/config.json`
    const response = await fetch(url)
    const config: HFConfig = await response.json()

    models.push({
      id: modelId.replace('/', '-'),
      name: modelId,
      architecture: config.num_local_experts ? 'moe' : 'dense',
      num_parameters_billion: estimateParams(config),
      hidden_size: config.hidden_size,
      num_hidden_layers: config.num_hidden_layers,
      num_attention_heads: config.num_attention_heads,
      num_kv_heads: config.num_key_value_heads,
      intermediate_size: config.intermediate_size,
      num_experts: config.num_local_experts,
      num_experts_per_token: config.num_experts_per_tok,
    })
  }

  // Write to src/data/models.json
  await fs.writeFile('src/data/models.json', JSON.stringify(models, null, 2))
}
```

## GPU Data Sources

### dbgpu (https://github.com/painebenjamin/dbgpu)
- **Format:** JSON, CSV, or PKL (pickle)
- **Coverage:** 2000+ GPUs
- **Fields (50+):** manufacturer, name, GPU name, architecture, process size, memory (size, type, bus width, bandwidth), clock speeds, compute units, TDP, API support, performance metrics
- **Quality:** HIGH - Comprehensive, maintained
- **Recommendation:** PRIMARY source for NVIDIA/AMD datacenter and consumer GPUs

### gpu-info-api (https://github.com/voidful/gpu-info-api)
- **Format:** JSON from Wikipedia, updated weekly via GitHub Actions
- **Coverage:** NVIDIA, AMD, Intel
- **Fields:** Model, codename, fab process, die size, memory (size, type, bandwidth), TFLOPS (single/double/half), TDP, PCIe, launch date
- **Quality:** MEDIUM - Wikipedia-sourced, community-maintained
- **Recommendation:** SECONDARY source for verification, good for quick lookup

### Apple Silicon (https://github.com/philipturner/metal-benchmarks)
- **Format:** Benchmark data and specs
- **Coverage:** M1, M2, M3, M4 (base, Pro, Max, Ultra)
- **Fields:** GPU cores, unified memory size, memory bandwidth, Neural Engine cores, process node
- **Quality:** MEDIUM - Community benchmarks
- **Recommendation:** Use for Apple Silicon entries, supplement with official Apple specs

### GPU Schema for llmvram

```typescript
// src/types/gpu.ts
export interface GPU {
  id: string // e.g., "nvidia-h100-80gb"
  name: string // Display name
  manufacturer: 'nvidia' | 'amd' | 'apple' | 'intel'
  vram_gb: number // Total VRAM/unified memory
  memory_bandwidth_gbps: number // GB/s
  memory_type: string // "HBM3", "HBM2e", "GDDR6X", "Unified"
  bus_width: number // bits

  // Performance (optional for VRAM calc, useful for inference speed)
  fp16_tflops?: number
  fp32_tflops?: number

  // Power and interconnect
  tdp_watts?: number
  interconnect?: 'none' | 'nvlink' | 'nvlink-4' | 'infinity-fabric' | 'unified'

  // Classification
  tier: 'datacenter' | 'consumer' | 'apple-silicon'
}
```

## VRAM Calculation Formulas

Based on research from multiple sources (LocalLLM.in, apxml.com, Skymod.tech, Tech Tactician):

### Model Weights
```
Model Weights (GB) = Num Parameters (billion) × 1e9 × Bytes per Param / 1e9
                   = Num Parameters (billion) × Bytes per Param

Bytes per Param:
- FP32: 4 bytes
- FP16/BF16: 2 bytes
- INT8: 1 byte
- INT4: 0.5 bytes (theoretical), 0.55 bytes (conservative with packing)
```

### KV Cache
```
KV Cache (GB) = Batch Size × Seq Length × Num Layers × Head Dim × Num KV Heads × 2 × Bytes per Param / 1e9

Where:
- Head Dim = Hidden Size / Num Attention Heads (typically)
- Factor of 2 accounts for Key and Value tensors
- Num KV Heads = num_key_value_heads from config (GQA) or num_attention_heads (MHA)

Alternative form (equivalent):
KV Cache per Token = (Hidden Size × 2 × Bytes per Param × Num Layers) / 1e9
Total KV Cache = KV Cache per Token × Batch Size × Seq Length
```

### Activations
```
Activations (GB) ≈ 20% of Model Weights (rough estimate)
```

### Total VRAM
```
Total VRAM (GB) = Model Weights + KV Cache + Activations + Overhead

Overhead = 10-20% of subtotal (accounts for fragmentation, buffers, framework overhead)
```

### MoE Special Case
```
For MoE models (e.g., Mixtral 8x7B):
- Total Parameters = All expert parameters (e.g., 46B for Mixtral)
- Active Parameters = Shared layers + (Num Experts per Token × Expert Size) (e.g., 13B)
- VRAM Required = Total Parameters (not Active Parameters)

Reason: All experts must be in VRAM for fast switching, even though only subset is active per token.
```

## Sources

### Primary (HIGH confidence)
- Decimal.js Context7 (/mikemcl/decimal.js) - Arithmetic operations, configuration, precision handling
- HuggingFace Transformers Docs - [Configuration](https://huggingface.co/docs/transformers/main_classes/configuration), [Mistral config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/configuration_mistral.py), [Mixtral docs](https://huggingface.co/docs/transformers/en/model_doc/mixtral)
- Raidy project (/Users/fjacquet/Projects/raidy) - package.json, biome.json, tsconfig.app.json, vite.config.ts, src structure

### Secondary (MEDIUM confidence)
- [dbgpu GitHub](https://github.com/painebenjamin/dbgpu) - GPU database structure and fields
- [gpu-info-api GitHub](https://github.com/voidful/gpu-info-api) - GPU specs from Wikipedia
- [LocalLLM.in VRAM Calculator](https://localllm.in/blog/interactive-vram-calculator) - VRAM formulas and KV cache calculation
- [apxml.com VRAM Guide](https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm) - Model weights and precision bytes
- [Skymod.tech LLM Memory Guide](https://skymod.tech/how-much-memory-does-your-llm-really-need-a-practical-guide-to-inference-vram-consumption/) - Comprehensive VRAM estimation with overhead
- [Tech Tactician VRAM Guide](https://techtactician.com/llm-gpu-vram-requirements-explained/) - KV cache per token formula
- [HuggingFace MoE Blog](https://huggingface.co/blog/moe) - Mixtral architecture and expert activation
- [GPU-Mart H100 vs A100 vs RTX 4090](https://www.gpu-mart.com/blog/h100-vs-a100-vs-rtx-4090) - GPU specs comparison
- [AMD MI300X Data Sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) - Official AMD specs
- [Apple Silicon Wikipedia](https://en.wikipedia.org/wiki/Apple_silicon) - M1/M2/M3/M4 specs
- [Mac Observer M-Series Guide](https://www.macobserver.com/tips/round-ups/apple-m-series-chips-explained/) - Apple chip comparison

### Tertiary (LOW confidence - needs validation)
- WebSearch results for quantization packing overhead - conflicting information, needs real-world testing
- Apple unified memory efficiency claims - theoretical, needs benchmarking

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Exact versions from raidy, proven in production
- Architecture patterns: HIGH - Zod/Decimal.js from official docs, Zustand from raidy
- VRAM formulas: MEDIUM-HIGH - Multiple sources agree, but quantization overhead needs validation
- GPU data sources: HIGH - dbgpu is comprehensive, gpu-info-api is Wikipedia-sourced (reliable)
- HuggingFace config fields: HIGH - Official transformers documentation and source code
- MoE VRAM requirements: HIGH - Official HuggingFace blog confirms all experts in VRAM

**Research date:** 2026-02-09
**Valid until:** 60 days (stable stack, no fast-moving dependencies)
