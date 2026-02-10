# Technology Stack: Fine-Tuning Features

**Project:** LLM VRAM Calculator - Fine-Tuning Milestone
**Researched:** 2026-02-10
**Overall Confidence:** HIGH (existing stack verified, no new libraries needed)

## Executive Summary

**NO NEW LIBRARIES REQUIRED** for fine-tuning VRAM estimation, gradient accumulation calculator, and framework presets. The existing validated stack (React 19, Decimal.js, Zod, Recharts, Zustand) is sufficient for all new features.

Fine-tuning features are:

1. **Pure calculation logic** (formulas for optimizer states, gradients, activations)
2. **UI extensions** (additional input fields, memory breakdown visualization)
3. **Static configuration data** (framework presets as JSON)

All can be implemented with the current stack.

---

## Stack Analysis: New Features vs. Existing Capabilities

### Feature 1: Fine-Tuning VRAM Estimation

**What's needed:**

- Calculate model weights (already done)
- Calculate gradients memory (same size as weights in FP32)
- Calculate optimizer states (AdamW: 2x params, SGD: 1x params, 8-bit: 0.5x params)
- Calculate activations memory (batch_size × seq_len × hidden_size × layers)
- Calculate LoRA adapter memory (rank × hidden_size × layers × 2)
- Calculate QLoRA memory (4-bit base + LoRA adapters)

**Stack coverage:**

- ✅ **Decimal.js** (already in stack) - precision arithmetic for all calculations
- ✅ **TypeScript strict** (already in stack) - type safety for training parameters
- ✅ **Zod** (already in stack) - validate training configurations
- ✅ **Pure function pattern** (already established) - engines/fine-tuning.ts follows same pattern as engines/inference.ts

**Confidence:** HIGH - Research confirms these are standard formulas with no specialized libraries available.

---

### Feature 2: Gradient Accumulation Calculator

**What's needed:**

- Calculate effective batch size (micro_batch × gradient_accumulation_steps × num_gpus)
- Calculate memory savings (activations reduce by 1/gradient_accumulation_steps)
- Calculate throughput impact (compute time increases)

**Stack coverage:**

- ✅ **Decimal.js** (already in stack) - arithmetic operations
- ✅ **Pure functions** (already established) - simple mathematical calculations

**Confidence:** HIGH - Pure math, no libraries exist for this specific calculation.

---

### Feature 3: Framework Presets (vLLM, TGI, Unsloth, DeepSpeed ZeRO)

**What's needed:**

- Configuration presets for popular frameworks
- Default values for common training scenarios
- Mapping between framework settings and VRAM estimation parameters

**Stack coverage:**

- ✅ **Static JSON data** (already established pattern) - same as models.json and gpus.json
- ✅ **Zod schemas** (already in stack) - validate preset configurations
- ✅ **TypeScript interfaces** (already established) - type-safe preset objects

**Preset data structure:**

```typescript
// data/framework-presets.json
{
  "deepspeed-zero2": {
    "name": "DeepSpeed ZeRO Stage 2",
    "framework": "deepspeed",
    "config": {
      "optimizer_offload": false,
      "param_offload": false,
      "gradient_accumulation_steps": 4,
      "gradient_checkpointing": true
    }
  },
  "vllm-default": {
    "name": "vLLM Default",
    "framework": "vllm",
    "config": {
      "gpu_memory_utilization": 0.9,
      "max_num_seqs": 256,
      "swap_space": 4
    }
  }
}
```

**Confidence:** HIGH - Framework configurations are well-documented static data.

---

### Feature 4: UI Extensions for Fine-Tuning

**What's needed:**

- Input components for training parameters (batch size, gradient accumulation, optimizer)
- Framework preset selector (dropdown with presets)
- Memory breakdown visualization (stacked bar chart showing model + gradients + optimizer + activations)
- Training configuration panel (LoRA rank, QLoRA settings)

**Stack coverage:**

- ✅ **React 19** (already in stack) - functional components with hooks
- ✅ **Headless UI** (already in stack) - dropdown, toggle, radio group components
- ✅ **Tailwind CSS v4** (already in stack) - styling for new components
- ✅ **Recharts** (already in stack) - stacked bar chart for memory breakdown
- ✅ **Zustand** (already in stack) - extend store with training state slice

**Confidence:** HIGH - All UI primitives already available.

---

## Current Stack Verification

### Dependencies (Verified Current - 2026-02-10)

From existing `package.json`:

| Library | Current Version | Status | Notes |
|---------|----------------|--------|-------|
| **decimal.js** | 10.4.3 | ✅ Keep | Perfect for VRAM calculations, no alternatives needed |
| **zod** | 3.25.76 | ✅ Keep | Extend schemas for training parameters |
| **react** | 19.2.0 | ✅ Keep | Latest stable, no changes needed |
| **zustand** | 5.0.9 | ✅ Keep | Add training state slice |
| **recharts** | 3.6.0 | ✅ Keep | Stacked bar charts for memory breakdown |
| **@headlessui/react** | 2.2.9 | ✅ Keep | Dropdown for framework presets |
| **@heroicons/react** | 2.2.0 | ✅ Keep | Icons for training UI |
| **tailwindcss** | 4.1.18 | ✅ Keep | Styling for new components |
| **lz-string** | 1.5.0 | ✅ Keep | URL hash compression (reuse for sharing training configs) |
| **sonner** | 2.0.7 | ✅ Keep | Toasts for training errors/warnings |

**All libraries are current and sufficient.**

---

## What NOT to Add

Based on research of similar projects and ecosystems:

### ❌ mathjs (13.x)

**Why avoid:**

- Adds 500KB for features we don't need
- Decimal.js handles all required precision arithmetic
- Overkill for simple training memory formulas

**When to reconsider:** If adding complex statistical analysis (e.g., memory efficiency distributions, performance modeling)

---

### ❌ Specialized VRAM Calculation Libraries

**Why avoid:**

- No NPM libraries exist for fine-tuning VRAM estimation
- Existing tools are Python scripts or standalone calculators
- Hand-rolled formulas are standard (well-documented in research)

**Research findings:**

- [Modal blog](https://modal.com/blog/how-much-vram-need-fine-tuning) - formulas, not libraries
- [GitHub gists](https://gist.github.com/lapp0/d28931ebc9f59838800faa7c73e3a0dc) - Python scripts, not reusable libraries
- [Hamel's blog](https://hamel.dev/notes/llm/finetuning/estimating_vram.html) - documented formulas

**Confidence:** HIGH - We implement formulas directly in engines/fine-tuning.ts

---

### ❌ Alternative Chart Libraries (Visx, Chart.js, D3)

**Why avoid:**

- Recharts handles stacked bar charts perfectly
- Already in stack and working
- Stacked bar chart is ideal for memory breakdown (model + gradients + optimizer + activations)

**Research findings:**

- [Top React Chart Libraries 2026](https://aglowiditsolutions.com/blog/react-chart-libraries/) - Recharts still recommended
- [Syncfusion blog](https://www.syncfusion.com/blogs/post/top-5-react-chart-libraries) - Recharts in top 5

**When to reconsider:** If adding complex training visualizations (e.g., memory over time, multi-scenario comparison graphs)

---

### ❌ Alternative Precision Libraries (big.js, bignumber.js, decimalish)

**Why avoid:**

- Decimal.js already in stack and working
- No performance issues with current calculations
- Switching adds risk without benefit

**Research findings:**

- [Decimal.js vs BigNumber.js](https://medium.com/@josephgathumbi/decimal-js-vs-c1471b362181) - both are excellent, no clear winner
- [npm-compare](https://npm-compare.com/big.js,bignumber.js,decimal.js,decimal.js-light) - Decimal.js most feature-complete

**Confidence:** HIGH - Decimal.js is the right choice, keep it

---

## Integration Points: New Features with Existing Stack

### 1. Calculation Engines

**Pattern:** Same as existing engines (inference.ts, kv-cache.ts, multi-gpu.ts)

```typescript
// engines/fine-tuning.ts (NEW)
import Decimal from 'decimal.js'
import type { FineTuningParams, FineTuningResult } from './types'

export function calculateFineTuningVRAM(
  params: FineTuningParams
): FineTuningResult {
  // Pure function - no side effects
  const modelVRAM = calculateModelWeights(params)
  const gradientsVRAM = calculateGradients(params)
  const optimizerVRAM = calculateOptimizerStates(params)
  const activationsVRAM = calculateActivations(params)

  return {
    modelVRAM,
    gradientsVRAM,
    optimizerVRAM,
    activationsVRAM,
    totalVRAM: new Decimal(modelVRAM)
      .plus(gradientsVRAM)
      .plus(optimizerVRAM)
      .plus(activationsVRAM)
      .toNumber()
  }
}
```

**Confidence:** HIGH - Follows established pattern

---

### 2. Zod Schemas

**Pattern:** Extend existing schemas in utils/schemas.ts

```typescript
// utils/schemas.ts (EXTEND)
export const FineTuningConfigSchema = z.object({
  method: z.enum(['full', 'lora', 'qlora']),
  optimizer: z.enum(['adamw', 'sgd', 'adamw_8bit']),
  batch_size: z.number().min(1).max(1024),
  gradient_accumulation_steps: z.number().min(1).max(128),
  gradient_checkpointing: z.boolean(),
  lora_rank: z.number().min(1).max(256).optional(),
  lora_alpha: z.number().min(1).max(512).optional(),
})

export type FineTuningConfig = z.infer<typeof FineTuningConfigSchema>
```

**Confidence:** HIGH - Zod already used for validation

---

### 3. Static Data

**Pattern:** Same as models.json and gpus.json

```typescript
// data/framework-presets.json (NEW)
[
  {
    "id": "deepspeed-zero1",
    "name": "DeepSpeed ZeRO Stage 1",
    "framework": "deepspeed",
    "description": "Optimizer state partitioning",
    "config": {
      "zero_stage": 1,
      "gradient_accumulation_steps": 4,
      "gradient_checkpointing": true,
      "optimizer_offload": false
    }
  },
  {
    "id": "vllm-inference",
    "name": "vLLM Inference",
    "framework": "vllm",
    "description": "Optimized inference with PagedAttention",
    "config": {
      "gpu_memory_utilization": 0.9,
      "max_num_seqs": 256,
      "swap_space": 4
    }
  }
]
```

**Confidence:** HIGH - Follows established data pattern

---

### 4. Zustand Store

**Pattern:** Add training slice to existing store

```typescript
// store/slices/trainingSlice.ts (NEW)
export const createTrainingSlice = (set, get) => ({
  // Training configuration
  trainingEnabled: false,
  fineTuningMethod: 'lora' as const,
  optimizer: 'adamw' as const,
  batchSize: 4,
  gradientAccumulationSteps: 4,
  gradientCheckpointing: true,
  loraRank: 16,
  loraAlpha: 32,

  // Framework preset
  selectedPreset: null,

  // Actions
  setTrainingEnabled: (enabled) => set({ trainingEnabled: enabled }),
  setFineTuningMethod: (method) => set({ fineTuningMethod: method }),
  setOptimizer: (optimizer) => set({ optimizer }),
  setBatchSize: (size) => set({ batchSize: size }),
  applyPreset: (preset) => set({
    selectedPreset: preset,
    gradientAccumulationSteps: preset.config.gradient_accumulation_steps,
    gradientCheckpointing: preset.config.gradient_checkpointing,
  }),
})
```

**Confidence:** HIGH - Extends existing Zustand pattern

---

### 5. UI Components

**Pattern:** Same structure as existing components

```typescript
// components/inputs/TrainingConfigPanel.tsx (NEW)
export function TrainingConfigPanel() {
  const {
    trainingEnabled,
    setTrainingEnabled,
    fineTuningMethod,
    setFineTuningMethod,
  } = useCalculatorStore()

  return (
    <div className="space-y-4">
      <Toggle
        enabled={trainingEnabled}
        onChange={setTrainingEnabled}
        label="Enable Fine-Tuning Estimation"
      />

      {trainingEnabled && (
        <>
          <RadioGroup
            value={fineTuningMethod}
            onChange={setFineTuningMethod}
            options={[
              { value: 'full', label: 'Full Fine-Tuning' },
              { value: 'lora', label: 'LoRA' },
              { value: 'qlora', label: 'QLoRA' },
            ]}
          />
          {/* More inputs... */}
        </>
      )}
    </div>
  )
}
```

**Confidence:** HIGH - Uses existing Headless UI components

---

### 6. Visualization

**Pattern:** Extend Recharts usage for memory breakdown

```typescript
// components/outputs/TrainingMemoryBreakdown.tsx (NEW)
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts'

export function TrainingMemoryBreakdown({ results }) {
  const data = [
    {
      name: 'Training Memory',
      'Model Weights': results.modelVRAM,
      'Gradients': results.gradientsVRAM,
      'Optimizer States': results.optimizerVRAM,
      'Activations': results.activationsVRAM,
    }
  ]

  return (
    <BarChart data={data} width={600} height={300}>
      <XAxis dataKey="name" />
      <YAxis label={{ value: 'VRAM (GB)', angle: -90 }} />
      <Tooltip />
      <Legend />
      <Bar dataKey="Model Weights" stackId="a" fill="#3b82f6" />
      <Bar dataKey="Gradients" stackId="a" fill="#8b5cf6" />
      <Bar dataKey="Optimizer States" stackId="a" fill="#ec4899" />
      <Bar dataKey="Activations" stackId="a" fill="#f59e0b" />
    </BarChart>
  )
}
```

**Confidence:** HIGH - Recharts handles stacked bar charts natively

---

## Domain-Specific Formulas (No Libraries Needed)

Based on research, these are the standard formulas to implement:

### Full Fine-Tuning Memory

```
Total = Model + Gradients + Optimizer + Activations + Overhead

Model weights = params × bytes_per_param
Gradients = params × 4 (FP32)
Optimizer states (AdamW) = params × 8 (2 × FP32 moments)
Optimizer states (AdamW 8-bit) = params × 2 (2 × INT8 moments)
Optimizer states (SGD) = params × 4 (1 × FP32 momentum)
Activations = batch_size × seq_len × hidden_size × num_layers × multiplier
  where multiplier ≈ 10-12x (attention + MLP + residuals)

With gradient checkpointing: activations ÷ sqrt(num_layers)
With gradient accumulation: activations ÷ gradient_accumulation_steps
```

**Sources:**

- [Modal: How much VRAM do I need for LLM model fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [Hamel's Blog: Estimating vRAM](https://hamel.dev/notes/llm/finetuning/estimating_vram.html)

---

### LoRA Fine-Tuning Memory

```
Total = Base Model + LoRA Adapters + Optimizer (adapters only) + Activations

Base model = params × bytes_per_param (can be quantized)
LoRA adapters = 2 × rank × hidden_size × num_layers × bytes_per_param
  (A and B matrices per layer)
Gradients (adapters only) = adapter_params × 4
Optimizer states (adapters only) = adapter_params × 8 (AdamW)
Activations = same as full fine-tuning

Typical adapter params ≈ 0.5-2% of base model params
```

**Sources:**

- [Modal: LoRA vs. QLoRA](https://modal.com/blog/lora-qlora)
- [HuggingFace Forums: How to calculate memory for LoRA](https://discuss.huggingface.co/t/how-to-calculate-the-memory-required-using-lora-fine-tuning/63049)

---

### QLoRA Fine-Tuning Memory

```
Total = Quantized Base + LoRA Adapters + Optimizer (adapters only) + Activations

Quantized base = params × 0.5 (4-bit NF4)
LoRA adapters = 2 × rank × hidden_size × num_layers × 2 (FP16)
Optimizer states (adapters only) = adapter_params × 8 (AdamW)
  with paged optimizer (CPU offload during spikes)
Activations = same as full fine-tuning

QLoRA adds paged optimizer: automatically pages optimizer states to CPU RAM
during memory spikes, then pages back to GPU when needed.
```

**Sources:**

- [Manalelaidouni: QLoRA 4-Bit Quantization](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)
- [RunPod: Fine-tune on a budget using LoRA and QLoRA](https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget)

---

### Gradient Accumulation

```
Effective batch size = micro_batch_size × gradient_accumulation_steps × num_gpus

Memory impact:
- Model weights: no change
- Gradients: no change (accumulated in-place)
- Optimizer states: no change
- Activations: reduced by 1/gradient_accumulation_steps
  (only micro_batch_size activations stored at once)

Throughput impact:
- Forward passes: gradient_accumulation_steps × micro_batch_time
- Backward passes: gradient_accumulation_steps × micro_batch_time
- Optimizer step: 1 × optimizer_step_time (not multiplied)
```

**Sources:**

- [HuggingFace: Batch size vs gradient accumulation](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260)
- [Axolotl Docs: Batch vs Grad](https://docs.axolotl.ai/docs/batch_vs_grad.html)
- [Unsloth Blog: Gradient Accumulation Bug Fixes](https://unsloth.ai/blog/gradient)

---

## Framework Presets: Research Findings

### DeepSpeed ZeRO

**Configuration structure:** [DeepSpeed Config JSON](https://www.deepspeed.ai/docs/config-json/)

**Key settings for VRAM estimation:**

- `zero_optimization.stage` (0/1/2/3) - determines what gets partitioned
- `zero_optimization.offload_optimizer` - CPU offload
- `zero_optimization.offload_param` - CPU offload (stage 3 only)
- `gradient_accumulation_steps`
- `gradient_checkpointing`

**Memory impact:**

- Stage 0: No partitioning (baseline)
- Stage 1: Optimizer state partitioning (÷ num_gpus)
- Stage 2: Gradient + optimizer partitioning (÷ num_gpus)
- Stage 3: Parameter + gradient + optimizer partitioning (÷ num_gpus)

---

### vLLM

**Configuration structure:** [vLLM Engine Arguments](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/vllm-engine-arguments)

**Key settings:**

- `gpu_memory_utilization` (default 0.9) - VRAM percentage to use
- `max_num_seqs` - max concurrent sequences
- `swap_space` - CPU swap space in GB
- `block_size` - KV cache block size (default 16)

**Memory impact:**

- KV cache uses `gpu_memory_utilization × total_vram - model_vram`
- PagedAttention reduces fragmentation (10-20% more efficient)

---

### Unsloth

**Configuration pattern:** [Fine-tuning Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)

**Key settings:**

- `max_seq_length` - sequence length (impacts KV cache)
- `load_in_4bit` - QLoRA mode
- `gradient_checkpointing` - activation recomputation
- `per_device_train_batch_size`
- `gradient_accumulation_steps`

**Memory optimizations:**

- 2x faster training than standard
- 70% less VRAM through kernel optimizations
- Compatible with vLLM for deployment

---

### TGI (Text Generation Inference)

**Configuration pattern:** Inference-focused (not training)

**Key settings:**

- `quantize` - quantization method (bitsandbytes, gptq, awq)
- `max_concurrent_requests`
- `max_batch_size`
- `max_input_length` / `max_total_tokens`

**Not applicable for training VRAM estimation** - TGI is inference-only

---

## Recommended Framework Presets (Static Data)

Create `data/framework-presets.json` with these presets:

```json
[
  {
    "id": "deepspeed-zero1",
    "name": "DeepSpeed ZeRO Stage 1",
    "category": "training",
    "framework": "deepspeed",
    "description": "Optimizer state partitioning across GPUs",
    "vram_modifier": {
      "optimizer_states_divisor": "num_gpus"
    },
    "config": {
      "zero_stage": 1,
      "gradient_accumulation_steps": 4,
      "gradient_checkpointing": true,
      "optimizer_offload": false
    }
  },
  {
    "id": "deepspeed-zero2",
    "name": "DeepSpeed ZeRO Stage 2",
    "category": "training",
    "framework": "deepspeed",
    "description": "Gradient + optimizer partitioning",
    "vram_modifier": {
      "gradients_divisor": "num_gpus",
      "optimizer_states_divisor": "num_gpus"
    },
    "config": {
      "zero_stage": 2,
      "gradient_accumulation_steps": 4,
      "gradient_checkpointing": true,
      "optimizer_offload": false
    }
  },
  {
    "id": "deepspeed-zero3",
    "name": "DeepSpeed ZeRO Stage 3",
    "category": "training",
    "framework": "deepspeed",
    "description": "Full parameter + gradient + optimizer partitioning",
    "vram_modifier": {
      "model_divisor": "num_gpus",
      "gradients_divisor": "num_gpus",
      "optimizer_states_divisor": "num_gpus"
    },
    "config": {
      "zero_stage": 3,
      "gradient_accumulation_steps": 8,
      "gradient_checkpointing": true,
      "optimizer_offload": true
    }
  },
  {
    "id": "vllm-default",
    "name": "vLLM Default",
    "category": "inference",
    "framework": "vllm",
    "description": "Optimized inference with PagedAttention",
    "config": {
      "gpu_memory_utilization": 0.9,
      "max_num_seqs": 256,
      "swap_space": 4,
      "block_size": 16
    }
  },
  {
    "id": "unsloth-qlora",
    "name": "Unsloth QLoRA",
    "category": "training",
    "framework": "unsloth",
    "description": "Memory-efficient QLoRA fine-tuning",
    "config": {
      "load_in_4bit": true,
      "gradient_checkpointing": true,
      "per_device_train_batch_size": 2,
      "gradient_accumulation_steps": 4,
      "lora_r": 16,
      "lora_alpha": 16
    }
  }
]
```

**Confidence:** HIGH - Based on official documentation

---

## Installation (No Changes Needed)

All required libraries are already installed. No new dependencies to add.

**Current package.json is sufficient:**

```json
{
  "dependencies": {
    "decimal.js": "^10.4.3",
    "zod": "^3.25.76",
    "react": "^19.2.0",
    "zustand": "^5.0.9",
    "recharts": "^3.6.0",
    "@headlessui/react": "^2.2.9",
    "tailwindcss": "^4.1.18"
  }
}
```

---

## Implementation Roadmap Implications

### Phase Structure Recommendation

**Phase 1: Fine-Tuning Calculation Engine** (1-2 days)

- Create `engines/fine-tuning.ts` with pure calculation functions
- Implement formulas for full/LoRA/QLoRA
- Add comprehensive tests (follow existing test patterns)
- **Dependencies:** None (pure functions)
- **Risk:** Low (well-documented formulas)

**Phase 2: Training State & Schemas** (1 day)

- Extend Zod schemas for training parameters
- Add training slice to Zustand store
- Create framework presets JSON
- **Dependencies:** Phase 1 (types from engines)
- **Risk:** Low (follows established patterns)

**Phase 3: Training UI Components** (2-3 days)

- Create TrainingConfigPanel component
- Create FrameworkPresetSelector component
- Create TrainingMemoryBreakdown visualization
- **Dependencies:** Phase 2 (state, schemas)
- **Risk:** Low (reuses existing UI components)

**Phase 4: Integration & Polish** (1-2 days)

- Integrate training calculation into main calculator flow
- Add training mode toggle
- Update URL hash to include training config
- Add error handling and validation
- **Dependencies:** Phase 1-3
- **Risk:** Low (integration layer)

**Total estimate:** 5-8 days for full implementation

---

## Confidence Assessment

| Area | Confidence | Rationale |
|------|-----------|-----------|
| **No new libraries needed** | HIGH | Research confirms existing stack is sufficient |
| **Decimal.js for calculations** | HIGH | Already validated for precision arithmetic |
| **Zod for validation** | HIGH | Already used extensively in codebase |
| **Recharts for visualization** | HIGH | Stacked bar charts handle memory breakdown perfectly |
| **Framework preset data** | HIGH | Well-documented configurations from official sources |
| **Formulas for fine-tuning** | HIGH | Standard formulas from authoritative sources |
| **Integration patterns** | HIGH | Following established codebase patterns |

---

## Risks & Mitigation

### Risk 1: Formula Accuracy

**Risk:** Fine-tuning memory formulas might not match real-world usage
**Mitigation:**

- Add 20% overhead to all calculations (standard practice)
- Include references to source formulas in code comments
- Add "Experimental" badge to training estimates in UI
- Plan validation against real training runs in later phase

### Risk 2: Framework Preset Complexity

**Risk:** Framework configurations are complex, presets might miss edge cases
**Mitigation:**

- Start with common presets only (5-10 presets)
- Add "Custom Configuration" option for advanced users
- Link to official framework documentation for each preset
- Allow users to modify preset values

### Risk 3: UI Complexity

**Risk:** Adding training UI might clutter calculator interface
**Mitigation:**

- Use collapsible sections (already established pattern)
- Add training mode toggle (opt-in)
- Keep training config separate from inference config
- Progressive disclosure (show advanced options only when needed)

---

## Sources

### Fine-Tuning VRAM Formulas

- [Modal: How much VRAM do I need for LLM model fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [Hamel's Blog: Estimating vRAM](https://hamel.dev/notes/llm/finetuning/estimating_vram.html)
- [GitHub Gist: LLM Memory Requirement Calculator](https://gist.github.com/lapp0/d28931ebc9f59838800faa7c73e3a0dc)
- [GitHub Gist: Calculating vRAM requirements for LLMs](https://gist.github.com/RahulSChand/4bc83d1529afc99be14d2a2a54b8e968)

### LoRA & QLoRA

- [Modal: LoRA vs. QLoRA](https://modal.com/blog/lora-qlora)
- [Manalelaidouni: QLoRA 4-Bit Quantization](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)
- [RunPod: Fine-tune LLMs on a budget using LoRA and QLoRA](https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget)
- [HuggingFace Forums: How to calculate memory for LoRA fine-tuning](https://discuss.huggingface.co/t/how-to-calculate-the-memory-required-using-lora-fine-tuning/63049)

### Gradient Accumulation

- [HuggingFace Forums: Batch size vs gradient accumulation](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260)
- [Axolotl Docs: Batch vs Grad](https://docs.axolotl.ai/docs/batch_vs_grad.html)
- [Unsloth Blog: Gradient Accumulation Bug Fixes](https://unsloth.ai/blog/gradient)
- [HuggingFace Docs: Gradient Accumulation with Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation)

### Framework Configuration

- [DeepSpeed: Configuration JSON](https://www.deepspeed.ai/docs/config-json/)
- [Unsloth Docs: vLLM Engine Arguments](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/vllm-engine-arguments)
- [Unsloth Docs: Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)

### React Libraries

- [Top React Chart Libraries 2026](https://aglowiditsolutions.com/blog/react-chart-libraries/)
- [Syncfusion: Top 5 React Chart Libraries 2026](https://www.syncfusion.com/blogs/post/top-5-react-chart-libraries)

### Precision Arithmetic

- [Decimal.js vs BigNumber.js](https://medium.com/@josephgathumbi/decimal-js-vs-c1471b362181)
- [npm-compare: JavaScript Arbitrary-Precision Libraries](https://npm-compare.com/big.js,bignumber.js,decimal.js,decimal.js-light)

---

## Conclusion

**The existing stack requires ZERO additions** for fine-tuning features. All capabilities are already present:

✅ **Calculation logic** → Decimal.js (precision arithmetic)
✅ **Validation** → Zod (schemas for training parameters)
✅ **UI components** → React + Headless UI + Tailwind CSS
✅ **Visualization** → Recharts (stacked bar charts)
✅ **State management** → Zustand (add training slice)
✅ **Static data** → JSON (framework presets)

**Implementation is pure extension work:**

1. New calculation engine (engines/fine-tuning.ts)
2. New Zod schemas (extend utils/schemas.ts)
3. New UI components (components/inputs/TrainingConfigPanel.tsx)
4. New static data (data/framework-presets.json)
5. New store slice (store/slices/trainingSlice.ts)

**All follow established patterns from v1.0.**

**Confidence:** HIGH - Research validates that no new libraries are needed, formulas are well-documented, and existing stack is sufficient.
