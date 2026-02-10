# Architecture Integration: Fine-Tuning Features

**Project:** LLM VRAM Calculator
**Milestone:** Fine-tuning VRAM estimation, gradient accumulation, framework presets
**Researched:** 2026-02-10
**Overall Confidence:** HIGH (official documentation verified)

## Executive Summary

The fine-tuning milestone extends the existing inference-focused architecture with training workload support. The architecture follows established patterns: **pure calculation engines** (Decimal.js), **Zod schemas for type safety**, **Zustand for state**, and **React components for UI**. New features integrate cleanly by adding parallel engines, extending the store, and creating new input/output components that reuse existing patterns.

**Key architectural principles maintained:**

- Engines remain pure functions (no React/DOM dependencies)
- All data validated through Zod schemas at boundaries
- Store drives UI state
- Components split by concern (inputs/, outputs/)

**Integration complexity:** MEDIUM - New calculation domains but familiar patterns

---

## Recommended Architecture

### Extended Directory Structure

```
src/
  engines/                    # Pure calculation logic (NEW + MODIFIED)
    inference.ts              # [EXISTING] Inference VRAM calculation
    finetuning.ts             # [NEW] Fine-tuning VRAM calculation (full/LoRA/QLoRA)
    optimizer.ts              # [NEW] Optimizer state memory (AdamW, Adafactor, 8-bit variants)
    lora.ts                   # [NEW] LoRA adapter parameter calculation
    gradient-accumulation.ts  # [NEW] Effective batch size, memory tradeoffs
    framework-presets.ts      # [NEW] vLLM/TGI/Unsloth/DeepSpeed optimization profiles
    quantization.ts           # [EXISTING] - Reused for QLoRA base model quantization
    types.ts                  # [MODIFIED] Add training types, optimizer types, framework types

  components/                 # React UI (NEW + MODIFIED)
    inputs/
      TrainingModeSelector.tsx       # [NEW] Full FT / LoRA / QLoRA / Inference toggle
      OptimizerSelector.tsx          # [NEW] AdamW, Adafactor, SGD, 8-bit variants
      LoRAConfigPanel.tsx            # [NEW] Rank, alpha, target modules, dropout
      GradientAccumulationInput.tsx  # [NEW] Steps, effective batch size display
      FrameworkPresetSelector.tsx    # [NEW] vLLM, TGI, Unsloth, DeepSpeed presets
      ModelSelector.tsx              # [EXISTING] Reused for training
      QuantizationPicker.tsx         # [EXISTING] Reused for QLoRA base quantization

    outputs/
      VRAMBreakdownChart.tsx         # [MODIFIED] Extend for training components
      MemoryBreakdownTable.tsx       # [MODIFIED] Add training mode rows
      FrameworkOptimizationBadge.tsx # [NEW] Show active optimizations from preset

  store/
    uiStore.ts                # [MODIFIED] Add training state
    comparisonStore.ts        # [MODIFIED] Add training results to snapshot

  data/
    models.json               # [EXISTING] Reused
    gpus.json                 # [EXISTING] Reused
    optimizers.json           # [NEW] Optimizer configs (memory multipliers, stability notes)
    framework-presets.json    # [NEW] Framework optimization profiles

  utils/
    schemas.ts                # [MODIFIED] Add training schemas, optimizer schemas
```

---

## Component Architecture

### 1. New Engines (Pure Functions)

#### `src/engines/finetuning.ts`

**Purpose:** Calculate total VRAM for fine-tuning workloads (full, LoRA, QLoRA)

**Interface:**

```typescript
export interface FineTuningVRAMBreakdown {
  // Base model components (reused from inference)
  modelWeights: Decimal       // Base model in training precision (or quantized for QLoRA)

  // Training-specific components
  gradients: Decimal          // Gradient storage (trainable params only for LoRA/QLoRA)
  optimizerStates: Decimal    // Optimizer momentum/variance (trainable params only)
  activations: Decimal        // Forward pass activations (larger than inference)
  loraAdapters: Decimal       // LoRA A/B matrices (0 for full fine-tuning)

  // Infrastructure
  frameworkOverhead: Decimal  // PyTorch + CUDA + training framework
  kvCache: Decimal            // KV cache for training (if applicable)

  // Total
  total: Decimal

  // Metadata
  trainableParams: number     // Number of trainable parameters
  frozenParams: number        // Number of frozen parameters
}

export type TrainingMode = 'full' | 'lora' | 'qlora'

export function calculateFineTuningVRAM(params: {
  model: Model
  mode: TrainingMode
  precision: QuantizationFormat    // Training precision (fp32, bf16, fp16)
  basePrecision?: QuantizationFormat // QLoRA only: base model quantization (nf4, int4)
  loraConfig?: LoRAConfig
  optimizerConfig: OptimizerConfig
  sequenceLength: number
  batchSize: number
  gradientAccumulation?: number
  gradientCheckpointing?: boolean
}): FineTuningVRAMBreakdown
```

**Key formulas:**

```typescript
// Full Fine-Tuning (mixed precision fp16/bf16 training)
modelWeights = params * 2        // fp16/bf16 weights
gradients = params * 4           // fp32 gradients
optimizerStates = params * K     // K = 8 for AdamW, 2 for Adafactor
activations = batchSize * seqLen * hiddenSize * layers * 12 // MLP expansion
total = modelWeights + gradients + optimizerStates + activations + overhead

// LoRA Fine-Tuning
baseModel = params * 2           // fp16 frozen weights
loraParams = rank * hiddenSize * targetLayers * 2 // A and B matrices
gradients = loraParams * 4       // Only for LoRA adapters
optimizerStates = loraParams * K // Only for LoRA adapters (~1% of full)
activations = same as full       // Full forward pass still needed
total = baseModel + loraParams + gradients + optimizerStates + activations + overhead

// QLoRA Fine-Tuning
baseModel = params * 0.5625      // 4-bit NF4 with double quantization overhead
loraParams = rank * hiddenSize * targetLayers * 2
gradients = loraParams * 4       // Only for adapters
optimizerStates = loraParams * K // Only for adapters
activations = same as full
total = baseModel + loraParams + gradients + optimizerStates + activations + overhead
```

**Dependencies:**

- `calculateModelWeightVRAM` from `quantization.ts`
- `calculateLoRAParams` from `lora.ts`
- `calculateOptimizerMemory` from `optimizer.ts`
- `calculateActivationMemory` from `inference.ts` (extended for training)

**Testing:**

- Validate against HuggingFace transformers reported memory
- Test cases: Llama 7B full FT, Llama 7B LoRA r=16, Llama 70B QLoRA r=64
- Error tolerance: <15% vs measured values

**Sources:**

- [HuggingFace Model Memory Anatomy](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

---

#### `src/engines/lora.ts`

**Purpose:** Calculate LoRA adapter parameters and memory

**Interface:**

```typescript
export interface LoRAConfig {
  rank: number                    // r: rank of low-rank matrices (4, 8, 16, 32, 64, 128)
  alpha: number                   // LoRA alpha scaling factor (usually rank or 2*rank)
  targetModules: LoRATargetModule[]  // Which modules to apply LoRA
  dropout: number                 // Dropout probability (0.0 - 0.1)
}

export type LoRATargetModule = 'q_proj' | 'k_proj' | 'v_proj' | 'o_proj' | 'gate_proj' | 'up_proj' | 'down_proj'

export interface LoRAParamBreakdown {
  perLayerParams: number          // Parameters per transformer layer
  totalLayers: number             // Number of layers with LoRA
  totalParams: number             // Total LoRA parameters
  percentOfBase: number           // Percentage of base model parameters
  memoryGB: Decimal               // Memory in GB (fp16)
}

export function calculateLoRAParams(
  model: Model,
  config: LoRAConfig,
): LoRAParamBreakdown

export const LORA_PRESETS: Record<string, LoRAConfig> = {
  minimal: { rank: 4, alpha: 8, targetModules: ['q_proj', 'v_proj'], dropout: 0.05 },
  standard: { rank: 16, alpha: 32, targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], dropout: 0.05 },
  full: { rank: 64, alpha: 128, targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], dropout: 0.1 },
}
```

**Key formulas:**

```typescript
// LoRA adds two low-rank matrices A (d × r) and B (r × d) per target module
// Update: ΔW = B × A (where W is d × d original weight)

paramsPerModule = 2 * hiddenSize * rank  // A and B matrices

// For attention projections (q, k, v, o): hiddenSize → hiddenSize
attentionModules = config.targetModules.filter(m => m.endsWith('_proj')).length
attentionParams = attentionModules * 2 * hiddenSize * rank * numLayers

// For MLP projections (gate, up, down): hiddenSize → intermediateSize or reverse
// gate_proj, up_proj: hiddenSize → intermediateSize
// down_proj: intermediateSize → hiddenSize
mlpParams = ...similar calculation...

totalParams = attentionParams + mlpParams
percentOfBase = (totalParams / baseModelParams) * 100  // Usually 0.1% - 2%
memoryGB = totalParams * 2 / (1024^3)  // fp16 storage
```

**Sources:**

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library Documentation](https://huggingface.co/docs/peft)

---

#### `src/engines/optimizer.ts`

**Purpose:** Calculate optimizer state memory requirements

**Interface:**

```typescript
export type OptimizerType =
  | 'adamw'           // Standard AdamW (fp32 states)
  | 'adamw_8bit'      // 8-bit AdamW (bitsandbytes)
  | 'sgd'             // SGD with momentum
  | 'adafactor'       // Memory-efficient Adafactor

export interface OptimizerConfig {
  type: OptimizerType
  learningRate?: number           // For reference only (doesn't affect memory)
  weightDecay?: number            // For reference only
}

export interface OptimizerMemoryBreakdown {
  type: OptimizerType
  bytesPerParam: number           // Memory bytes per trainable parameter
  totalMemoryGB: Decimal          // Total optimizer state memory
  notes: string                   // Stability/performance notes
}

export function calculateOptimizerMemory(
  trainableParams: number,
  config: OptimizerConfig,
): OptimizerMemoryBreakdown

export const OPTIMIZER_SPECS: Record<OptimizerType, {
  bytesPerParam: number
  stability: 'high' | 'medium' | 'low'
  notes: string
}> = {
  adamw: {
    bytesPerParam: 8,  // 4 bytes momentum + 4 bytes variance (fp32)
    stability: 'high',
    notes: 'Standard choice for fine-tuning. Stable convergence.',
  },
  adamw_8bit: {
    bytesPerParam: 2,  // 1 byte momentum + 1 byte variance (int8)
    stability: 'high',
    notes: '4x memory reduction vs AdamW. Same performance (bitsandbytes).',
  },
  adafactor: {
    bytesPerParam: 4,  // Factored representation reduces from 8 to ~4
    stability: 'medium',
    notes: 'Memory-efficient but less stable. May need tuning.',
  },
  sgd: {
    bytesPerParam: 4,  // 4 bytes momentum (fp32)
    stability: 'low',
    notes: 'Minimal memory but poor convergence for LLMs.',
  },
}
```

**Key formulas:**

```typescript
// AdamW (standard): stores 2 states per parameter
// - First moment (momentum): fp32 (4 bytes)
// - Second moment (variance): fp32 (4 bytes)
optimizerMemory = trainableParams * 8 / (1024^3)  // GB

// AdamW 8-bit: quantized states
// - First moment: int8 (1 byte)
// - Second moment: int8 (1 byte)
optimizerMemory = trainableParams * 2 / (1024^3)  // GB

// Adafactor: factored representation
// For matrix-shaped params (n × m), stores O(n + m) instead of O(n*m)
// Approximation: ~50% of AdamW memory
optimizerMemory = trainableParams * 4 / (1024^3)  // GB
```

**Sources:**

- [DeepSpeed ZeRO Memory Requirements](https://deepspeed.readthedocs.io/en/latest/memory.html)
- [HuggingFace Optimizers Documentation](https://huggingface.co/docs/transformers/en/optimizers)
- [bitsandbytes 8-bit Adam](https://github.com/TimDettmers/bitsandbytes)

---

#### `src/engines/gradient-accumulation.ts`

**Purpose:** Calculate effective batch size and memory tradeoffs

**Interface:**

```typescript
export interface GradientAccumulationConfig {
  microBatchSize: number          // Actual batch size per step
  accumulationSteps: number       // Number of steps to accumulate
  effectiveBatchSize: number      // microBatchSize * accumulationSteps
}

export interface GradientAccumulationImpact {
  memoryReduction: number         // % reduction vs effective batch size
  timeIncrease: number            // % increase in training time
  recommendationLevel: 'optimal' | 'acceptable' | 'inefficient'
  warnings: string[]
}

export function calculateGradientAccumulationImpact(
  config: GradientAccumulationConfig,
): GradientAccumulationImpact

export function recommendGradientAccumulation(
  desiredBatchSize: number,
  availableVRAM: number,
  requiredVRAMPerSample: Decimal,
): GradientAccumulationConfig
```

**Key formulas:**

```typescript
// Memory savings primarily from activation memory
// Model weights, gradients, optimizer states remain constant regardless of batch size
// KV cache and activations scale with batch size

// Memory with gradient accumulation
memoryWithGA = modelWeights + gradients + optimizerStates +
               (activations * microBatchSize) + (kvCache * microBatchSize)

// Memory without gradient accumulation (full effective batch)
memoryWithoutGA = modelWeights + gradients + optimizerStates +
                  (activations * effectiveBatchSize) + (kvCache * effectiveBatchSize)

// Memory reduction (activations and KV cache only)
reductionFactor = (effectiveBatchSize - microBatchSize) / effectiveBatchSize

// Time increase (sequential micro-batches)
timeIncrease = (accumulationSteps - 1) / accumulationSteps * 100  // Percentage
```

**Warnings:**

- Recent research (2025) shows gradient accumulation may be wasteful for small batches
- Recommend batch size 1 with proper tuning over gradient accumulation
- Only suggest GA when memory constraint is absolute blocker

**Sources:**

- [Small Batch Size Training for Language Models (2025)](https://arxiv.org/abs/2507.07101)
- [HuggingFace Gradient Accumulation Discussion](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260)

---

#### `src/engines/framework-presets.ts`

**Purpose:** Apply framework-specific optimization profiles

**Interface:**

```typescript
export type FrameworkType =
  | 'baseline'      // No framework-specific optimizations
  | 'vllm'          // vLLM PagedAttention optimizations
  | 'tgi'           // Hugging Face Text Generation Inference
  | 'unsloth'       // Unsloth fast fine-tuning
  | 'deepspeed'     // DeepSpeed ZeRO

export interface FrameworkPreset {
  framework: FrameworkType
  optimizations: string[]         // List of active optimizations
  kvCacheReduction: number        // % reduction in KV cache memory (vLLM PagedAttention)
  activationReduction: number     // % reduction in activation memory (gradient checkpointing)
  overheadIncrease: number        // Additional framework overhead (GB)
  speedupFactor: number           // Training/inference speedup multiplier
  memoryReductionFactor: number   // Overall memory reduction multiplier
  requirements: string[]          // Framework-specific requirements
  notes: string
}

export const FRAMEWORK_PRESETS: Record<FrameworkType, FrameworkPreset> = {
  vllm: {
    framework: 'vllm',
    optimizations: ['PagedAttention', 'Continuous batching', 'Optimized CUDA kernels'],
    kvCacheReduction: 96,         // 96% reduction in KV cache waste
    activationReduction: 0,       // Inference-only
    overheadIncrease: 1.5,        // vLLM runtime overhead
    speedupFactor: 2.5,           // 2-4x throughput increase
    memoryReductionFactor: 0.65,  // Effective memory reduction from paging
    requirements: ['Inference mode only', 'CUDA GPU'],
    notes: 'PagedAttention reduces KV cache waste from 60-80% to <4%',
  },
  tgi: {
    framework: 'tgi',
    optimizations: ['Flash Attention', 'Paged Attention', 'Dynamic batching'],
    kvCacheReduction: 85,
    activationReduction: 0,
    overheadIncrease: 1.0,
    speedupFactor: 2.0,
    memoryReductionFactor: 0.70,
    requirements: ['Inference mode only', 'CUDA GPU'],
    notes: 'Optimized for production inference with auto-scaling',
  },
  unsloth: {
    framework: 'unsloth',
    optimizations: ['Manual backprop kernels', 'Triton kernels', 'LoRA optimization'],
    kvCacheReduction: 0,
    activationReduction: 26,      // 74% less memory = 26% of original
    overheadIncrease: 0.5,
    speedupFactor: 2.0,
    memoryReductionFactor: 0.26,  // 74% memory reduction
    requirements: ['Training mode only', 'LoRA or QLoRA'],
    notes: '2x faster training with 70% less memory vs standard LoRA',
  },
  deepspeed: {
    framework: 'deepspeed',
    optimizations: ['ZeRO Stage 1/2/3', 'Gradient checkpointing', 'CPU offloading'],
    kvCacheReduction: 0,
    activationReduction: 70,
    overheadIncrease: 2.0,
    speedupFactor: 0.9,
    memoryReductionFactor: 0.125, // ZeRO-3: ~8x reduction
    requirements: ['Multi-GPU or CPU offloading', 'Training mode'],
    notes: 'ZeRO-3 partitions model, gradients, optimizer across GPUs/CPU',
  },
  baseline: {
    framework: 'baseline',
    optimizations: [],
    kvCacheReduction: 0,
    activationReduction: 0,
    overheadIncrease: 0,
    speedupFactor: 1.0,
    memoryReductionFactor: 1.0,
    requirements: [],
    notes: 'No framework-specific optimizations',
  },
}

export function applyFrameworkOptimizations(
  baselineBreakdown: FineTuningVRAMBreakdown | InferenceVRAMBreakdown,
  framework: FrameworkType,
): { breakdown: typeof baselineBreakdown; preset: FrameworkPreset }
```

**Sources:**

- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [HuggingFace TGI Documentation](https://huggingface.co/docs/text-generation-inference/en/index)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)

---

### 2. Extended Types (`src/engines/types.ts`)

```typescript
// Add to existing types.ts

/**
 * Training mode selection
 */
export type TrainingMode = 'inference' | 'full-ft' | 'lora' | 'qlora'

/**
 * Fine-tuning VRAM breakdown (extends inference breakdown)
 */
export interface FineTuningVRAMBreakdown extends InferenceVRAMBreakdown {
  // Additional training components
  gradients: Decimal          // Gradient storage
  optimizerStates: Decimal    // Optimizer momentum/variance
  loraAdapters: Decimal       // LoRA adapters (0 for full FT)

  // Metadata
  trainableParams: number
  frozenParams: number

  // Override total to include training components
  total: Decimal
}

/**
 * Optimizer configuration schema
 */
export const OptimizerConfigSchema = z.object({
  type: z.enum(['adamw', 'adamw_8bit', 'sgd', 'adafactor']),
  learningRate: z.number().positive().optional(),
  weightDecay: z.number().nonnegative().optional(),
})

/**
 * LoRA configuration schema
 */
export const LoRAConfigSchema = z.object({
  rank: z.number().int().positive().min(1).max(256),
  alpha: z.number().int().positive(),
  targetModules: z.array(z.enum([
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
  ])),
  dropout: z.number().min(0).max(1),
})

/**
 * Framework preset type
 */
export type FrameworkType = 'baseline' | 'vllm' | 'tgi' | 'unsloth' | 'deepspeed'
```

---

### 3. Store Extensions (`src/store/uiStore.ts`)

```typescript
// Add to UIState interface

interface UIState {
  // ... existing fields ...

  // Training configuration
  trainingMode: TrainingMode
  optimizerConfig: OptimizerConfig
  loraConfig: LoRAConfig | null
  gradientAccumulationSteps: number
  gradientCheckpointing: boolean
  frameworkPreset: FrameworkType

  // Training precision (separate from weight quantization)
  trainingPrecision: 'fp32' | 'bf16' | 'fp16'

  // Actions
  setTrainingMode: (mode: TrainingMode) => void
  setOptimizerConfig: (config: OptimizerConfig) => void
  setLoRAConfig: (config: LoRAConfig | null) => void
  setGradientAccumulationSteps: (steps: number) => void
  setGradientCheckpointing: (enabled: boolean) => void
  setFrameworkPreset: (framework: FrameworkType) => void
  setTrainingPrecision: (precision: 'fp32' | 'bf16' | 'fp16') => void
}

// Default state additions
{
  trainingMode: 'inference',
  optimizerConfig: { type: 'adamw' },
  loraConfig: null,
  gradientAccumulationSteps: 1,
  gradientCheckpointing: false,
  frameworkPreset: 'baseline',
  trainingPrecision: 'bf16',
}
```

**State persistence:** Training config should NOT be persisted (URL hash only), similar to existing pattern.

---

### 4. New Input Components

#### `components/inputs/TrainingModeSelector.tsx`

**Purpose:** Toggle between inference, full fine-tuning, LoRA, QLoRA

**UI Pattern:** Radio group or segmented control

```typescript
<RadioGroup value={trainingMode} onChange={setTrainingMode}>
  <Radio value="inference">Inference</Radio>
  <Radio value="full-ft">Full Fine-Tuning</Radio>
  <Radio value="lora">LoRA</Radio>
  <Radio value="qlora">QLoRA (4-bit)</Radio>
</RadioGroup>
```

**Interactions:**

- When switching to LoRA/QLoRA, auto-populate `loraConfig` with "standard" preset
- When switching to full-ft, clear `loraConfig`
- Show contextual help: "LoRA: 10-20% of full FT memory", "QLoRA: Fine-tune 70B on 24GB GPU"

---

#### `components/inputs/LoRAConfigPanel.tsx`

**Purpose:** Configure LoRA rank, alpha, target modules

**UI Pattern:** Collapsible panel (only shown when mode = lora or qlora)

```typescript
{(trainingMode === 'lora' || trainingMode === 'qlora') && (
  <LoRAConfigPanel>
    {/* Preset selector */}
    <select value={selectedPreset} onChange={applyPreset}>
      <option value="minimal">Minimal (r=4, Q+V only)</option>
      <option value="standard">Standard (r=16, QKVO)</option>
      <option value="full">Full (r=64, All projections)</option>
      <option value="custom">Custom</option>
    </select>

    {/* Live calculation of LoRA params */}
    <div className="text-sm text-gray-600">
      LoRA parameters: {loraParams.toLocaleString()}
      ({percentOfBase.toFixed(2)}% of base model)
    </div>
  </LoRAConfigPanel>
)}
```

---

#### `components/inputs/OptimizerSelector.tsx`

**Purpose:** Select optimizer type (AdamW, Adafactor, 8-bit variants)

**UI Pattern:** Dropdown with descriptions

```typescript
<select value={optimizerType} onChange={setOptimizerType}>
  <option value="adamw">AdamW (Standard - 8 bytes/param)</option>
  <option value="adamw_8bit">AdamW 8-bit (4x less memory)</option>
  <option value="adafactor">Adafactor (Memory efficient)</option>
  <option value="sgd">SGD (Minimal memory, poor convergence)</option>
</select>
```

---

#### `components/inputs/GradientAccumulationInput.tsx`

**Purpose:** Configure gradient accumulation steps

**UI Pattern:** Number input with live effective batch size display

```typescript
<div>
  <label>Micro-batch Size</label>
  <input type="number" value={batchSize} min={1} max={64} />

  <label>Gradient Accumulation Steps</label>
  <input type="number" value={accumulationSteps} min={1} max={64} />

  <div className="text-sm font-medium">
    Effective Batch Size: {batchSize * accumulationSteps}
  </div>
</div>
```

---

#### `components/inputs/FrameworkPresetSelector.tsx`

**Purpose:** Select framework optimizations (vLLM, TGI, Unsloth, DeepSpeed)

**UI Pattern:** Radio cards with optimization badges

```typescript
<RadioGroup value={frameworkPreset} onChange={setFrameworkPreset}>
  <RadioCard value="baseline">
    <h3>Baseline</h3>
    <p>No framework-specific optimizations</p>
  </RadioCard>

  <RadioCard value="vllm" disabled={trainingMode !== 'inference'}>
    <h3>vLLM PagedAttention</h3>
    <p>2-4x inference speedup, 96% less KV cache waste</p>
    <Badge>Inference only</Badge>
  </RadioCard>

  <RadioCard value="unsloth" disabled={!(trainingMode === 'lora' || trainingMode === 'qlora')}>
    <h3>Unsloth</h3>
    <p>2x faster training, 70% less memory</p>
    <Badge>LoRA/QLoRA only</Badge>
  </RadioCard>
</RadioGroup>
```

---

### 5. Modified Output Components

#### `components/outputs/VRAMBreakdownChart.tsx` (MODIFIED)

**Changes:** Support training mode breakdown with additional slices

```typescript
const data = trainingMode !== 'inference'
  ? [
      { name: 'Model Weights', value: breakdown.modelWeights.toNumber(), color: COLORS.modelWeights },
      { name: 'Gradients', value: breakdown.gradients.toNumber(), color: COLORS.gradients },
      { name: 'Optimizer States', value: breakdown.optimizerStates.toNumber(), color: COLORS.optimizerStates },
      { name: 'LoRA Adapters', value: breakdown.loraAdapters.toNumber(), color: COLORS.loraAdapters },
      { name: 'Activations', value: breakdown.activations.toNumber(), color: COLORS.activations },
      { name: 'Framework', value: breakdown.frameworkOverhead.toNumber(), color: COLORS.framework },
    ].filter(d => d.value > 0)  // Hide zero-value slices
  : [
      // ... existing inference breakdown ...
    ]
```

---

#### `components/outputs/MemoryBreakdownTable.tsx` (MODIFIED)

**Changes:** Add training-specific rows

```typescript
{trainingMode !== 'inference' && (
  <>
    <tr>
      <td>Gradients</td>
      <td>{breakdown.gradients.toFixed(2)} GB</td>
      <td>Trainable params only ({trainableParams.toLocaleString()} params)</td>
    </tr>
    <tr>
      <td>Optimizer States</td>
      <td>{breakdown.optimizerStates.toFixed(2)} GB</td>
      <td>{optimizerType} ({bytesPerParam} bytes/param)</td>
    </tr>
  </>
)}
```

---

### 6. New Output Components

#### `components/outputs/FrameworkOptimizationBadge.tsx`

**Purpose:** Show active framework optimizations

```typescript
{frameworkPreset !== 'baseline' && (
  <div className="flex items-center gap-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
    <CheckCircleIcon className="w-5 h-5 text-blue-600" />
    <div>
      <h4 className="font-medium">{preset.framework} Optimizations Active</h4>
      <ul className="text-sm text-gray-600 dark:text-gray-400">
        {preset.optimizations.map(opt => (
          <li key={opt}>• {opt}</li>
        ))}
      </ul>
    </div>
  </div>
)}
```

---

### 7. Data Files

#### `src/data/optimizers.json` (NEW)

```json
[
  {
    "type": "adamw",
    "name": "AdamW",
    "description": "Standard AdamW optimizer with fp32 states",
    "bytesPerParam": 8,
    "stability": "high",
    "recommendedFor": ["full-ft", "lora"],
    "notes": "Default choice for fine-tuning. Stable convergence."
  },
  {
    "type": "adamw_8bit",
    "name": "AdamW 8-bit",
    "description": "8-bit quantized AdamW (bitsandbytes)",
    "bytesPerParam": 2,
    "stability": "high",
    "recommendedFor": ["full-ft", "lora", "qlora"],
    "notes": "4x memory reduction with same performance. Requires bitsandbytes."
  }
]
```

---

## Data Flow

### Inference Mode (EXISTING)

```
User Input (Model, GPU, Quantization, Batch, Seq Length)
  ↓
uiStore (Zustand)
  ↓
Engine: calculateInferenceVRAM()
  ├─ calculateModelWeightVRAM()
  ├─ calculateKVCacheVRAM()
  └─ calculateActivationMemory()
  ↓
InferenceVRAMBreakdown (Decimal.js)
  ↓
Components render (VRAMBreakdownChart, MemoryBreakdownTable, FitIndicator)
```

### Fine-Tuning Mode (NEW)

```
User Input (Model, Training Mode, LoRA Config, Optimizer, Framework Preset)
  ↓
uiStore (Zustand) - extended with training state
  ↓
Engine: calculateFineTuningVRAM()
  ├─ calculateModelWeightVRAM() [REUSED]
  ├─ calculateLoRAParams() [NEW]
  ├─ calculateOptimizerMemory() [NEW]
  ├─ calculateActivationMemory() [REUSED, extended]
  └─ applyFrameworkOptimizations() [NEW]
  ↓
FineTuningVRAMBreakdown (Decimal.js)
  ↓
Components render (VRAMBreakdownChart [EXTENDED], MemoryBreakdownTable [EXTENDED])
```

---

## Build Order (Suggested Sequence)

### Phase 1: Core Fine-Tuning Engine (Week 1)

1. **Extend types** (`src/engines/types.ts`)
2. **Build optimizer engine** (`src/engines/optimizer.ts`)
3. **Build LoRA engine** (`src/engines/lora.ts`)
4. **Build fine-tuning engine** (`src/engines/finetuning.ts`)

**Validation:** Test against known values (Llama 7B LoRA r=16 ≈ 16GB)

---

### Phase 2: Store & Integration (Week 1-2)

1. **Extend store** (`src/store/uiStore.ts`)
2. **Extend comparison store** (`src/store/comparisonStore.ts`)

---

### Phase 3: Input Components (Week 2)

1. **TrainingModeSelector** (`components/inputs/TrainingModeSelector.tsx`)
2. **LoRAConfigPanel** (`components/inputs/LoRAConfigPanel.tsx`)
3. **OptimizerSelector** (`components/inputs/OptimizerSelector.tsx`)
4. **GradientAccumulationInput** (`components/inputs/GradientAccumulationInput.tsx`)

---

### Phase 4: Output Components (Week 2-3)

1. **Extend VRAMBreakdownChart** (`components/outputs/VRAMBreakdownChart.tsx`)
2. **Extend MemoryBreakdownTable** (`components/outputs/MemoryBreakdownTable.tsx`)
3. **FrameworkOptimizationBadge** (`components/outputs/FrameworkOptimizationBadge.tsx`)

---

### Phase 5: Framework Presets (Week 3)

1. **Framework presets engine** (`src/engines/framework-presets.ts`)
2. **FrameworkPresetSelector** (`components/inputs/FrameworkPresetSelector.tsx`)
3. **Data files** (`src/data/framework-presets.json`, `src/data/optimizers.json`)

---

### Phase 6: Testing & Validation (Week 3-4)

1. **Integration tests**
2. **UI/UX testing**
3. **Documentation**

---

## Patterns to Follow

### 1. Pure Engine Functions

```typescript
// ✅ GOOD: Pure function, no side effects
export function calculateLoRAParams(
  model: Model,
  config: LoRAConfig,
): LoRAParamBreakdown {
  const paramsPerModule = 2 * model.hidden_size * config.rank
  return breakdown
}
```

### 2. Zod-First Type System

```typescript
export const LoRAConfigSchema = z.object({
  rank: z.number().int().positive().min(1).max(256),
  alpha: z.number().int().positive(),
})

export type LoRAConfig = z.infer<typeof LoRAConfigSchema>
```

### 3. Conditional Component Rendering

```typescript
{(trainingMode === 'lora' || trainingMode === 'qlora') && (
  <LoRAConfigPanel />
)}
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Engine Calling React Hooks

```typescript
// ❌ BAD
export function calculateFineTuningVRAM() {
  const model = useUIStore(s => s.selectedModel) // NO!
}

// ✅ GOOD
export function calculateFineTuningVRAM(params: {
  model: Model
}): FineTuningVRAMBreakdown {
  // ... pure calculation ...
}
```

---

## Sources

### High Confidence (Official Documentation)

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DeepSpeed ZeRO Documentation](https://www.deepspeed.ai/tutorials/zero/)
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [HuggingFace TGI Documentation](https://huggingface.co/docs/text-generation-inference/en/index)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [HuggingFace Optimizers](https://huggingface.co/docs/transformers/en/optimizers)

### Medium Confidence (Web Search Verified)

- [Gradient Accumulation Analysis (2025)](https://arxiv.org/abs/2507.07101)
- [Modal Blog: Fine-Tuning VRAM](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [Databricks LoRA Guide](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)

---

## Summary

Fine-tuning features integrate cleanly with existing architecture by:

1. **Adding parallel engines** (`finetuning.ts`, `lora.ts`, `optimizer.ts`, `framework-presets.ts`)
2. **Extending store** with training state (mode, optimizer, LoRA config, framework preset)
3. **Reusing existing primitives** (quantization, activation calculation, model schema)
4. **Creating conditional components** (LoRAConfigPanel shows only for LoRA/QLoRA)
5. **Extending visualizations** (VRAMBreakdownChart adds training slices)

**Integration complexity:** MEDIUM - New calculation domains but familiar patterns

**Build order:** Engines → Store → Input Components → Output Components → Framework Presets

**Key risk:** Accurate formulas for optimizer memory and LoRA parameters. Mitigation: Validate against HuggingFace transformers reported memory.
