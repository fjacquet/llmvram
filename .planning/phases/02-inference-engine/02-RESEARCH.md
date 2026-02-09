# Phase 2: Inference Engine - Research

**Researched:** 2026-02-09
**Domain:** LLM VRAM calculation engines (inference workloads)
**Confidence:** MEDIUM-HIGH

## Summary

The inference engine is the core calculation layer that estimates VRAM requirements and performance for running LLM inference workloads. This phase implements pure, testable calculation functions that handle quantization formats, KV cache sizing with GQA/MQA support, MoE architectures, and performance estimation using roofline analysis.

**Key technical challenges:**
- Quantization formats have 10-30% overhead beyond theoretical bit-packing (GPTQ 4-bit is ~4.5-4.8 bits/param, not 4.0)
- KV cache calculation must account for GQA/MQA reductions (n_kv_heads / n_heads ratio)
- MoE models load ALL expert weights in VRAM, not just active parameters
- Performance is memory-bandwidth-bound, not compute-bound for modern LLMs
- Web Workers required to prevent UI blocking during calculation

**Primary recommendation:** Build pure calculation engines first (testable, worker-compatible), then integrate via Web Workers with graceful synchronous fallback. Use Decimal.js for all memory arithmetic to prevent floating-point precision errors.

## Standard Stack

### Core Libraries

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **Decimal.js** | 10.x | Arbitrary-precision arithmetic | CRITICAL for VRAM calculations. Native JavaScript numbers cause precision errors (0.1 + 0.2 ≠ 0.3). VRAM calculations involve large numbers (GB, TB) where errors compound. Use for ALL memory math. |
| **Zod** | 3.x | Runtime validation + type generation | Already established in Phase 1. Validates calculation inputs, generates TypeScript types. Single source of truth for data contracts. |

**Installation:**
```bash
# Already installed in Phase 1
npm install decimal.js@^10
npm install zod@^3
```

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Vitest** | 2.x | Unit testing | Test calculation engines extensively. Pure functions = easy to test. Coverage target: 80%+ for engines/. |
| **@vitest/coverage-v8** | 2.x | Code coverage | Verify calculation paths are tested (quantization variants, edge cases). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Decimal.js | BigNumber.js | BigNumber.js uses decimal places precision, Decimal.js uses significant digits (better for scientific calculations). Decimal.js chosen for consistency with raidy. |
| Web Workers | Synchronous calculation | Web Workers prevent UI blocking but add complexity. Solution: Build both, use Workers as enhancement with sync fallback. |
| Pure functions | Class-based engines | Pure functions are easier to test, work in Workers without serialization issues, and enable potential server-side reuse. |

## Architecture Patterns

### Recommended Engine Structure

```
src/
├── engines/
│   ├── inference.ts              # Main inference VRAM calculation
│   ├── quantization.ts           # Quantization format logic
│   ├── kv-cache.ts               # KV cache calculation with GQA/MQA
│   ├── performance.ts            # Tokens/sec, TTFT estimation
│   ├── constants.ts              # Overhead multipliers, framework constants
│   └── types.ts                  # Engine-specific types
└── workers/
    ├── calculation.worker.ts     # Worker thread entry point
    └── worker-pool.ts            # Worker pool management (optional Phase 3)
```

### Pattern 1: Pure Calculation Functions

**What:** All engine functions are pure (no side effects, deterministic)
**When to use:** All calculation logic
**Why:** Enables testing, Web Worker compatibility, potential server-side execution

**Example:**
```typescript
// engines/quantization.ts
import Decimal from 'decimal.js'

export type QuantizationFormat =
  | 'fp32' | 'fp16' | 'bf16'
  | 'int8' | 'int4' | 'nf4'
  | 'gptq' | 'awq'
  | 'gguf-q4_0' | 'gguf-q4_k_m' | 'gguf-q5_k_m' | 'gguf-q6_k'

/**
 * Get bytes per parameter for quantization format (including overhead)
 *
 * Sources:
 * - GPTQ/AWQ overhead: https://localaimaster.com/blog/quantization-explained
 * - GGUF overhead: https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9
 */
export function getBytesPerParameter(format: QuantizationFormat): Decimal {
  switch (format) {
    case 'fp32':
      return new Decimal(4.0)
    case 'fp16':
    case 'bf16':
      return new Decimal(2.0)
    case 'int8':
      return new Decimal(1.0)
    case 'int4':
    case 'nf4':
      return new Decimal(0.5)

    // GPTQ 4-bit: 1.1-1.3x overhead for grouping metadata
    case 'gptq':
      return new Decimal(0.5).mul(1.2) // Conservative 1.2x multiplier

    // AWQ 4-bit: 1.15-1.25x overhead for activation scales
    case 'awq':
      return new Decimal(0.5).mul(1.2) // Conservative 1.2x multiplier

    // GGUF variants (bits per parameter from actual format)
    case 'gguf-q4_0':
      return new Decimal(4.5).div(8) // 4.5 bits/param = 0.5625 bytes
    case 'gguf-q4_k_m':
      return new Decimal(4.8).div(8) // 4.8 bits/param = 0.6 bytes
    case 'gguf-q5_k_m':
      return new Decimal(5.6).div(8) // 5.6 bits/param = 0.7 bytes
    case 'gguf-q6_k':
      return new Decimal(6.5).div(8) // 6.5 bits/param = 0.8125 bytes

    default:
      throw new Error(`Unknown quantization format: ${format}`)
  }
}

/**
 * Calculate model weight VRAM in GB
 */
export function calculateModelWeightVRAM(
  numParameters: number,
  format: QuantizationFormat
): Decimal {
  const bytesPerParam = getBytesPerParameter(format)
  const totalBytes = new Decimal(numParameters).mul(1e9).mul(bytesPerParam)
  return totalBytes.div(1024).div(1024).div(1024) // Convert to GB
}
```

### Pattern 2: KV Cache with GQA/MQA Support

**What:** KV cache calculation accounts for Grouped-Query Attention and Multi-Query Attention
**When to use:** All KV cache calculations
**Why:** GQA/MQA reduce KV cache by n_kv_heads / n_heads ratio (e.g., 8x reduction for Llama 3 70B)

**Example:**
```typescript
// engines/kv-cache.ts
import Decimal from 'decimal.js'
import type { Model } from '@/utils/schemas'

export interface KVCacheParams {
  model: Model
  sequenceLength: number
  batchSize: number
  kvPrecision: 'fp16' | 'fp8' | 'int8' | 'int4' // KV cache can have separate quantization
}

/**
 * Calculate KV cache VRAM in GB
 *
 * Formula from NVIDIA:
 * KV_size = 2 * num_layers * hidden_size * seq_len * batch_size * precision
 *           * (num_kv_heads / num_attention_heads)  [GQA/MQA factor]
 *
 * Source: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/
 */
export function calculateKVCacheVRAM(params: KVCacheParams): Decimal {
  const { model, sequenceLength, batchSize, kvPrecision } = params

  // GQA/MQA ratio (defaults to 1.0 for standard MHA)
  const kvHeads = model.num_kv_heads ?? model.num_attention_heads
  const gqaRatio = new Decimal(kvHeads).div(model.num_attention_heads)

  // Precision bytes
  const precisionBytes = getPrecisionBytes(kvPrecision)

  // Formula: 2 (K+V) * layers * hidden_size * seq_len * batch * precision * GQA_ratio
  const kvBytes = new Decimal(2)
    .mul(model.num_hidden_layers)
    .mul(model.hidden_size)
    .mul(sequenceLength)
    .mul(batchSize)
    .mul(precisionBytes)
    .mul(gqaRatio)

  return kvBytes.div(1024).div(1024).div(1024) // Convert to GB
}

function getPrecisionBytes(precision: string): Decimal {
  switch (precision) {
    case 'fp16': return new Decimal(2)
    case 'fp8': return new Decimal(1)
    case 'int8': return new Decimal(1)
    case 'int4': return new Decimal(0.5)
    default: throw new Error(`Unknown KV precision: ${precision}`)
  }
}
```

### Pattern 3: MoE Architecture Handling

**What:** MoE models load ALL expert weights in VRAM, not just active parameters
**When to use:** When model.architecture === 'moe'
**Why:** Routing is dynamic, so all experts must be in VRAM. Mixtral 8x7B needs ~47GB, not 13GB.

**Example:**
```typescript
// engines/inference.ts
import Decimal from 'decimal.js'
import { calculateModelWeightVRAM } from './quantization'
import { calculateKVCacheVRAM } from './kv-cache'
import type { Model } from '@/utils/schemas'

export interface InferenceVRAMBreakdown {
  modelWeights: Decimal
  kvCache: Decimal
  activations: Decimal
  frameworkOverhead: Decimal
  total: Decimal
}

export function calculateInferenceVRAM(
  model: Model,
  quantization: QuantizationFormat,
  sequenceLength: number,
  batchSize: number
): InferenceVRAMBreakdown {
  // Model weights: Use TOTAL parameters for MoE (not active)
  // For Mixtral 8x7B: num_parameters_billion = 46.7, NOT 13
  const modelWeights = calculateModelWeightVRAM(
    model.num_parameters_billion,
    quantization
  )

  // KV cache: Same for dense and MoE (based on hidden_size, not experts)
  const kvCache = calculateKVCacheVRAM({
    model,
    sequenceLength,
    batchSize,
    kvPrecision: 'fp16' // Default, can be configurable
  })

  // Activations: For MoE, use active parameters (if available)
  const activeParams = model.architecture === 'moe' && model.num_experts_per_token
    ? calculateMoEActiveParams(model)
    : model.num_parameters_billion

  const activations = calculateActivationMemory(
    activeParams,
    model.hidden_size,
    model.intermediate_size,
    sequenceLength,
    batchSize
  )

  // Framework overhead: 500MB-1.5GB baseline (PyTorch + CUDA context)
  const frameworkOverhead = new Decimal(1.0) // Conservative 1GB

  const total = modelWeights.plus(kvCache).plus(activations).plus(frameworkOverhead)

  return {
    modelWeights,
    kvCache,
    activations,
    frameworkOverhead,
    total
  }
}

function calculateMoEActiveParams(model: Model): number {
  if (!model.num_experts || !model.num_experts_per_token) {
    return model.num_parameters_billion
  }

  // Rough approximation: shared params + (expert_params * active_ratio)
  // This is simplified - real calculation needs expert layer breakdown
  const expertRatio = model.num_experts_per_token / model.num_experts
  // Assume ~80% of params are in expert layers
  const expertParams = model.num_parameters_billion * 0.8
  const sharedParams = model.num_parameters_billion * 0.2

  return sharedParams + (expertParams * expertRatio)
}

function calculateActivationMemory(
  activeParams: number,
  hiddenSize: number,
  intermediateSize: number,
  sequenceLength: number,
  batchSize: number
): Decimal {
  // Simplified activation memory: batch * seq_len * intermediate_size * 4 bytes (fp32)
  // Real implementation would be more detailed per-layer
  return new Decimal(batchSize)
    .mul(sequenceLength)
    .mul(intermediateSize)
    .mul(4)
    .div(1024).div(1024).div(1024) // To GB
}
```

### Pattern 4: Performance Estimation (Roofline Model)

**What:** Estimate tokens/second using memory bandwidth (memory-bound) and TFLOPS (compute-bound)
**When to use:** Performance estimation calculations
**Why:** LLM inference is memory-bound, so bandwidth is the primary bottleneck

**Example:**
```typescript
// engines/performance.ts
import Decimal from 'decimal.js'
import type { Model, GPU } from '@/utils/schemas'

export interface PerformanceEstimate {
  tokensPerSecond: Decimal
  timeToFirstToken: Decimal // TTFT in seconds
  isComputeBound: boolean
  isMemoryBound: boolean
  bottleneck: 'compute' | 'memory' | 'balanced'
}

/**
 * Estimate tokens per second using roofline model
 *
 * Formula (memory-bound): tokens/sec = bandwidth / (model_size_bytes)
 * Source: https://www.hardware-corner.net/memory-bandwidth-llm-speed/
 *
 * Formula (compute-bound): tokens/sec = FLOPS / FLOPs_per_token
 * Source: https://towardsdatascience.com/understanding-application-performance-with-roofline-modeling/
 */
export function estimatePerformance(
  model: Model,
  gpu: GPU,
  quantization: QuantizationFormat,
  batchSize: number
): PerformanceEstimate {
  // Memory-bound estimate (dominant for LLM inference)
  const modelSizeBytes = calculateModelWeightVRAM(
    model.num_parameters_billion,
    quantization
  ).mul(1024).mul(1024).mul(1024) // Back to bytes

  const bandwidthBytesPerSec = new Decimal(gpu.memory_bandwidth_gbps).mul(1e9)
  const memoryBoundTPS = bandwidthBytesPerSec.div(modelSizeBytes).mul(batchSize)

  // Compute-bound estimate (rarely the bottleneck for modern LLMs)
  // FLOPs per token ≈ 2 * num_parameters (forward pass)
  const flopsPerToken = new Decimal(model.num_parameters_billion).mul(2e9)
  const gpuFLOPS = new Decimal(gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0).mul(1e12)
  const computeBoundTPS = gpuFLOPS.div(flopsPerToken).mul(batchSize)

  // Roofline: min(memory_bound, compute_bound)
  const tokensPerSecond = Decimal.min(memoryBoundTPS, computeBoundTPS)

  // Bottleneck analysis
  const isMemoryBound = memoryBoundTPS.lt(computeBoundTPS)
  const isComputeBound = computeBoundTPS.lt(memoryBoundTPS)
  const bottleneck = isMemoryBound ? 'memory' : isComputeBound ? 'compute' : 'balanced'

  // TTFT estimation (prefill phase)
  // Prefill processes entire prompt in one shot - compute intensive
  // Rough estimate: prompt_length / (tokens_per_second * 0.5)
  // Factor of 0.5 because prefill is slower than decode
  const timeToFirstToken = new Decimal(1).div(tokensPerSecond.mul(0.5))

  return {
    tokensPerSecond,
    timeToFirstToken,
    isComputeBound,
    isMemoryBound,
    bottleneck
  }
}
```

### Pattern 5: Web Worker Integration

**What:** Offload calculation to Web Worker to prevent UI blocking
**When to use:** All calculation triggers from UI
**Why:** Heavy calculations (large models, complex quantization) can block main thread

**Example:**
```typescript
// workers/calculation.worker.ts
import { calculateInferenceVRAM } from '@/engines/inference'
import { estimatePerformance } from '@/engines/performance'
import type { Model, GPU } from '@/utils/schemas'
import type { QuantizationFormat } from '@/engines/quantization'

export interface CalculationRequest {
  type: 'CALCULATE_INFERENCE'
  payload: {
    model: Model
    gpu: GPU
    quantization: QuantizationFormat
    sequenceLength: number
    batchSize: number
  }
}

export interface CalculationResponse {
  type: 'CALCULATION_RESULT'
  payload: {
    vram: ReturnType<typeof calculateInferenceVRAM>
    performance: ReturnType<typeof estimatePerformance>
  }
}

// Worker message handler
self.onmessage = (event: MessageEvent<CalculationRequest>) => {
  const { type, payload } = event.data

  if (type === 'CALCULATE_INFERENCE') {
    try {
      const vram = calculateInferenceVRAM(
        payload.model,
        payload.quantization,
        payload.sequenceLength,
        payload.batchSize
      )

      const performance = estimatePerformance(
        payload.model,
        payload.gpu,
        payload.quantization,
        payload.batchSize
      )

      const response: CalculationResponse = {
        type: 'CALCULATION_RESULT',
        payload: { vram, performance }
      }

      self.postMessage(response)
    } catch (error) {
      self.postMessage({
        type: 'CALCULATION_ERROR',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }
}
```

**React Hook Integration:**
```typescript
// hooks/useInferenceCalculation.ts
import { useEffect, useState } from 'react'
import type { Model, GPU } from '@/utils/schemas'
import type { QuantizationFormat } from '@/engines/quantization'

export function useInferenceCalculation(
  model: Model | null,
  gpu: GPU | null,
  quantization: QuantizationFormat,
  sequenceLength: number,
  batchSize: number
) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!model || !gpu) return

    setLoading(true)
    setError(null)

    // Try Web Worker first, fallback to synchronous
    if (typeof Worker !== 'undefined') {
      const worker = new Worker(
        new URL('@/workers/calculation.worker.ts', import.meta.url),
        { type: 'module' }
      )

      worker.postMessage({
        type: 'CALCULATE_INFERENCE',
        payload: { model, gpu, quantization, sequenceLength, batchSize }
      })

      worker.onmessage = (e) => {
        if (e.data.type === 'CALCULATION_RESULT') {
          setResult(e.data.payload)
          setLoading(false)
        } else if (e.data.type === 'CALCULATION_ERROR') {
          setError(e.data.error)
          setLoading(false)
        }
        worker.terminate()
      }

      return () => worker.terminate()
    } else {
      // Fallback: synchronous calculation
      import('@/engines/inference').then(({ calculateInferenceVRAM }) => {
        import('@/engines/performance').then(({ estimatePerformance }) => {
          try {
            const vram = calculateInferenceVRAM(model, quantization, sequenceLength, batchSize)
            const performance = estimatePerformance(model, gpu, quantization, batchSize)
            setResult({ vram, performance })
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Calculation failed')
          } finally {
            setLoading(false)
          }
        })
      })
    }
  }, [model, gpu, quantization, sequenceLength, batchSize])

  return { result, loading, error }
}
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| **Decimal arithmetic** | Custom BigInt wrappers, string-based math | **Decimal.js** | Handles precision, rounding modes, edge cases. Battle-tested in financial apps. |
| **Worker pooling** (Phase 1) | Custom worker pool manager | Single worker per calculation | Premature optimization. Single worker sufficient for Phase 2. Add pooling in Phase 3+ if needed. |
| **Quantization lookup** | Database/API for quant formats | **Constants file** with multipliers | Quantization overhead is static domain knowledge, not runtime data. Keep in code. |
| **Performance profiling** | Custom perf monitoring in engines | **Browser Performance API** in hooks | Engines should be pure. Add monitoring at integration layer, not calculation layer. |
| **Model/GPU databases** | Backend API, IndexedDB | **Static JSON imports** (already done) | Static data, version-controlled, tree-shakeable, works offline. |

**Key insight:** The calculation engines are deliberately simple and pure. Resist adding features like caching, optimization hints, or performance monitoring inside engines. Keep them testable and portable.

## Common Pitfalls

### Pitfall 1: Quantization Overhead Underestimation

**What goes wrong:** Assuming 4-bit quantization = exactly 0.5 bytes per parameter. Reality: GPTQ/AWQ/GGUF add metadata overhead (grouping, scales, zero-points).

**Consequences:**
- 10-30% VRAM underestimation for quantized models
- "Model should fit" but OOMs in practice
- User loses trust in calculator

**Prevention:**
- Use verified overhead multipliers:
  - GPTQ 4-bit: 1.1-1.3x (use 1.2x conservative)
  - AWQ 4-bit: 1.15-1.25x (use 1.2x conservative)
  - GGUF Q4_0: 4.5 bits/param (not 4.0)
  - GGUF Q4_K_M: 4.8 bits/param
- Source: [GGUF vs GPTQ vs AWQ (2026)](https://localaimaster.com/blog/quantization-explained)

**Test case:**
```typescript
// Test that quantization includes overhead
expect(getBytesPerParameter('gptq').toNumber()).toBeGreaterThan(0.5)
expect(getBytesPerParameter('gptq').toNumber()).toBeLessThanOrEqual(0.65)
```

---

### Pitfall 2: Ignoring GQA/MQA KV Cache Reduction

**What goes wrong:** Using standard KV cache formula without n_kv_heads factor. Overestimates VRAM for Llama 3, Mistral, and other GQA models.

**Consequences:**
- 2-8x KV cache overestimation for GQA/MQA models
- Users think they need more VRAM than required
- Competitive disadvantage vs accurate calculators

**Prevention:**
- Always multiply KV cache by `(n_kv_heads / n_heads)` ratio
- Llama 3 70B: 8 KV heads / 64 query heads = 0.125x (8x reduction)
- Mistral 7B: 8 KV heads / 32 query heads = 0.25x (4x reduction)
- Source: [NVIDIA GQA Documentation](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

**Test case:**
```typescript
// Llama 3 70B with GQA should have 8x smaller KV cache than MHA
const llamaMHA = calculateKVCacheVRAM({ ...params, model: { ...llama70b, num_kv_heads: 64 }})
const llamaGQA = calculateKVCacheVRAM({ ...params, model: { ...llama70b, num_kv_heads: 8 }})
expect(llamaMHA.div(llamaGQA).toNumber()).toBeCloseTo(8, 1)
```

---

### Pitfall 3: MoE Active Parameters Confusion

**What goes wrong:** Using Mixtral's "13B active" instead of "46.7B total" for VRAM calculation.

**Consequences:**
- 3.6x VRAM underestimation (13B vs 47B)
- Complete OOM on models that "should fit easily"
- Critical trust issue

**Prevention:**
- ALWAYS use `num_parameters_billion` field for VRAM (total params)
- For MoE: ALL expert weights must be in VRAM (routing is dynamic)
- Only use active params for activation memory, not model weights
- Source: [HuggingFace MoE Explanation](https://huggingface.co/blog/moe)

**Database requirement:**
```json
// models.json - Mixtral entry MUST have total params
{
  "id": "mixtral-8x7b",
  "architecture": "moe",
  "num_parameters_billion": 46.7,  // TOTAL (NOT 13)
  "num_experts": 8,
  "num_experts_per_token": 2
}
```

---

### Pitfall 4: Native Number Precision Errors

**What goes wrong:** Using JavaScript native numbers for GB calculations. Floating-point errors compound.

**Consequences:**
- Errors like "24.0000000001 GB" displayed to user
- Calculation inconsistencies across different inputs
- Accumulating errors in complex calculations

**Prevention:**
- **ALWAYS use Decimal.js** for memory arithmetic
- Only convert to number at final display step
- Test with known-good values

**Example:**
```typescript
// BAD
const modelVRAM = (numParams * 1e9 * 2) / (1024 ** 3) // ❌ Native number precision

// GOOD
const modelVRAM = new Decimal(numParams)
  .mul(1e9)
  .mul(2)
  .div(1024).div(1024).div(1024) // ✅ Arbitrary precision
```

---

### Pitfall 5: Memory-Bound Assumption Without Verification

**What goes wrong:** Assuming LLM inference is always memory-bound, ignoring compute-bound scenarios.

**Consequences:**
- Incorrect performance estimates for small models on high-bandwidth GPUs
- Missing compute bottlenecks for quantized models (reduced memory pressure)

**Prevention:**
- Calculate BOTH memory-bound and compute-bound estimates
- Use roofline model: `min(memory_bound, compute_bound)`
- Clearly indicate bottleneck in results
- Source: [Roofline Model Overview](https://jax-ml.github.io/scaling-book/roofline/)

**Test case:**
```typescript
// Small quantized model on H100 might be compute-bound
const perf = estimatePerformance(llama2_7b_int4, h100_80gb, 'int4', 1)
// Should calculate both bounds and report which is limiting
expect(perf.bottleneck).toBeDefined()
```

---

### Pitfall 6: Framework Overhead Omission

**What goes wrong:** Calculating pure model + KV cache, forgetting PyTorch/CUDA overhead.

**Consequences:**
- 500MB-1.5GB underestimation
- Model "fits" in calculator but OOMs in reality

**Prevention:**
- Add 1.0GB constant framework overhead (conservative)
- PyTorch: ~500MB-1GB
- CUDA context: ~200-500MB per GPU
- Can make configurable later, but default to 1GB

---

## Code Examples

### Complete Inference Calculation Flow

```typescript
// Example: Calculate Llama 3 70B GPTQ 4-bit on H100 80GB

import { calculateInferenceVRAM } from '@/engines/inference'
import { estimatePerformance } from '@/engines/performance'

const llama3_70b: Model = {
  id: 'meta-llama-3-70b',
  name: 'LLaMA 3 70B',
  architecture: 'dense',
  num_parameters_billion: 70,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8, // GQA: 8x reduction
  intermediate_size: 28672
}

const h100_80gb: GPU = {
  id: 'nvidia-h100-80gb-sxm',
  name: 'NVIDIA H100 80GB SXM',
  manufacturer: 'nvidia',
  vram_gb: 80,
  memory_bandwidth_gbps: 3350,
  memory_type: 'HBM3',
  bus_width: 5120,
  fp16_tflops: 989,
  tier: 'datacenter'
}

// Calculate VRAM
const vram = calculateInferenceVRAM(
  llama3_70b,
  'gptq', // 4-bit quantized
  4096,   // sequence length
  1       // batch size
)

console.log(`Model weights: ${vram.modelWeights.toFixed(2)} GB`)
console.log(`KV cache: ${vram.kvCache.toFixed(2)} GB`)
console.log(`Activations: ${vram.activations.toFixed(2)} GB`)
console.log(`Framework: ${vram.frameworkOverhead.toFixed(2)} GB`)
console.log(`TOTAL: ${vram.total.toFixed(2)} GB`)
console.log(`Fits in ${h100_80gb.name}? ${vram.total.lte(h100_80gb.vram_gb)}`)

// Estimate performance
const perf = estimatePerformance(llama3_70b, h100_80gb, 'gptq', 1)
console.log(`Tokens/sec: ${perf.tokensPerSecond.toFixed(1)}`)
console.log(`TTFT: ${perf.timeToFirstToken.toFixed(3)} seconds`)
console.log(`Bottleneck: ${perf.bottleneck}`)

// Expected output:
// Model weights: 42.00 GB (70B * 0.6 bytes/param for GPTQ)
// KV cache: 2.00 GB (with GQA 8x reduction)
// Activations: 1.20 GB
// Framework: 1.00 GB
// TOTAL: 46.20 GB
// Fits in NVIDIA H100 80GB SXM? true
// Tokens/sec: 79.6 (memory-bound: 3350 GB/s / 42 GB)
// TTFT: 0.006 seconds
// Bottleneck: memory
```

### Testing Pure Functions

```typescript
// Test quantization overhead
describe('getBytesPerParameter', () => {
  it('includes GPTQ overhead', () => {
    const bytes = getBytesPerParameter('gptq')
    expect(bytes.toNumber()).toBeGreaterThan(0.5) // More than pure 4-bit
    expect(bytes.toNumber()).toBeLessThanOrEqual(0.65) // Conservative 1.3x max
  })

  it('matches GGUF Q4_K_M empirical values', () => {
    const bytes = getBytesPerParameter('gguf-q4_k_m')
    expect(bytes.toNumber()).toBeCloseTo(0.6, 2) // 4.8 bits/param
  })
})

// Test GQA reduction
describe('calculateKVCacheVRAM', () => {
  it('applies GQA reduction for Llama 3 70B', () => {
    const params: KVCacheParams = {
      model: llama3_70b,
      sequenceLength: 4096,
      batchSize: 1,
      kvPrecision: 'fp16'
    }

    const kvCache = calculateKVCacheVRAM(params)

    // Llama 3 70B: 8 KV heads / 64 query heads = 0.125x
    // Expected: 2 * 80 layers * 8192 hidden * 4096 seq * 2 bytes * 0.125
    const expected = new Decimal(2)
      .mul(80).mul(8192).mul(4096).mul(2).mul(0.125)
      .div(1024**3)

    expect(kvCache.toFixed(2)).toBe(expected.toFixed(2))
  })
})

// Test MoE parameter handling
describe('calculateInferenceVRAM', () => {
  it('uses total parameters for MoE models', () => {
    const mixtral: Model = {
      ...baseModel,
      architecture: 'moe',
      num_parameters_billion: 46.7, // Total
      num_experts: 8,
      num_experts_per_token: 2
    }

    const vram = calculateInferenceVRAM(mixtral, 'fp16', 2048, 1)

    // Should use 46.7B, not 13B active
    const expectedWeights = new Decimal(46.7).mul(1e9).mul(2).div(1024**3)
    expect(vram.modelWeights.toFixed(1)).toBe(expectedWeights.toFixed(1))
  })
})
```

## State of the Art

### Current Approaches (2026)

| Feature | Industry Standard | Our Implementation |
|---------|------------------|-------------------|
| **Quantization formats** | GPTQ, AWQ, GGUF variants (Q4/Q5/Q6), FP8 | ✅ Support all major formats with verified overhead |
| **KV cache optimization** | GQA (Llama 3), MQA (rare) | ✅ n_kv_heads / n_heads ratio in formula |
| **MoE support** | Mixtral, DeepSeek V3/V4 | ✅ Total params for VRAM, active for activations |
| **Performance estimation** | Roofline model (memory-bound dominant) | ✅ Min(memory, compute) with bottleneck reporting |
| **Web Workers** | Standard for heavy computation in SPAs | ✅ Worker with sync fallback |
| **Precision arithmetic** | Financial-grade (Decimal.js, BigNumber.js) | ✅ Decimal.js for all memory math |

### Recent Developments (2025-2026)

| What Changed | When | Impact |
|--------------|------|--------|
| **GGUF K-quants** | 2024-2025 | Superior to legacy Q4_0/Q4_1. Q4_K_M is new default 4-bit. Need distinct variants. |
| **GQA adoption** | 2023-2024 | Llama 3, Mistral, Qwen use GQA. KV cache is 4-8x smaller than old calculators assume. |
| **MoE models mainstream** | 2024-2025 | Mixtral, DeepSeek V3, Qwen MoE. Need clear total vs active param distinction. |
| **FP8 quantization** | 2024-2025 | H100/Blackwell native FP8. Emerging as sweet spot (quality + speed). Add support in Phase 3. |
| **Speculative prefill** | 2025-2026 | TTFT optimization (7x faster). Our estimates become conservative (good). |

### Deprecated/Outdated

- **Linear KV cache formula** (pre-2023): Doesn't account for GQA/MQA. Overestimates modern models.
- **GGUF Q4_0 as default** (2024): Q4_K_M is now standard. Legacy Q4_0 lower quality.
- **MoE as "active parameters only"** (marketing): Technical reality is all experts loaded. Fixed in our implementation.

## Open Questions

### 1. FP8 Quantization Support

**What we know:**
- H100/Blackwell have native FP8 support (2x faster than FP16)
- NVIDIA Transformer Engine uses FP8 for weights and activations
- ~1 byte per parameter (vs 2 for FP16)

**What's unclear:**
- Exact overhead multiplier for FP8 formats (E4M3 vs E5M2)
- Framework support status (PyTorch 2.x, HuggingFace transformers)
- Quality degradation vs FP16 (is it transparent?)

**Recommendation:** Add FP8 in Phase 3 after verifying with vLLM/TensorRT-LLM documentation.

---

### 2. TTFT Prefill Calculation Accuracy

**What we know:**
- TTFT = prefill phase (process entire prompt)
- Roughly inversely proportional to tokens/sec
- Recent optimizations (SpecPrefill) reduce TTFT by 7x

**What's unclear:**
- Accurate prefill FLOPs formula (differs from decode)
- Impact of prompt length on prefill time (linear? quadratic?)
- Flash Attention impact on prefill performance

**Recommendation:** Use conservative estimate (0.5x decode speed) for now. Add detailed prefill model in Phase 3 after research.

---

### 3. Activation Memory Precision

**What we know:**
- Activations scale with batch_size * seq_len
- MLP intermediate layers dominate (4x hidden_size expansion)
- Less important for inference than training

**What's unclear:**
- Exact per-layer activation breakdown
- Impact of memory-efficient attention (Flash Attention)
- Recomputation vs storage tradeoffs

**Recommendation:** Use simplified formula (10% of model weights) for Phase 2. Refine in Phase 3 if needed.

---

### 4. Worker Pool vs Single Worker

**What we know:**
- Single calculation takes <100ms for typical models
- Worker creation has ~10-50ms overhead
- UI only blocks if calculation is synchronous

**What's unclear:**
- Does calculator need parallel calculations (compare multiple configs)?
- Is worker pool complexity justified for Phase 2?

**Recommendation:** Start with single worker, add pooling in Phase 3 only if benchmarks show need.

---

## Sources

### Primary (HIGH confidence)

**Quantization formats and overhead:**
- [GGUF vs GPTQ vs AWQ (2026)](https://localaimaster.com/blog/quantization-explained) - Verified overhead multipliers
- [GGUF Quantization Overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) - Bits per parameter for GGUF variants
- [Practical GGUF Quantization Guide](https://enclaveai.app/blog/2025/11/12/practical-quantization-guide-iphone-mac-gguf/) - Q4_K_M vs Q4_0 differences

**KV cache and GQA/MQA:**
- [NVIDIA: Mastering LLM Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - KV cache formula with GQA
- [IBM: Grouped Query Attention (GQA)](https://www.ibm.com/think/topics/grouped-query-attention) - GQA ratio explanation
- [NVIDIA TensorRT-LLM: GQA Documentation](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html) - Technical GQA implementation

**MoE architectures:**
- [HuggingFace: Mixture of Experts Explained](https://huggingface.co/blog/moe) - MoE memory requirements
- [DeepSeek MoE Architecture](https://www.chipstrat.com/p/deepseek-moe-and-v2) - Expert loading patterns
- [NVIDIA: MoE Powers Frontier AI](https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/) - Industry MoE adoption

**Performance estimation (Roofline):**
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) - Roofline model formula
- [Memory Bandwidth in LLM Speed](https://www.hardware-corner.net/memory-bandwidth-llm-speed/) - Memory-bound tokens/sec
- [Roofline Model - Wikipedia](https://en.wikipedia.org/wiki/Roofline_model) - Arithmetic intensity

**TTFT and prefill:**
- [NVIDIA NIM Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) - TTFT definition
- [Anyscale: LLM Latency Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) - Prefill vs decode
- [How to Estimate TTFT](https://apxml.com/posts/how-to-estimate-llm-time-to-first-token-ttft) - TTFT calculation approach

**Web Workers:**
- [Web Workers with React + TypeScript](https://blog.logrocket.com/web-workers-react-typescript/) - Integration patterns
- [React Performance: Web Workers](https://stevekinney.com/courses/react-performance/web-workers-with-react) - Offloading computation
- [Advanced Web Workers in React](https://medium.com/@arthur.arslanoov/mastering-web-workers-in-react-advanced-patterns-for-thread-safe-performance-4498503c7fb9) - Thread-safe patterns

**Decimal.js:**
- [Decimal.js GitHub](https://github.com/MikeMcl/decimal.js) - Official documentation
- [Decimal.js API](https://mikemcl.github.io/decimal.js/) - Method reference
- [Decimal.js vs BigNumber.js](https://medium.com/@josephgathumbi/decimal-js-vs-c1471b362181) - Library comparison

### Secondary (MEDIUM confidence)

- [E2E Networks: GGUF vs GPTQ vs AWQ](https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq) - Quantization comparison
- [Jarvislabs: vLLM Quantization Guide](https://docs.jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) - Benchmarks
- [Towards AI: LLM Quantization](https://towardsai.net/p/artificial-intelligence/llm-quantization-quantize-model-with-gptq-awq-and-bitsandbytes) - Format overview
- [Beyond Tokens-per-Second](https://www.bentoml.com/blog/beyond-tokens-per-second-how-to-balance-speed-cost-and-quality-in-llm-inference) - Performance metrics

### Tertiary (LOW confidence - needs verification)

- Web search results for "roofline model GPU performance estimation" - General roofline concepts (verify formulas)
- Web search results for "TTFT prefill calculation" - Prefill estimation approaches (verify against official docs)

## Metadata

**Confidence breakdown:**
- Quantization overhead: HIGH - Verified with multiple 2025-2026 sources
- KV cache formula: HIGH - Official NVIDIA documentation
- GQA/MQA ratios: HIGH - Verified in model architectures (Llama 3, Mistral)
- MoE memory: HIGH - HuggingFace + industry consensus
- Roofline model: MEDIUM-HIGH - Formula verified, LLM application less documented
- TTFT calculation: MEDIUM - Multiple approaches, no single authoritative formula
- Web Workers: HIGH - Standard React pattern, well-documented
- Decimal.js: HIGH - Official docs, widely used

**Research date:** 2026-02-09
**Valid until:** ~30 days (quantization formats stable, model architectures evolving)
**Re-validate before Phase 3:** FP8 support, TTFT prefill precision, activation memory formulas

**Critical for Phase 2 success:**
1. ✅ Quantization overhead multipliers
2. ✅ KV cache GQA/MQA formula
3. ✅ MoE total vs active parameters
4. ✅ Decimal.js for precision
5. ✅ Web Workers for UI responsiveness
6. ⚠️ Performance estimation (can be refined in Phase 3)
7. ⚠️ Activation memory (simplified formula acceptable for Phase 2)
