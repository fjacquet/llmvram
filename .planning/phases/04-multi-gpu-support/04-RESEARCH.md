# Phase 4: Multi-GPU Support - Research

**Researched:** 2026-02-09
**Domain:** Multi-GPU LLM inference memory distribution and performance modeling
**Confidence:** MEDIUM

## Summary

Multi-GPU support for LLM inference requires understanding two parallelism strategies (tensor parallelism and pipeline parallelism), accounting for memory replication overhead (embeddings, layer norms), communication buffer sizes (NCCL), and interconnect bandwidth differences (NVLink vs PCIe). This phase extends the existing pure calculation engines with multi-GPU distribution logic and adds per-GPU visualization.

**Key insight:** Multi-GPU memory is NOT simply `total_memory / n_gpus`. Tensor parallelism achieves 85-90% effective split due to replicated components and communication buffers. Pipeline parallelism adds 10-15% activation stashing overhead per stage.

**Primary recommendation:** Implement tensor parallelism first (simpler, more commonly used for inference), add pipeline parallelism as optional advanced mode. Use NVLink bandwidth as a warning indicator, not a blocker. Visualize per-GPU breakdown with stacked bar chart showing memory distribution.

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Decimal.js | current | Precise VRAM calculations | Already used throughout engines |
| Recharts | current | Data visualization | Already used for pie chart, has BarChart component |
| Zustand | current | State management | Already managing UI state |
| Zod | current | Schema validation | Already validates GPU/Model data |

### New Components Needed
| Component | Purpose | Implementation Pattern |
|-----------|---------|------------------------|
| Multi-GPU calculation engine | Calculate per-GPU VRAM distribution | Pure function in `src/engines/multi-gpu.ts` |
| Stacked bar chart | Visualize per-GPU memory breakdown | `src/components/outputs/MultiGPUBreakdownChart.tsx` |
| GPU count selector | Input for number of GPUs (1-8) | `src/components/inputs/GPUCountSelector.tsx` |
| Sharding strategy selector | TP/PP selection with explanations | `src/components/inputs/ShardingStrategySelector.tsx` |

**No new dependencies required** - all functionality can be implemented with existing stack.

## Architecture Patterns

### Recommended Type Extensions

```typescript
// src/engines/types.ts - Add multi-GPU types

/** Sharding strategy for multi-GPU distribution */
export type ShardingStrategy = 'tensor-parallel' | 'pipeline-parallel'

/** Per-GPU VRAM breakdown for multi-GPU configuration */
export interface MultiGPUVRAMBreakdown {
  /** Number of GPUs in configuration */
  numGPUs: number
  /** Sharding strategy used */
  strategy: ShardingStrategy
  /** Per-GPU memory breakdown */
  perGPU: InferenceVRAMBreakdown
  /** Components replicated across all GPUs (embeddings, layer norms) */
  replicatedMemory: Decimal
  /** Communication buffer overhead per GPU (NCCL) */
  communicationOverhead: Decimal
  /** Total memory requirement per GPU */
  totalPerGPU: Decimal
  /** Utilization percentage per GPU (totalPerGPU / gpu_vram_gb) */
  utilizationPercent: Decimal
}
```

### Pattern 1: Tensor Parallelism Memory Distribution

**What:** Split model weights horizontally across GPUs, with some components replicated
**When to use:** Single-node multi-GPU (requires fast interconnect like NVLink)
**Memory formula:**

```typescript
// Source: vLLM documentation + Megatron-LM paper
// https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
// https://arxiv.org/abs/1909.08053

function calculateTensorParallelVRAM(
  singleGPUBreakdown: InferenceVRAMBreakdown,
  model: Model,
  numGPUs: number
): MultiGPUVRAMBreakdown {
  // 1. Model weights are MOSTLY sharded (85-90% effective split)
  //    Embeddings and layer norms are replicated across all GPUs
  const embeddingMemory = calculateEmbeddingMemory(model) // vocab_size * hidden_size * bytes_per_param
  const layerNormMemory = calculateLayerNormMemory(model) // num_layers * 2 * hidden_size * 4 (fp32)
  const replicatedMemory = embeddingMemory.add(layerNormMemory)

  const shardableWeights = singleGPUBreakdown.modelWeights.sub(replicatedMemory)
  const weightsPerGPU = shardableWeights.div(numGPUs).add(replicatedMemory)

  // 2. KV cache is sharded by sequence dimension (linear split)
  const kvCachePerGPU = singleGPUBreakdown.kvCache.div(numGPUs)

  // 3. Activations are reduced (per-GPU tensor size is smaller)
  const activationsPerGPU = singleGPUBreakdown.activations.div(numGPUs)

  // 4. Framework overhead is PER-GPU (PyTorch + CUDA context + NCCL buffers)
  const baseOverhead = singleGPUBreakdown.frameworkOverhead // 1.0 GB
  const ncclBufferOverhead = new Decimal(0.2).mul(numGPUs - 1) // 200MB per peer GPU
  const frameworkOverheadPerGPU = baseOverhead.add(ncclBufferOverhead)

  // 5. Communication overhead (10-15% for tensor parallelism)
  const communicationOverhead = weightsPerGPU.mul(0.12)

  const totalPerGPU = weightsPerGPU
    .add(kvCachePerGPU)
    .add(activationsPerGPU)
    .add(frameworkOverheadPerGPU)
    .add(communicationOverhead)

  return {
    numGPUs,
    strategy: 'tensor-parallel',
    perGPU: {
      modelWeights: weightsPerGPU,
      kvCache: kvCachePerGPU,
      activations: activationsPerGPU,
      frameworkOverhead: frameworkOverheadPerGPU,
      total: totalPerGPU,
    },
    replicatedMemory,
    communicationOverhead,
    totalPerGPU,
    utilizationPercent: totalPerGPU.div(gpu.vram_gb).mul(100),
  }
}
```

**Key insight:** Embeddings (2-5% of model) and layer norms (<1%) are replicated. For Llama 3 70B, that's ~2-3GB replicated, not perfectly divided.

### Pattern 2: Pipeline Parallelism Memory Distribution

**What:** Split model layers into sequential stages across GPUs
**When to use:** Multi-node or when tensor parallelism bandwidth requirements exceed interconnect capability
**Memory formula:**

```typescript
// Source: DeepSpeed pipeline parallelism documentation
// https://www.deepspeed.ai/tutorials/pipeline/

function calculatePipelineParallelVRAM(
  singleGPUBreakdown: InferenceVRAMBreakdown,
  model: Model,
  numGPUs: number
): MultiGPUVRAMBreakdown {
  const layersPerStage = Math.ceil(model.num_hidden_layers / numGPUs)

  // 1. Model weights are EVENLY divided by layers
  //    Each GPU stores ~(num_layers / n_gpus) layers
  const weightsPerGPU = singleGPUBreakdown.modelWeights.div(numGPUs)

  // 2. KV cache must be stored at EVERY stage (not sharded for PP)
  const kvCachePerGPU = singleGPUBreakdown.kvCache // NO division!

  // 3. Activations are SMALLER per stage but need stashing buffers
  const baseActivationsPerGPU = singleGPUBreakdown.activations.div(numGPUs)
  const activationStashingOverhead = baseActivationsPerGPU.mul(0.12) // 10-15% stashing
  const activationsPerGPU = baseActivationsPerGPU.add(activationStashingOverhead)

  // 4. Framework overhead per GPU
  const frameworkOverheadPerGPU = singleGPUBreakdown.frameworkOverhead

  // 5. Pipeline bubble overhead (not in memory, but affects performance)
  const communicationOverhead = new Decimal(0.05).mul(weightsPerGPU) // 5% buffer

  const totalPerGPU = weightsPerGPU
    .add(kvCachePerGPU)
    .add(activationsPerGPU)
    .add(frameworkOverheadPerGPU)
    .add(communicationOverhead)

  return {
    numGPUs,
    strategy: 'pipeline-parallel',
    perGPU: {
      modelWeights: weightsPerGPU,
      kvCache: kvCachePerGPU, // NOT divided!
      activations: activationsPerGPU,
      frameworkOverhead: frameworkOverheadPerGPU,
      total: totalPerGPU,
    },
    replicatedMemory: new Decimal(0), // No replication in PP
    communicationOverhead,
    totalPerGPU,
    utilizationPercent: totalPerGPU.div(gpu.vram_gb).mul(100),
  }
}
```

**Critical difference:** Pipeline parallelism does NOT shard KV cache. Each stage needs full KV cache for its layers. This makes PP less memory-efficient than TP for inference.

### Pattern 3: Interconnect Performance Impact

**What:** NVLink vs PCIe bandwidth affects multi-GPU efficiency
**When to warn:** PCIe-only GPUs with TP degree > 2
**Implementation:**

```typescript
// Source: NVIDIA technical blogs
// https://developer.nvidia.com/blog/scaling-ai-inference-performance-and-flexibility-with-nvidia-nvlink-and-nvlink-fusion/

interface InterconnectSpec {
  type: 'nvlink-4' | 'nvlink-5' | 'pcie-4' | 'pcie-5' | 'none'
  bandwidthGBps: number
  recommendedMaxTPDegree: number
}

const INTERCONNECT_SPECS: Record<string, InterconnectSpec> = {
  'nvlink-4': { type: 'nvlink-4', bandwidthGBps: 900, recommendedMaxTPDegree: 8 },
  'nvlink-5': { type: 'nvlink-5', bandwidthGBps: 1800, recommendedMaxTPDegree: 8 },
  'pcie-4': { type: 'pcie-4', bandwidthGBps: 64, recommendedMaxTPDegree: 2 },
  'pcie-5': { type: 'pcie-5', bandwidthGBps: 128, recommendedMaxTPDegree: 4 },
  'none': { type: 'none', bandwidthGBps: 0, recommendedMaxTPDegree: 1 },
}

function validateInterconnect(
  gpu: GPU,
  numGPUs: number,
  strategy: ShardingStrategy
): { valid: boolean; warning?: string } {
  if (numGPUs === 1) return { valid: true }

  const interconnect = gpu.interconnect || 'none'
  const spec = INTERCONNECT_SPECS[interconnect] || INTERCONNECT_SPECS['none']

  if (strategy === 'tensor-parallel' && numGPUs > spec.recommendedMaxTPDegree) {
    return {
      valid: true,
      warning: `${gpu.name} uses ${spec.type} (${spec.bandwidthGBps} GB/s). Tensor parallelism with ${numGPUs} GPUs may have significant communication overhead. Consider pipeline parallelism or NVLink-equipped GPUs for better performance.`
    }
  }

  if (spec.type === 'none' && numGPUs > 1) {
    return {
      valid: false,
      warning: `${gpu.name} does not support multi-GPU operation (no interconnect).`
    }
  }

  return { valid: true }
}
```

**Key thresholds:**
- NVLink 4/5: TP up to 8 GPUs with minimal overhead
- PCIe 5.0: TP up to 4 GPUs acceptable, performance degrades beyond
- PCIe 4.0: TP only 2 GPUs, recommend PP for larger configs

### Anti-Patterns to Avoid

- **Naive division:** `memory_per_gpu = total / n_gpus` — Ignores replication, communication buffers, and strategy differences
- **Missing interconnect validation:** Allowing TP degree > 4 on PCIe-only GPUs without warning
- **Pipeline parallelism for single node:** PP adds latency overhead, TP is better for single-node multi-GPU
- **Treating all components as shardable:** Embeddings, layer norms, and (for PP) KV cache have special handling

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Communication overhead modeling | Custom all-reduce timing | Use fixed percentages from research (10-15% for TP, 5% for PP) | NCCL performance is hardware/network dependent, impossible to model accurately without profiling |
| Optimal sharding strategy selection | ML-based optimizer to choose TP vs PP | Simple heuristic: single node → TP, multi-node → PP | Real-world choice depends on batch size, sequence length, latency requirements—calculator can't know user's priorities |
| Embedding memory calculation | Parse model architecture to identify embedding layers | Use approximation: `vocab_size * hidden_size * bytes_per_param` | Good enough for estimation, real value requires loading model files |
| Inter-layer dependencies | Analyze which layers can be sharded | Assume standard Transformer architecture (all attention/MLP shardable, embeddings/norms replicated) | Model-specific optimizations (like Megatron's sequence parallelism) are framework-dependent |

**Key insight:** Multi-GPU distribution formulas are well-established in vLLM, DeepSpeed, and Megatron-LM. Use their proven constants rather than trying to derive from first principles.

## Common Pitfalls

### Pitfall 1: Naive Division (Already in PITFALLS.md)

**What goes wrong:** Calculator shows `memory_per_gpu = 80GB / 4 GPUs = 20GB`, but actual usage is 24GB per GPU
**Why it happens:**
- Embeddings replicated (2-5% of model)
- Layer norms replicated (<1%)
- NCCL buffers per peer GPU (200MB × 3 peers = 600MB)
- Communication overhead buffers (10-15%)

**How to avoid:** Use tensor parallelism formula above with explicit replication accounting
**Warning signs:** User reports "4×24GB should fit 80GB model but OOM occurs"

### Pitfall 2: Ignoring Sharding Strategy Differences

**What goes wrong:** Calculator shows same per-GPU memory for TP and PP with same GPU count
**Why it happens:** Treating all multi-GPU as identical, not understanding KV cache replication in PP
**Consequences:** Pipeline parallelism estimate is 20-40% too low (KV cache not divided)

**How to avoid:**
```typescript
// WRONG - same formula for both
const perGPU = total.div(numGPUs)

// RIGHT - strategy-specific
if (strategy === 'tensor-parallel') {
  kvPerGPU = kvCache.div(numGPUs) // Sharded
} else {
  kvPerGPU = kvCache // Replicated across all stages
}
```

**Warning signs:** PP estimates lower than TP for same config (should be higher due to KV replication)

### Pitfall 3: PCIe Multi-GPU Overconfidence

**What goes wrong:** Calculator allows TP degree 8 on PCIe 4.0 GPUs without warning
**Why it happens:** Validating memory fit but not communication bandwidth
**Consequences:** Configuration fits in memory but runs 3-5× slower than expected

**How to avoid:** Validate interconnect against TP degree, show warning when exceeding thresholds
**Warning signs:** User says "multi-GPU is slower than single GPU" (communication bottleneck)

### Pitfall 4: Missing MoE Implications

**What goes wrong:** Mixtral 8x7B on 2 GPUs calculates as if all experts can be split
**Why it happens:** Expert routing requires certain experts on specific GPUs, limits sharding efficiency
**Consequences:** Underestimation for MoE models in multi-GPU (expert load balancing overhead)

**How to avoid:** Add 15-20% overhead multiplier for MoE models in multi-GPU configurations
**Warning signs:** vLLM reports higher VRAM usage for Mixtral than calculator predicts

### Pitfall 5: Visualization Clutter

**What goes wrong:** Showing 8 separate pie charts for 8-GPU configuration
**Why it happens:** Reusing single-GPU visualization pattern for multi-GPU
**Consequences:** User can't compare per-GPU distribution at a glance

**How to avoid:** Use stacked bar chart with one bar per GPU, stacked segments for memory components
**Warning signs:** User feedback "can't see which GPU is the bottleneck"

## Code Examples

### Example 1: Multi-GPU Calculation Engine

```typescript
// src/engines/multi-gpu.ts
import type { Model, GPU } from '@utils/schemas'
import type { InferenceVRAMBreakdown, ShardingStrategy } from './types'
import Decimal from 'decimal.js'

/**
 * Calculate VRAM distribution for multi-GPU configuration
 *
 * @param singleGPUBreakdown - VRAM breakdown for single-GPU baseline
 * @param model - Model configuration
 * @param gpu - GPU specification (for interconnect validation)
 * @param numGPUs - Number of GPUs (1-8)
 * @param strategy - Sharding strategy ('tensor-parallel' or 'pipeline-parallel')
 * @returns Per-GPU VRAM breakdown with replication and overhead
 */
export function calculateMultiGPUVRAM(
  singleGPUBreakdown: InferenceVRAMBreakdown,
  model: Model,
  gpu: GPU,
  numGPUs: number,
  strategy: ShardingStrategy
): MultiGPUVRAMBreakdown {
  // Validate GPU count
  if (numGPUs < 1 || numGPUs > 8) {
    throw new Error('Number of GPUs must be between 1 and 8')
  }

  // Single GPU - no distribution needed
  if (numGPUs === 1) {
    return {
      numGPUs: 1,
      strategy,
      perGPU: singleGPUBreakdown,
      replicatedMemory: new Decimal(0),
      communicationOverhead: new Decimal(0),
      totalPerGPU: singleGPUBreakdown.total,
      utilizationPercent: singleGPUBreakdown.total.div(gpu.vram_gb).mul(100),
    }
  }

  // Choose calculation based on strategy
  if (strategy === 'tensor-parallel') {
    return calculateTensorParallelVRAM(singleGPUBreakdown, model, gpu, numGPUs)
  } else {
    return calculatePipelineParallelVRAM(singleGPUBreakdown, model, gpu, numGPUs)
  }
}
```

### Example 2: Stacked Bar Chart Visualization

```typescript
// src/components/outputs/MultiGPUBreakdownChart.tsx
import type { MultiGPUVRAMBreakdown } from '@engines/types'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useUIStore } from '@store/uiStore'

interface MultiGPUBreakdownChartProps {
  breakdown: MultiGPUVRAMBreakdown
  gpuVRAM: number
}

export function MultiGPUBreakdownChart({ breakdown, gpuVRAM }: MultiGPUBreakdownChartProps) {
  const isDarkMode = useUIStore((s) => s.isDarkMode)

  // Create data array with one entry per GPU
  const data = Array.from({ length: breakdown.numGPUs }, (_, i) => ({
    name: `GPU ${i + 1}`,
    'Model Weights': breakdown.perGPU.modelWeights.toNumber(),
    'KV Cache': breakdown.perGPU.kvCache.toNumber(),
    'Activations': breakdown.perGPU.activations.toNumber(),
    'Framework Overhead': breakdown.perGPU.frameworkOverhead.toNumber(),
    'Communication': breakdown.communicationOverhead.toNumber(),
    'Available': Math.max(0, gpuVRAM - breakdown.totalPerGPU.toNumber()),
  }))

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? '#374151' : '#e5e7eb'} />
          <XAxis
            dataKey="name"
            stroke={isDarkMode ? '#9ca3af' : '#6b7280'}
          />
          <YAxis
            label={{ value: 'VRAM (GB)', angle: -90, position: 'insideLeft' }}
            stroke={isDarkMode ? '#9ca3af' : '#6b7280'}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.95)' : 'rgba(255, 255, 255, 0.95)',
              border: isDarkMode ? '1px solid #374151' : '1px solid #e5e7eb',
              borderRadius: '0.375rem',
            }}
            formatter={(value: number) => `${value.toFixed(2)} GB`}
          />
          <Legend />
          <Bar dataKey="Model Weights" stackId="used" fill="#3b82f6" />
          <Bar dataKey="KV Cache" stackId="used" fill="#10b981" />
          <Bar dataKey="Activations" stackId="used" fill="#f59e0b" />
          <Bar dataKey="Framework Overhead" stackId="used" fill="#8b5cf6" />
          <Bar dataKey="Communication" stackId="used" fill="#ef4444" />
          <Bar dataKey="Available" stackId="used" fill="#d1d5db" opacity={0.3} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

### Example 3: Zustand State Extension

```typescript
// src/store/uiStore.ts - Add multi-GPU state

interface UIState {
  // ... existing state ...

  // Multi-GPU configuration
  numGPUs: number
  shardingStrategy: ShardingStrategy

  // Actions
  setNumGPUs: (numGPUs: number) => void
  setShardingStrategy: (strategy: ShardingStrategy) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // ... existing state ...
      numGPUs: 1,
      shardingStrategy: 'tensor-parallel',

      setNumGPUs: (numGPUs) => set({ numGPUs }),
      setShardingStrategy: (strategy) => set({ shardingStrategy: strategy }),
    }),
    {
      name: 'llmvram-ui-preferences',
      partialize: (state) => ({
        // ... existing persisted state ...
        numGPUs: state.numGPUs,
        shardingStrategy: state.shardingStrategy,
      }),
    },
  ),
)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pure tensor parallelism | Hybrid TP + sequence parallelism | 2023 (Megatron-LM v3) | Further reduces activation memory by sharding LayerNorm along sequence dimension |
| Pipeline parallelism micro-batching | Interleaved pipeline schedules | 2024 (PipeDream-2BW) | Reduces pipeline bubbles from 25% to 10-15% |
| PCIe-only multi-GPU | NVLink/NVSwitch required for efficient TP | 2022-2023 | Consumer GPUs (RTX 4090 with PCIe 4.0) limited to TP=2, datacenter GPUs (H100 with NVLink) scale to TP=8 |
| Static NCCL buffers | Dynamic NCCL buffer sizing | 2024 (NCCL 2.27+) | Reduces small-message latency by 7.6x with symmetric memory |

**Deprecated/outdated:**
- **Model parallelism without pipelining**: Original Megatron-LM (2019) did pure TP without PP, causing high communication overhead for large models. Modern approach uses 3D parallelism (TP × PP × DP).
- **Naive gradient accumulation for PP**: Early pipeline implementations (GPipe) had 40-50% bubble overhead. Current implementations (DeepSpeed, Megatron) achieve 10-15% overhead with interleaved schedules.

**Emerging (not yet for calculators):**
- **Expert parallelism for MoE**: Separate dimension for sharding experts across GPUs (orthogonal to TP/PP)
- **ZeRO-Offload for inference**: Offload inactive layers to CPU memory (impractical for latency-sensitive inference)

## Open Questions

1. **MoE expert placement overhead**
   - What we know: MoE models have additional routing overhead in multi-GPU
   - What's unclear: Exact memory overhead percentage for expert load balancing buffers
   - Recommendation: Use conservative 15-20% multiplier on communication overhead for MoE, validate against vLLM profiling

2. **Sequence parallelism for inference**
   - What we know: Sequence parallelism reduces activation memory by sharding LayerNorm
   - What's unclear: Whether inference frameworks (vLLM, TGI) use sequence parallelism or only training frameworks (DeepSpeed)
   - Recommendation: Research vLLM source code or assume not used for MVP (more conservative estimate)

3. **Mixed precision impact on multi-GPU**
   - What we know: FP8 KV cache reduces memory, FP16 weights with FP32 optimizer states in training
   - What's unclear: Whether mixed precision affects replication overhead (do embeddings stay FP16 while attention goes FP8?)
   - Recommendation: Treat all components as same precision for MVP, add mixed-precision deep dive in future phase

4. **Interconnect validation for custom GPUs**
   - What we know: GPU database has `interconnect` field (nvlink-4, pcie-5, etc.)
   - What's unclear: User may input custom GPU without interconnect specification
   - Recommendation: Default to 'none' for custom GPUs, show warning that multi-GPU requires NVLink/PCIe info

## Sources

### Primary (HIGH confidence)
- [vLLM Parallelism and Scaling Documentation](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) - Tensor parallelism memory distribution patterns
- [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/) - Pipeline parallelism activation stashing
- [NVIDIA Megatron-LM Parallelisms Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html) - Tensor parallelism implementation details
- [Megatron-LM Paper (arXiv:1909.08053)](https://arxiv.org/abs/1909.08053) - Original model parallelism formulas

### Secondary (MEDIUM confidence)
- [NVLink vs PCIe for AI Workloads](https://www.hyperstack.cloud/blog/case-study/nvlink-vs-pcie-whats-the-difference-for-ai-workloads) - Bandwidth comparisons (1800 GB/s vs 128 GB/s)
- [NVIDIA NVLink Technical Blog](https://developer.nvidia.com/blog/scaling-ai-inference-performance-and-flexibility-with-nvidia-nvlink-and-nvlink-fusion/) - Multi-GPU scaling efficiency (90-95% with NVLink, 60-70% with PCIe)
- [NCCL 2.27 Technical Blog](https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/) - Communication buffer sizing (symmetric memory reduces latency 7.6x)
- [Recharts Stacked Bar Chart Documentation](https://recharts.github.io/en-US/examples/StackedBarChart/) - Visualization pattern implementation

### Tertiary (LOW confidence - needs validation)
- [LLM VRAM Calculator Comparisons](https://apxml.com/tools/vram-calculator) - Other calculators use simple division (validates need for better approach)
- [BentoML Multi-GPU Guide](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism) - General multi-GPU patterns
- [Medium articles on tensor parallelism](https://medium.com/@kaige.yang0110/vllm-throughput-optimization-1-basic-of-vllm-parameters-c39ace00a519) - Community knowledge (useful but unverified)

## Metadata

**Confidence breakdown:**
- Tensor parallelism formula: MEDIUM - Derived from multiple sources (vLLM docs, Megatron paper, community articles) but not officially documented with exact percentages. Replication overhead (10-15%) and embedding replication (2-5%) are widely cited but need validation against real profiling.
- Pipeline parallelism formula: MEDIUM - DeepSpeed documentation provides implementation details but not exact memory formulas. Activation stashing overhead (10-15%) inferred from academic papers, not official docs.
- Interconnect bandwidth thresholds: HIGH - NVIDIA official blogs provide exact bandwidth numbers (NVLink 5.0 = 1800 GB/s, PCIe 5.0 = 128 GB/s). Scaling efficiency percentages (90-95% vs 60-70%) verified across multiple sources.
- Visualization patterns: HIGH - Recharts official documentation with examples. Stacked bar chart pattern is standard and well-documented.

**Limitations:**
- No access to vLLM or DeepSpeed source code for exact formulas (relied on documentation and papers)
- NCCL buffer sizing is hardware/network dependent, used conservative estimates from community sources
- MoE multi-GPU overhead is under-documented (15-20% estimate is educated guess, needs validation)

**Research date:** 2026-02-09
**Valid until:** 60 days (slower-moving domain - multi-GPU patterns stable since Megatron-LM 2019, incremental improvements)

**Next steps for validation:**
1. Profile vLLM with Llama 3 70B on 2×H100 (TP=2) to measure actual replication overhead
2. Compare calculator estimates against vLLM `--tensor-parallel-size` memory usage
3. Validate NCCL buffer overhead with `nvidia-smi` monitoring during multi-GPU inference
4. Test pipeline parallelism formulas with DeepSpeed inference (less common for inference, might skip for MVP)
