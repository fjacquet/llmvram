---
phase: 04-multi-gpu-support
plan: 01
subsystem: engines
tags: [multi-gpu, tensor-parallel, pipeline-parallel, interconnect, calculation-engine]
requires: [inference-engine, kv-cache, quantization-engine]
provides: [calculateMultiGPUVRAM, resolveInterconnect, validateInterconnect]
affects: []
tech-stack:
  added: []
  patterns: [pure-functions, decimal-precision, tdd-red-green-refactor]
key-files:
  created:
    - src/engines/multi-gpu.ts
    - src/engines/multi-gpu.test.ts
  modified:
    - src/engines/types.ts
    - src/engines/constants.ts
decisions: []
metrics:
  duration_minutes: 5.4
  completed_date: 2026-02-09
---

# Phase 04 Plan 01: Multi-GPU VRAM Calculation Engine Summary

**One-liner:** Pure calculation engine for tensor-parallel and pipeline-parallel VRAM distribution with interconnect validation and MoE support

## What Was Built

Built the computational foundation for multi-GPU VRAM distribution across 2-8 GPUs using tensor parallelism (TP) or pipeline parallelism (PP) strategies. The engine takes a single-GPU VRAM breakdown and distributes memory components according to strategy-specific rules, accounting for replicated components, communication overhead, and interconnect capabilities.

### Core Functions

**calculateMultiGPUVRAM**: Main entry point that routes to TP or PP calculation based on strategy. Validates numGPUs range (1-8) and provides passthrough for single GPU with zero overhead.

**calculateTensorParallelVRAM** (internal): Shards model weights (except embeddings/layer norms which are replicated), divides KV cache and activations across GPUs, adds NCCL buffers for peer communication, applies 12% communication overhead (15% extra for MoE).

**calculatePipelineParallelVRAM** (internal): Divides layers across GPUs (weights split evenly), keeps full KV cache per GPU (each GPU needs full context for its layers), applies 12% activation stashing overhead, 5% communication overhead (15% extra for MoE), no NCCL buffers.

**resolveInterconnect**: Maps GPU interconnect strings to standardized engine types (nvlink-4/5, pcie-4/5, none). Handles generic 'nvlink' → nvlink-4, AMD 'infinity-fabric' → pcie-5, Apple 'unified' → none, fallback to pcie-5/4 based on tier.

**validateInterconnect**: Checks if interconnect supports requested multi-GPU config. Returns invalid for 'none' with numGPUs > 1, warns when TP degree exceeds recommended max (e.g., PCIe 4.0 with 4 GPUs), always valid for single GPU.

### Types Added

```typescript
ShardingStrategy: 'tensor-parallel' | 'pipeline-parallel'
InterconnectType: 'nvlink-4' | 'nvlink-5' | 'pcie-4' | 'pcie-5' | 'none'
InterconnectSpec: { type, bandwidthGBps, recommendedMaxTPDegree }
MultiGPUVRAMBreakdown: { numGPUs, strategy, perGPU, replicatedMemory, totalPerGPU, utilizationPercent, singleGPUBaseline }
InterconnectValidation: { valid, warning, interconnect }
```

### Constants Added

- `INTERCONNECT_SPECS`: Bandwidth and recommended TP limits for each interconnect type
- `NCCL_BUFFER_PER_PEER_GB`: 0.2 GB per peer GPU (TP only)
- `EMBEDDING_WEIGHT_FRACTION`: 3% of model weights (replicated in TP)
- `TP_COMMUNICATION_OVERHEAD`: 12%
- `PP_COMMUNICATION_OVERHEAD`: 5%
- `PP_ACTIVATION_STASHING_OVERHEAD`: 12%
- `MOE_MULTI_GPU_OVERHEAD`: 15% extra for MoE models

## How It Works

### Tensor Parallelism (TP)

1. Calculate replicated memory: layer norms (num_layers * 2 * hidden_size * 4 bytes) + embeddings (~3% of weights)
2. Shardable weights = total weights - replicated
3. Weights per GPU = (shardable / numGPUs) + replicated
4. KV cache divided by numGPUs (sharded across GPUs)
5. Activations divided by numGPUs
6. NCCL buffers = 0.2 GB * (numGPUs - 1) added to framework overhead
7. Communication overhead = weightsPerGPU * 0.12 * (1.15 if MoE else 1.0)

**Example:** Llama 70B GPTQ on 4x H100
- Single GPU: 41.83 GB total (39.12 weights + 1.25 KV + 0.46 activations + 1.0 framework)
- Replicated: ~1.18 GB (0.0049 layer norms + 1.17 embeddings)
- Weights per GPU: (39.12 - 1.18) / 4 + 1.18 = 10.67 GB
- KV per GPU: 1.25 / 4 = 0.31 GB
- NCCL: 0.6 GB (3 peers)
- Comm overhead: 10.67 * 0.12 = 1.28 GB
- Total per GPU: ~13.96 GB (vs 41.83 single GPU)

### Pipeline Parallelism (PP)

1. Weights divided evenly (no replication, layers split)
2. KV cache NOT divided (each GPU needs full cache for its layers)
3. Activations divided with 12% stashing overhead
4. No NCCL buffers (point-to-point communication only)
5. Communication overhead = weightsPerGPU * 0.05 * (1.15 if MoE else 1.0)

**Example:** Same Llama 70B GPTQ on 4 GPUs
- Weights per GPU: 39.12 / 4 = 9.78 GB
- KV per GPU: 1.25 GB (NOT divided!)
- Activations per GPU: 0.46 / 4 * 1.12 = 0.13 GB
- Comm overhead: 9.78 * 0.05 = 0.49 GB
- Total per GPU: ~12.64 GB

**Key insight:** For this scenario, TP uses MORE memory per GPU than PP (13.96 vs 12.64 GB) because the KV cache is relatively small (~1.25 GB). TP pays for weight replication (~1.18 GB), NCCL buffers (0.6 GB), and higher comm overhead (12% vs 5%) which outweighs the benefit of dividing the KV cache. The tradeoff depends on KV cache size (sequence length, batch size) vs model size.

## Deviations from Plan

None - plan executed exactly as written. All specified types, constants, functions, and test cases were implemented according to the TDD methodology.

## Testing

**Test coverage:** 25 tests, all passing

### Test categories:

1. **Tensor Parallelism** (3 tests): Llama 70B breakdown verification, MoE overhead, Decimal instances
2. **Pipeline Parallelism** (2 tests): Llama 70B breakdown, KV cache behavior vs TP
3. **Single GPU** (2 tests): Passthrough with zero overhead, strategy-independent
4. **Edge Cases** (3 tests): numGPUs < 1 throws, numGPUs > 8 throws, 8 GPUs max works
5. **Interconnect Resolution** (9 tests): All mapping scenarios (nvlink-4/5, generic nvlink, pcie-4/5, infinity-fabric, unified, undefined with different tiers)
6. **Interconnect Validation** (6 tests): Valid/warning scenarios, 'none' invalid for multi-GPU, single GPU always valid, PP vs TP warning differences

### Key test insights:

- Replicated memory is always > 0 and < 5% of model weights for TP
- PP has zero replicated memory (layers split, not sharded)
- MoE communication overhead is exactly 15% higher than dense equivalent
- TP and PP produce different per-GPU totals (TP not always more efficient)
- All Decimal values used for precision throughout
- Component sums equal totalPerGPU (no rounding errors)

## Next Steps

**Integration points:**
- Phase 04 Plan 02 will integrate this engine into Zustand store
- Phase 04 Plan 03 will add UI components for multi-GPU input and visualization
- Future: Consider supporting hybrid TP+PP strategies for very large models

## Dependencies

**Runtime:**
- `src/engines/inference.ts`: Provides InferenceVRAMBreakdown as input
- `src/engines/constants.ts`: Uses BYTES_PER_GB for replicated memory calculation
- `@utils/schemas`: Uses Model and GPU types

**Development:**
- Followed existing engine patterns from inference.ts, kv-cache.ts
- Used same test fixtures as inference.test.ts for consistency

## Self-Check: PASSED

**Files created:**
- FOUND: src/engines/multi-gpu.ts
- FOUND: src/engines/multi-gpu.test.ts

**Files modified:**
- FOUND: src/engines/types.ts (added 5 types)
- FOUND: src/engines/constants.ts (added 8 constants)

**Commit:**
- FOUND: af6e14b feat(04-01): implement multi-GPU VRAM calculation engine

All specified outputs verified present and correct.
