# Roadmap: LLM VRAM Calculator

## Overview

This roadmap transforms the LLM VRAM Calculator from concept to functional browser-based tool across five phases. Starting with project infrastructure and curated data, we build calculation engines for inference and multi-GPU scenarios, wrap them in a responsive UI with visualization, and finish with sharing capabilities. Each phase delivers verifiable user value while avoiding common pitfalls around quantization overhead, KV cache scaling, and multi-GPU memory distribution.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation & Data** - Project infrastructure and static databases
- [ ] **Phase 2: Inference Engine** - Core calculation logic with Web Workers
- [ ] **Phase 3: Core UI** - Basic calculator interface with visualization
- [ ] **Phase 4: Multi-GPU Support** - Multi-GPU calculation and distribution
- [ ] **Phase 5: Sharing & Comparison** - URL persistence and side-by-side comparison

## Phase Details

### Phase 1: Foundation & Data
**Goal**: Project infrastructure and static databases ready for development
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07
**Success Criteria** (what must be TRUE):
  1. Developer can run `npm install && npm run dev` and see a basic React app with TypeScript strict mode
  2. Biome linting runs with zero errors on all code
  3. Vitest test suite runs with passing tests for data schemas and validation
  4. Model database includes 30+ popular models with accurate parameter counts, architecture types, and layer configurations
  5. GPU database includes 15+ GPUs (NVIDIA datacenter/consumer, AMD MI300X, Apple Silicon) with VRAM, bandwidth, and FLOPS specs
**Plans**: 4 plans in 3 waves

Plans:
- [ ] 01-01-PLAN.md — Project infrastructure and Zod schemas (Wave 1)
- [ ] 01-02-PLAN.md — GPU database with 15+ entries (Wave 2)
- [ ] 01-03-PLAN.md — Model database with 30+ entries (Wave 2)
- [ ] 01-04-PLAN.md — Data refresh scripts and verification (Wave 3)

### Phase 2: Inference Engine
**Goal**: Accurate VRAM calculations for all model types and quantization formats
**Depends on**: Phase 1
**Requirements**: INFRA-07, INFER-01, INFER-02, INFER-03, INFER-04, INFER-05, INFER-06, INFER-07, INFER-08, INFER-09
**Success Criteria** (what must be TRUE):
  1. Engine correctly calculates VRAM for FP32, FP16, BF16, INT8, INT4, NF4, GPTQ, AWQ, GGUF quantized models with proper overhead (1.1-1.3x multipliers)
  2. KV cache calculation accounts for GQA/MQA architectures by factoring n_kv_heads/n_heads ratio
  3. MoE models (Mixtral, DeepSeek) calculate ALL expert weights in VRAM, not just active parameters
  4. Performance estimation (tokens/sec, TTFT) reflects GPU compute capability and memory bandwidth using roofline model
  5. Heavy calculations run in Web Worker without blocking UI, with graceful fallback to synchronous execution
**Plans**: TBD

Plans:
- TBD after planning phase

### Phase 3: Core UI
**Goal**: Users can calculate and visualize VRAM for single-GPU configurations
**Depends on**: Phase 2
**Requirements**: VIZ-01, VIZ-02, VIZ-03, VIZ-06, VIZ-07
**Success Criteria** (what must be TRUE):
  1. User can select a model from curated database or enter custom specs (parameter count, layers, hidden size, attention heads)
  2. User can select a GPU from curated database or enter custom VRAM amount
  3. User can choose quantization format and adjust sequence length/batch size to see VRAM breakdown update in real-time
  4. Memory breakdown chart shows weights, KV cache, activations, and overhead as visual segments with exact values
  5. Fit indicator clearly shows if model fits on GPU with percentage utilization and color-coded status
  6. When model doesn't fit, recommendations suggest actionable changes (lower quantization, reduce context length, add GPUs)
  7. UI is responsive on mobile, tablet, and desktop with readable charts on all screen sizes
  8. User can toggle dark mode and preference persists across sessions
**Plans**: TBD

Plans:
- TBD after planning phase

### Phase 4: Multi-GPU Support
**Goal**: Users can calculate VRAM for multi-GPU configurations with sharding strategies
**Depends on**: Phase 3
**Requirements**: MGPU-01, MGPU-02, MGPU-03, MGPU-04, MGPU-05, MGPU-06
**Success Criteria** (what must be TRUE):
  1. User can select number of GPUs (1-8) and see per-GPU VRAM calculation accounting for tensor parallelism distribution
  2. Calculation includes communication overhead (10-20%) and memory replication for embeddings and layer norms
  3. User can select interconnect type (NVLink, PCIe) with visual indication of performance impact
  4. User can choose sharding strategy (Tensor Parallel, Pipeline Parallel) with memory distribution explanation
  5. Visualization shows memory distribution across GPUs with per-GPU utilization percentage
**Plans**: TBD

Plans:
- TBD after planning phase

### Phase 5: Sharing & Comparison
**Goal**: Users can share configurations and compare scenarios
**Depends on**: Phase 4
**Requirements**: VIZ-04, VIZ-05
**Success Criteria** (what must be TRUE):
  1. Current configuration is saved to URL hash automatically and shareable link works when opened in new browser
  2. User can add 2-3 configurations to comparison view and see VRAM differences side-by-side
  3. Comparison clearly highlights differences between configurations (quantization, GPU, context length)
**Plans**: TBD

Plans:
- TBD after planning phase

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Data | 0/TBD | Not started | - |
| 2. Inference Engine | 0/TBD | Not started | - |
| 3. Core UI | 0/TBD | Not started | - |
| 4. Multi-GPU Support | 0/TBD | Not started | - |
| 5. Sharing & Comparison | 0/TBD | Not started | - |
