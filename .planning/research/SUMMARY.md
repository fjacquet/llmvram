# Project Research Summary

**Project:** LLM VRAM Calculator
**Domain:** Browser-based technical calculator (LLM memory estimation)
**Researched:** 2026-02-09
**Confidence:** MEDIUM

## Executive Summary

LLM VRAM calculators are specialized technical tools that estimate GPU memory requirements for running and training large language models. The domain requires precision arithmetic, complex calculation engines, and clear visual presentation of memory breakdowns. Success depends on calculation accuracy (users trust exact numbers) and handling edge cases like quantization overhead, MoE architectures, and multi-GPU sharding.

The recommended approach uses React 19 + TypeScript for UI, Web Workers for non-blocking calculations, and Decimal.js for precision arithmetic. The architecture isolates calculation engines (pure functions) from UI concerns, enabling thorough testing and future flexibility. Static JSON databases for models/GPUs keep the tool client-side only. The sister project (raidy) uses similar patterns, reducing cognitive load for shared maintenance.

Critical risks center on calculation accuracy. Quantization overhead (10-30% underestimation if ignored), KV cache scaling (2-8x overestimation without GQA/MQA support), and MoE parameter confusion (4x underestimation) are the top three pitfalls. Mitigation requires formula validation against real implementations (vLLM, HuggingFace) and explicit handling of modern architecture variants. A phased rollout starting with inference-only MVP allows validation before adding training complexity.

## Key Findings

### Recommended Stack

React 19 + TypeScript with strict mode provides the foundation. Vite handles builds with fast HMR. Zustand manages state with minimal boilerplate. Tailwind CSS accelerates UI development with utility classes. Recharts provides declarative charting for VRAM breakdowns. Biome replaces ESLint/Prettier for 10-100x faster linting. Vitest enables Jest-compatible testing with Vite integration.

**Core technologies:**
- **React 19 + TypeScript 5.8+**: Component-based UI with compile-time safety for calculation accuracy
- **Vite 7.x**: Fast build tool with native ESM and tree-shaking (verify version - may need 5.x or 6.x)
- **Zustand 5.x**: Lightweight state management (1kb) with selector-based subscriptions for calculator state
- **Decimal.js 10.x**: Arbitrary-precision arithmetic (CRITICAL - prevents floating-point errors in VRAM calculations)
- **Zod 3.x**: Runtime validation for model/GPU JSON schemas with TypeScript type generation
- **Recharts 2.x**: React-native charting for VRAM breakdown visualizations
- **Web Workers**: Offload heavy calculations to prevent UI blocking
- **Biome 1.x**: All-in-one linting/formatting (replaces ESLint + Prettier)

**Critical dependencies:**
- Decimal.js is non-negotiable (VRAM calculations involve large numbers where precision errors compound)
- Web Workers required for responsive UI during complex calculations
- TypeScript strict mode essential (catches numerical edge cases at compile time)

**Version verification needed:**
- Vite 7.x may not exist (training data showed 5.x) - verify current stable version
- Tailwind CSS 4.x was early release - check stability before using
- React 19 stability - verify production readiness

### Expected Features

Research identified clear feature tiers based on competitive analysis (apxml.com/tools/vram-calculator) and domain expectations.

**Must have (table stakes):**
- Model selection database with parameter specs
- GPU selection database with VRAM specs
- Quantization format selection (FP16, FP32, INT8, INT4, NF4, GPTQ, AWQ)
- Inference VRAM calculation (weights, KV cache, activations, overhead)
- VRAM usage breakdown visualization
- Basic input parameters (context length, batch size)
- Custom model input (power users with unreleased models)
- Multi-GPU support (device count, memory distribution)
- Fine-tuning VRAM estimation (full fine-tuning, LoRA, QLoRA)

**Should have (competitive differentiation):**
- KV cache quantization options (FP16/FP8/INT8 cache precision)
- Optimization framework presets (Unsloth, DeepSpeed ZeRO, PEFT)
- Advanced memory breakdown (separate activations, gradients, optimizer states)
- Export/share configurations (URL params, JSON export)
- Fine-tuning method comparison (side-by-side Full vs LoRA vs QLoRA)
- Gradient accumulation calculator

**Defer (v2+):**
- Performance estimation (tokens/sec, TTFT) - requires benchmarking infrastructure
- Multi-GPU interconnect awareness (NVLink vs PCIe efficiency) - high complexity
- CPU/NVMe offloading estimation - dependent on performance estimation
- Training cost estimation - requires performance data + cloud pricing integration
- Community benchmarks - high complexity, moderation burden
- Energy/carbon footprint - low value compared to alternatives

**Anti-features (explicitly avoid):**
- Model training/inference execution (out of scope, use vLLM/Ollama instead)
- Model downloading/hosting (infrastructure burden, link to Hugging Face)
- User accounts/authentication (stateless browser tool, localStorage for favorites)
- AI-powered recommendations (rule-based suggestions clearer and more predictable)

### Architecture Approach

The architecture isolates calculation logic from UI concerns through pure function engines, Web Workers for async computation, and Zustand for reactive state management. This enables thorough testing of calculation accuracy independent of React rendering.

**Major components:**

1. **Calculation Engines** (`src/engines/`) - Pure functions with zero dependencies. Separate modules for inference, fine-tuning, multi-GPU, each exporting typed input/output interfaces. Enables unit testing without UI and future server-side execution.

2. **Web Worker Pool** (`src/workers/`) - Pool of workers (size = CPU cores) that import engines and execute calculations off main thread. Prevents UI blocking during complex calculations. Graceful fallback to synchronous if workers unavailable.

3. **State Management** (`src/store/`) - Zustand store organized in slices (inputs, results, config). Middleware intercepts input changes and triggers debounced calculation via worker pool. Selector-based subscriptions prevent unnecessary re-renders.

4. **Component Architecture** (`src/components/`) - Organized by responsibility: `inputs/` (ModelSelector, GPUSelector), `outputs/` (VRAMBreakdown, PerformanceMetrics), `common/` (reusable UI), `layout/` (Calculator shell). Each component subscribes to minimal store state via selectors.

5. **Static Data** (`src/data/`) - Model and GPU specifications in JSON with TypeScript schemas. Zod validates at load time, generates types for compile-time safety. Structured for tree-shaking (separate files per vendor).

**Data flow:**
```
User Input → Component → Store Update → Middleware → Worker Pool → Engine (pure calc)
→ Worker Result → Store Update → Component Re-render → UI Update
```

Key characteristics: Unidirectional flow, async calculations, reactive updates, debounced inputs (300ms).

**Build order implications:**
1. Data schemas must come first (everything depends on TypeScript interfaces)
2. Engines can be built in parallel once schemas exist (pure functions, no dependencies)
3. Workers wrap engines (depends on engines being complete)
4. Components depend on store + hooks (can be built in parallel if hook interfaces defined)

### Critical Pitfalls

Research identified 15 pitfalls across three severity levels. The top five are project-killers if not addressed in MVP.

1. **Quantization Overhead Ignored** - Assuming 4-bit quantization = exactly 0.5 bytes/param leads to 10-30% underestimation. GPTQ/AWQ add metadata, scale factors, padding. Use 1.1-1.3x multiplier. CRITICAL for Phase 1 MVP - affects all quantized calculations.

2. **KV Cache Quadratic Scaling Assumed** - Using `KV = 2 * layers * d_model * seq_len` formula overestimates GQA/MQA models by 2-8x. Must factor in `n_kv_heads / n_heads` ratio. Llama 3 70B uses 8 KV heads / 64 query heads = 0.125x. CRITICAL for Phase 1 - affects all modern models.

3. **MoE Active Parameters Miscalculation** - Confusing "13B active" with "13B loaded" for Mixtral 8x7B leads to 4x underestimation. ALL expert weights must be in VRAM (46.7B), only subset activated per token. CRITICAL for Phase 2 - requires separate logic from dense models.

4. **Multi-GPU Memory Split Naive** - Dividing total memory by GPU count ignores replication overhead (embeddings, layernorms replicated on all GPUs), pipeline activation stashing, and NCCL communication buffers. Leads to 10-20% underestimation. CRITICAL for Phase 4 - multi-GPU needs dedicated calculation path.

5. **Fine-tuning Optimizer State Multiplier Wrong** - Using fixed "AdamW = 8 bytes per param" for LoRA leads to 10-100x overestimation. Only adapter weights have optimizer states (1-2% of params for rank 16-64). Full fine-tuning requires 13x model weight (model + gradients + optimizer). CRITICAL for Phase 3 - must distinguish trainable vs frozen params.

**Phase-specific warnings:**
- Phase 1 MVP: Must handle quantization overhead (#1) and KV cache scaling (#2) or core functionality is broken
- Phase 2 Architecture: Add MoE support (#3), context length optimizations, GGUF variants
- Phase 3 Training: Optimizer state calculation (#5), activation memory formulas, gradient checkpointing
- Phase 4 Multi-GPU: Sharding overhead (#4), communication buffers, TP/PP strategies
- Phase 5+: Framework overhead, batch size scaling, precision conversion

## Implications for Roadmap

Based on combined research, recommended phase structure prioritizes calculation accuracy and incremental validation. Each phase delivers user value while building toward complete feature set.

### Phase 1: Core Infrastructure & Data
**Rationale:** Establishes foundation for all calculations. Pure functions enable testing without UI complexity. Static data allows frontend-only deployment.

**Delivers:**
- TypeScript interfaces for all data models (ModelSpec, GPUSpec, CalculatorInputs, Results)
- Static JSON databases (models.json with 20-30 popular models, gpus.json with 15-20 common GPUs)
- Zod schemas with runtime validation
- Inference calculation engine (weights, KV cache, activations, overhead)
- Unit tests for calculation accuracy

**Avoids:**
- Building UI before calculation logic is validated (saves rework)
- Database backend complexity (JSON in version control)

**Duration:** 1-2 weeks
**Complexity:** Medium (data curation effort, formula validation)
**Research needed:** No (well-established patterns)

---

### Phase 2: MVP Calculator (Inference)
**Rationale:** Delivers minimum viable product - users can answer "will this model fit on my GPU?" Proves architecture with vertical slice before expanding scope.

**Delivers:**
- Basic UI layout (Calculator shell, Header, Footer)
- Input components (ModelSelector, GPUSelector, QuantizationPicker, ParameterInputs)
- Output component (VRAMBreakdown with Recharts visualization)
- Store setup (Zustand with inputs/results slices)
- Basic calculation flow (synchronous initially, workers in Phase 3)
- Custom model input for power users

**Addresses features:**
- Model selection database (table stakes)
- GPU selection database (table stakes)
- Quantization format selection (table stakes)
- Inference VRAM calculation (table stakes)
- VRAM usage breakdown (table stakes)
- Clear result display (table stakes)

**Avoids pitfalls:**
- Quantization overhead (#1) - Apply 1.1-1.3x multipliers in inference engine
- KV cache scaling (#2) - Expose n_kv_heads parameter, calculate GQA/MQA correctly

**Duration:** 2-3 weeks
**Complexity:** Medium (UI implementation, state wiring)
**Research needed:** No (standard React patterns, calculator UIs well-documented)

---

### Phase 3: Worker Infrastructure & Optimization
**Rationale:** Adds performance layer without changing UI contracts. Prevents blocking on complex calculations (128K context, 70B+ models).

**Delivers:**
- Web Worker implementation (calculation.worker.ts)
- Worker pool management (size = CPU cores)
- Updated useCalculation hook (async worker-based)
- Calculation middleware (debouncing, worker orchestration)
- Loading states and error boundaries

**Uses stack:**
- Web Workers (native browser API)
- TypeScript worker interfaces (typed postMessage)
- Zustand middleware for side effects

**Duration:** 1-2 weeks
**Complexity:** Low-Medium (worker API straightforward, testing async flows)
**Research needed:** No (Web Worker patterns well-documented)

---

### Phase 4: Training Support
**Rationale:** Completes table stakes features. Training use case distinct from inference (different memory profile). Many users need fine-tuning estimation.

**Delivers:**
- Fine-tuning calculation engine (finetuning.ts)
- Training method selection (Full, LoRA, QLoRA)
- LoRA configuration inputs (rank, alpha)
- Optimizer state calculation (AdamW, SGD)
- Gradient accumulation calculator
- Dataset size input (samples, tokens per sample, epochs)
- Training-specific VRAM breakdown (model + gradients + optimizer + activations)

**Addresses features:**
- Fine-tuning VRAM estimation (table stakes)
- Gradient accumulation calculator (differentiator)
- Fine-tuning method comparison (differentiator)

**Avoids pitfalls:**
- Optimizer state multiplier (#5) - Separate logic for Full FT (13x model) vs LoRA (1.2-1.5x)
- Activation memory (#10) - Account for MLP expansion (4x hidden_size)
- MoE parameters (#3) - Special handling for Mixtral fine-tuning (all experts loaded)

**Duration:** 2-3 weeks
**Complexity:** High (complex formulas, many edge cases)
**Research needed:** CONSIDER - Fine-tuning formulas may need validation against HuggingFace accelerate, DeepSpeed docs

---

### Phase 5: Multi-GPU Support
**Rationale:** Common production deployment pattern. Enables large model calculations. Requires distinct logic from single-GPU.

**Delivers:**
- Multi-GPU calculation engine (multigpu.ts)
- GPU count input (1-16)
- Sharding strategy selection (Tensor Parallel, Pipeline Parallel, Hybrid)
- Per-GPU memory calculation (accounts for replication overhead)
- Multi-GPU visualization (memory distribution across GPUs)
- Interconnect notes (NVLink recommended for >4 GPUs)

**Addresses features:**
- Multi-GPU support (table stakes completion)
- Multi-GPU visualization (differentiator)

**Avoids pitfalls:**
- Naive memory split (#4) - Add 10-15% overhead for replicated components
- Communication overhead (#11) - Warn when bandwidth insufficient for model size

**Implements architecture:**
- Separate calculation engine (multigpu.ts imports inference.ts, composes results)
- Worker pool enables parallel calculation for comparison scenarios

**Duration:** 1-2 weeks
**Complexity:** Medium (overhead constants need research, visualization complexity)
**Research needed:** CONSIDER - Multi-GPU strategies may need research on DeepSpeed ZeRO, Megatron-LM papers for accurate overhead constants

---

### Phase 6: Differentiation Features
**Rationale:** Polish and competitive advantages. Lower complexity, high user value. These features distinguish from basic calculators.

**Delivers:**
- KV cache quantization options (FP16/FP8/INT8 cache, separate from model quantization)
- Optimization framework presets (Unsloth, DeepSpeed ZeRO-2/3, PEFT, FSDP templates)
- Advanced memory breakdown (separate activations, gradients, optimizer, temporary buffers)
- Export/share configurations (URL params for sharing, JSON export for saving)
- Fine-tuning method comparison view (side-by-side Full vs LoRA vs QLoRA)

**Addresses features:**
- KV cache quantization (differentiator) - Low complexity, high value for long-context
- Optimization presets (differentiator) - Medium complexity, high user value
- Advanced breakdown (differentiator) - Medium complexity, educational value
- Export/share (differentiator) - Low complexity, collaboration enabler
- Method comparison (differentiator) - Medium complexity, decision support

**Duration:** 2-3 weeks
**Complexity:** Low-Medium (mostly UI/UX work, calculations reuse existing engines)
**Research needed:** No (standard patterns, UI polish)

---

### Phase 7: Advanced Features (Optional)
**Rationale:** If time permits. High complexity or lower ROI. Can defer to v2.

**Potential features:**
- Performance estimation (tokens/sec, TTFT, throughput) - Requires benchmarking database
- CPU/NVMe offloading estimation - Depends on performance estimation
- Training cost estimation - Requires performance data + cloud pricing integration
- Comparative analysis - Side-by-side 2-3 configurations
- Memory optimization suggestions - Rules engine for "try 4-bit quantization" hints

**Complexity:** High (benchmarking infrastructure, validation burden)
**Research needed:** YES - Performance estimation requires extensive research

---

### Phase Ordering Rationale

**Dependency-driven:**
1. Phase 1 (Infrastructure) → Phase 2 (MVP) → Phase 3 (Workers) is strict dependency chain
2. Phase 4 (Training) and Phase 5 (Multi-GPU) can run in parallel if resources allow (separate engines)
3. Phase 6 (Differentiation) requires Phase 2-5 complete (reuses all engines)

**Risk-driven:**
- Phase 2 proves core value before expanding scope
- Phase 3 adds performance without changing contracts (low risk refactor)
- Phase 4-5 are independent feature expansions (failure doesn't affect other phases)

**Pitfall mitigation:**
- Critical pitfalls (#1, #2) addressed in Phase 1-2 foundation
- Training pitfalls (#5) isolated to Phase 4
- Multi-GPU pitfalls (#4) isolated to Phase 5
- Each phase can be validated against real implementations before next phase

**User value prioritization:**
- Phase 2 delivers to hobbyists (primary persona): "Will X fit on my GPU?"
- Phase 4 adds data scientists (secondary persona): "Can I fine-tune this?"
- Phase 5 adds ML engineers (secondary persona): "How should I configure multi-GPU?"
- Phase 6 adds competitive polish

### Research Flags

**Standard patterns (skip `/gsd:research-phase`):**
- **Phase 1:** Data modeling, JSON schemas, TypeScript interfaces are well-documented
- **Phase 2:** React calculator UIs are common, many examples available
- **Phase 3:** Web Worker patterns well-established, browser API stable
- **Phase 6:** UI polish and export features are straightforward implementations

**May need targeted research:**
- **Phase 4 (Training):** Fine-tuning memory formulas should be validated against HuggingFace accelerate docs, DeepSpeed papers. Complexity: LoRA/QLoRA/Full FT have different memory profiles that need verification. Consider `/gsd:research-phase` if formulas uncertain.

- **Phase 5 (Multi-GPU):** Multi-GPU overhead constants (replication, communication buffers) may need research on Megatron-LM, DeepSpeed ZeRO documentation. Tensor Parallel vs Pipeline Parallel efficiency factors need validation. Consider `/gsd:research-phase` if overhead estimates are rough.

**Definitely needs research:**
- **Phase 7 (Performance):** Requires extensive benchmarking database research. Tokens/sec depends on GPU architecture, model size, batch size, quantization. Would need `/gsd:research-phase` with performance benchmarking focus.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | Core choices (React, TypeScript, Zustand) are solid. Version specifics need verification (Vite 7 may not exist, Tailwind 4 early release). Decimal.js and Web Workers are proven choices for domain. |
| Features | HIGH | Competitive analysis (apxml.com) provided clear feature tiers. Table stakes vs differentiators well-defined. MVP scope clear. Anti-features help avoid scope creep. |
| Architecture | MEDIUM-HIGH | Patterns are established (engine isolation, worker pools, Zustand state). raidy alignment reduces risk. Build order is logical. Some uncertainty on worker pool implementation details. |
| Pitfalls | MEDIUM | Comprehensive list from training data. Top pitfalls (#1-5) are well-documented in community. Lower severity pitfalls (#6-15) less validated. Need real-world testing against vLLM/HuggingFace to confirm accuracy. |

**Overall confidence:** MEDIUM-HIGH

Research provides solid foundation for roadmap. Core technology choices are proven. Feature prioritization is clear. Architecture patterns are established. Main uncertainty is calculation accuracy for edge cases (MoE, multi-GPU overhead constants) - these can be validated during implementation.

### Gaps to Address

**Version verification (before Phase 1):**
- Vite 7.x may not be released (training data showed 5.x latest) - verify current stable version, may need 5.x or 6.x
- Tailwind CSS 4.x was early release in training - check stability, may need 3.x if v4 unstable
- React 19 was newly released - verify production readiness, check for major issues
- Zustand 5.x version check (4.x was latest in training)

**Formula validation (during Phase 4-5):**
- Fine-tuning memory formulas - validate against HuggingFace transformers memory anatomy docs
- Multi-GPU overhead constants - validate against Megatron-LM, DeepSpeed papers for replication factors
- MoE model handling - verify with Mixtral/DeepSeek V3 actual measurements
- Quantization overhead multipliers - test against GPTQ/AWQ/GGUF actual memory usage

**Domain knowledge validation (during implementation):**
- KV cache scaling for GQA/MQA - verify n_kv_heads values for popular models (Llama 3, Mistral, GPT-4)
- Framework overhead - measure PyTorch/CUDA baseline (500MB-1GB estimate needs confirmation)
- Activation memory formulas - validate MLP expansion factors (4x hidden_size assumption)
- Batch size scaling - test padding inefficiency with variable-length sequences

**Data curation (ongoing):**
- Model database needs 20-30 popular models with accurate parameter counts - requires manual curation
- GPU database needs 15-20 common GPUs with VRAM specs - straightforward from vendor specs
- Architecture parameters (n_kv_heads, intermediate_size, vocab_size) - extract from Hugging Face model cards

**Testing strategy:**
- Compare calculator results against vLLM memory profiling for common configs
- Test against HuggingFace accelerate memory estimates
- Validate quantized model estimates against actual GPTQ/AWQ/GGUF measurements
- Aim for <10% error on common configurations, <20% on edge cases

## Sources

### Primary (HIGH confidence)
- **apxml.com/tools/vram-calculator** (fetched 2026-02-09) - Comprehensive competitive analysis, feature landscape, industry expectations. Provided clear table stakes vs differentiators.
- **Training data on React/TypeScript/Web Workers patterns** (Jan 2025 cutoff) - Established architecture patterns, component composition, state management with Zustand.
- **raidy project** (provided as baseline) - Sister project architecture alignment for React, TypeScript, Vite, Tailwind, Biome patterns. Reduces cognitive load.

### Secondary (MEDIUM confidence)
- **Training data on LLM memory calculation patterns** (Jan 2025 cutoff) - VRAM formulas (KV cache, activations, optimizer states), quantization overhead, multi-GPU sharding.
- **EleutherAI Cookbook references** (fetched 2026-02-09) - Confirms ecosystem tools exist, validates domain patterns.
- **Training data on calculator application patterns** - Form-heavy UIs, numerical precision requirements, visualization best practices.

### Tertiary (LOW confidence - flagged for validation)
- **Vite 7.x, Tailwind CSS 4.x versions** - Assumed based on version progression, may not exist or be stable yet. MUST verify official documentation.
- **Fine-tuning memory formulas** - Based on training data, should validate against current HuggingFace docs, DeepSpeed papers.
- **Multi-GPU overhead constants** - Rough estimates (10-15% replication, 100-500MB buffers), need validation against Megatron-LM, production systems.
- **Quantization overhead multipliers** - Training data estimates (1.1-1.3x), should validate against actual GPTQ/AWQ/GGUF measurements.
- **MoE model specifics** - Mixtral/DeepSeek V3 parameter counts should be verified against official model cards.
- **Performance estimation approaches** - Tokens/sec formulas would require extensive benchmarking research (deferred to Phase 7).

### Missing (unable to access)
- **Context7 tool** - Could not verify current React 19, Zustand 5, Vite 7 API documentation
- **Hugging Face accelerate tool** - Could not access interface to compare features
- **vram.asmirnov.xyz** - Access blocked, could not analyze competitor
- **GPU_poor** - Access blocked, could not analyze competitor
- **Microsoft Learn** - Could not verify current best practices for 2026

**Recommended validation sources:**
- Official Vite documentation (https://vite.dev) - verify latest stable version
- Official Tailwind CSS documentation (https://tailwindcss.com) - verify v4 release status
- Official React documentation (https://react.dev) - verify React 19 stability
- Raidy project's actual package.json - confirm exact versions for alignment
- Zustand documentation (https://zustand.docs.pmnd.rs) - API patterns
- HuggingFace transformers memory anatomy - formula validation
- vLLM documentation - memory model validation
- DeepSpeed/Megatron-LM papers - multi-GPU overhead validation

---

*Research completed: 2026-02-09*
*Ready for roadmap: Yes*
