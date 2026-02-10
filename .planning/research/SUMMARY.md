# Project Research Summary

**Project:** LLM VRAM Calculator - Fine-Tuning Milestone
**Domain:** Browser-based VRAM estimation tool for LLM fine-tuning workloads
**Researched:** 2026-02-10
**Confidence:** HIGH

## Executive Summary

Fine-tuning VRAM estimation extends the existing inference calculator with training workload support. Research confirms that **no new libraries are required** — the existing stack (React 19, Decimal.js, Zod, Recharts, Zustand) handles all requirements. Fine-tuning features are pure calculation logic (formulas for optimizer states, gradients, activations), UI extensions (input panels, memory breakdowns), and static configuration data (framework presets). All formulas are well-documented in authoritative sources (HuggingFace, DeepSpeed, QLoRA papers).

The recommended approach follows established patterns: pure calculation engines (no React dependencies), Zod-first type system, Zustand state management, and component-based UI. Critical differentiators are framework presets (DeepSpeed ZeRO, Unsloth, vLLM/TGI) and LoRA/QLoRA support, enabling fine-tuning estimation on consumer hardware. The architecture cleanly separates training from inference logic to avoid the most critical pitfall: reusing inference KV cache calculations for training (causes 30-60% underestimation).

Key risks center on formula accuracy: optimizer states must account for FP32 precision (8 bytes/param for AdamW) even in mixed precision training, LoRA optimizer states apply only to adapters (~1% of params, not full model), and gradient accumulation reduces activation memory but not gradient or optimizer state memory. Mitigation: validate against real training runs with <15% error target, implement 16 critical pitfall checks, and cross-validate against other calculators.

## Key Findings

### Recommended Stack

**No new libraries needed.** The existing validated stack is sufficient for all fine-tuning features. Fine-tuning requires three types of extensions: (1) calculation engines with standard formulas, (2) UI components reusing existing primitives, and (3) static JSON configuration data. Research confirms no NPM libraries exist for fine-tuning VRAM estimation — implementations use hand-rolled formulas from papers and documentation.

**Core technologies (all existing):**
- **Decimal.js** — precision arithmetic for all training memory calculations (weights, gradients, optimizer states, activations)
- **Zod** — validate training configurations (optimizer type, LoRA rank, batch size, gradient accumulation)
- **React 19 + Headless UI** — training mode selectors, LoRA config panels, optimizer dropdowns
- **Recharts** — extend memory breakdown chart with training components (optimizer states, gradients, activations expanded)
- **Zustand** — add training state slice (mode, optimizer config, LoRA config, framework preset)

**Key integration points:**
- Extend `utils/schemas.ts` with `FineTuningConfigSchema`, `OptimizerConfigSchema`, `LoRAConfigSchema`
- Create new engines: `finetuning.ts`, `lora.ts`, `optimizer.ts`, `gradient-accumulation.ts`, `framework-presets.ts`
- Reuse existing: `quantization.ts` (for QLoRA base), model database, GPU database, URL serialization

### Expected Features

**Must have (table stakes):**
- Training mode toggle (Inference / Full FT / LoRA / QLoRA) — users cannot estimate without specifying method
- Optimizer selection (AdamW, SGD, 8-bit Adam) — 4x memory difference between optimizers
- Full fine-tuning calculation — model weights + gradients + optimizer states (8B/param) + activations
- LoRA calculation — frozen base + adapters (~1% params) + optimizer/gradients (adapters only)
- QLoRA calculation — 4-bit base + FP16 adapters + paged optimizer
- Training memory breakdown visualization — extend donut chart with optimizer states, gradients, activations
- Gradient accumulation input + effective batch size display — critical for memory optimization
- Mixed precision toggle (FP16/BF16) — reduces activations/gradients to 2 bytes

**Should have (competitive differentiators):**
- Framework presets (DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI) — one-click accurate configs
- Gradient checkpointing toggle — 50-80% activation memory reduction (with 25-40% slowdown warning)
- Flash Attention toggle — 50-80% KV cache reduction during training
- LoRA rank/alpha inputs — customize adapter size (rank 8-256, alpha typically 2x rank)
- Multi-GPU training estimation — DeepSpeed ZeRO stage impact (2x/4x/8x memory savings)

**Defer (v2+):**
- CPU offloading for training (ZeRO-Offload complexity)
- Training cost estimation (provider-specific, high variance)
- FSDP/Megatron strategies (DeepSpeed only for v1.1)
- Evaluation during training memory (separate concern)
- Training speed estimation (requires benchmarking data)

### Architecture Approach

The fine-tuning milestone extends the existing inference-focused architecture with parallel engines and conditional UI components. Architecture follows established patterns: pure calculation engines (Decimal.js), Zod schemas for type safety, Zustand for state, and React components split by concern. New features integrate cleanly by adding parallel engines, extending the store, and creating new input/output components that reuse existing patterns.

**Major components:**
1. **Calculation Engines (Pure Functions)** — `finetuning.ts` (full/LoRA/QLoRA VRAM), `lora.ts` (adapter parameter calculation), `optimizer.ts` (state memory by type), `gradient-accumulation.ts` (effective batch size), `framework-presets.ts` (optimization profiles)
2. **Store Extensions** — Add training slice to Zustand store (mode, optimizer config, LoRA config, framework preset, gradient accumulation steps, gradient checkpointing flag)
3. **Input Components** — `TrainingModeSelector` (radio group), `LoRAConfigPanel` (rank/alpha/targets), `OptimizerSelector` (dropdown), `GradientAccumulationInput` (steps + effective batch display), `FrameworkPresetSelector` (preset cards)
4. **Output Components** — Extend `VRAMBreakdownChart` with training slices, extend `MemoryBreakdownTable` with training rows, add `FrameworkOptimizationBadge` (show active optimizations)
5. **Static Data** — `framework-presets.json` (DeepSpeed ZeRO, Unsloth, vLLM/TGI configs), `optimizers.json` (bytes/param, stability notes)

**Data flow:** User input (training mode, optimizer, LoRA config) → Zustand store → `calculateFineTuningVRAM()` engine → `FineTuningVRAMBreakdown` (Decimal.js) → Components render (chart, table, badges)

**Key pattern:** Training calculations are completely separate from inference (no KV cache reuse), avoiding the most critical pitfall. Conditional rendering shows training-specific UI only when training mode is active.

### Critical Pitfalls

1. **Reusing inference KV cache formula for training** — Training processes batches in parallel with full attention matrices, not sequential generation. KV cache behavior is fundamentally different. Without Flash Attention, training needs `2 * batch * n_heads * seq_len² * 2 bytes` for attention matrices (not the inference KV cache). Prevention: Separate training calculation path, add Flash Attention toggle for training.

2. **Optimizer state precision assumption** — AdamW maintains states in FP32 (4 bytes) for numerical stability, even when training in BF16/FP16. Total: 8 bytes per trainable parameter (4B momentum + 4B variance). Prevention: Fixed formula `optimizer_states = trainable_params * 8 bytes`, independent of training precision.

3. **LoRA adapter-only optimizer states ignored** — Only LoRA adapter parameters (~0.2-2% of model) are trainable. Frozen base model weights have no optimizer states. Prevention: `optimizer_memory = adapter_params * 8 bytes` (NOT `base_model_params * 8 bytes`). For Llama 7B rank 16: 16.8M adapter params, not 7B total params.

4. **Gradient accumulation peak memory misconception** — Gradient accumulation reduces per-step batch size (activation memory), but does NOT reduce peak gradient or optimizer state memory. Prevention: `peak_memory = weights + gradients + optimizer_states + (micro_batch_activations)` (NOT effective batch activations). Savings: ~10-15%, not 87% with 8x accumulation.

5. **DeepSpeed ZeRO stage memory profile confusion** — Memory savings are 2x/4x/8-10x respectively (Stage 1/2/3), NOT simple divide-by-N. Stage 1: replicate weights+gradients, partition optimizer (2x savings). Stage 2: replicate weights, partition gradients+optimizer (4x savings). Stage 3: partition everything (8-10x savings, 15-30% throughput loss). Prevention: Use 2x/4x/8x multipliers, not `total_memory / num_gpus`.

## Implications for Roadmap

Based on research, suggested phase structure prioritizes accurate calculation formulas first (avoiding critical pitfalls), then state management and basic UI, then advanced features and framework presets, and finally validation.

### Phase 1: Fine-Tuning Calculation Engine (Core)
**Rationale:** Pure functions with no UI dependencies enable independent testing and formula validation before UI complexity. Addresses critical pitfalls #1, #2, #3 (KV cache reuse, optimizer precision, LoRA adapter-only states).

**Delivers:**
- `engines/finetuning.ts` — `calculateFineTuningVRAM()` for full/LoRA/QLoRA modes
- `engines/lora.ts` — `calculateLoRAParams()` for adapter parameter calculation
- `engines/optimizer.ts` — `calculateOptimizerMemory()` for AdamW/SGD/8-bit variants
- Comprehensive tests validating formulas against known values (Llama 7B LoRA r=16 ≈ 16GB)

**Addresses:** Table stakes features (full FT, LoRA, QLoRA calculations), critical formulas must be accurate to avoid OOM failures.

**Avoids:** Pitfall #1 (separate training from inference), Pitfall #2 (FP32 optimizer states), Pitfall #3 (adapter-only optimizer memory).

**Research flag:** Standard formulas — skip research-phase, implementation ready.

---

### Phase 2: Training State & UI Foundation
**Rationale:** Extend store and add basic UI before advanced features. Training mode selector enables conditional rendering for all subsequent phases. Framework presets data prepared early for preset selector in Phase 4.

**Delivers:**
- Extended Zod schemas (`FineTuningConfigSchema`, `OptimizerConfigSchema`, `LoRAConfigSchema`)
- Zustand store training slice (mode, optimizer, LoRA config, framework preset, gradient accumulation, checkpointing)
- `TrainingModeSelector` component (Inference / Full FT / LoRA / QLoRA radio group)
- `data/framework-presets.json` (DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI configurations)
- `data/optimizers.json` (bytes/param, stability notes)

**Uses:** Zod (schemas), Zustand (store), React (components), Headless UI (radio group).

**Implements:** State management foundation for training features.

**Research flag:** Standard patterns — skip research-phase, follows existing store structure.

---

### Phase 3: Advanced Training Inputs
**Rationale:** After core calculations work (Phase 1) and state management exists (Phase 2), add advanced input components. LoRA config panel and gradient accumulation calculator provide fine-grained control.

**Delivers:**
- `LoRAConfigPanel` component (rank, alpha, target modules, presets)
- `OptimizerSelector` component (dropdown with memory impact shown)
- `GradientAccumulationInput` component (steps input + effective batch size display)
- Mixed precision toggle (FP16/BF16/FP32 selector)
- Gradient checkpointing toggle (with memory reduction % and slowdown warning)

**Addresses:** Should-have features (LoRA customization, gradient accumulation, checkpointing).

**Avoids:** Pitfall #4 (gradient accumulation only reduces activations), Pitfall #7 (checkpointing shows compute cost).

**Research flag:** Standard patterns — skip research-phase, UI components reuse existing primitives.

---

### Phase 4: Memory Visualization & Output
**Rationale:** After inputs exist (Phase 3), extend output components to display training memory breakdown. Memory visualization helps users understand where VRAM goes (optimizer states, gradients, activations).

**Delivers:**
- Extended `VRAMBreakdownChart` with training slices (optimizer states, gradients, activations, LoRA adapters)
- Extended `MemoryBreakdownTable` with training rows (trainable params, frozen params, optimizer type)
- `FrameworkOptimizationBadge` component (show active optimizations from preset)
- Training feasibility indicator ("Fits in VRAM" / "Tight fit" / "Insufficient VRAM")

**Addresses:** Table stakes (training memory breakdown visualization).

**Implements:** Training-specific output components extending existing chart/table patterns.

**Research flag:** Standard patterns — skip research-phase, Recharts handles stacked bar charts natively.

---

### Phase 5: Framework Presets & Multi-GPU
**Rationale:** After single-GPU training works (Phases 1-4), add framework-specific optimizations and multi-GPU support. DeepSpeed ZeRO requires careful stage modeling (2x/4x/8x, not divide-by-N).

**Delivers:**
- `engines/framework-presets.ts` — `applyFrameworkOptimizations()` for vLLM/TGI/Unsloth/DeepSpeed
- `FrameworkPresetSelector` component (preset cards with optimization badges)
- `GradientAccumulationCalculator` — `recommendGradientAccumulation()` for memory-constrained scenarios
- Multi-GPU training estimation with DeepSpeed ZeRO stage impact
- Per-GPU memory display for distributed training

**Addresses:** Differentiator features (framework presets, multi-GPU).

**Avoids:** Pitfall #6 (ZeRO stage 2x/4x/8x savings, not linear), Pitfall #10 (framework-specific overhead differences).

**Research flag:** Needs validation — DeepSpeed ZeRO formulas inferred from documentation, test against real multi-GPU runs.

---

### Phase 6: Validation & Polish
**Rationale:** Final phase validates all formulas against ground truth, cross-checks other calculators, and tests edge cases. Target <15% error vs real training runs.

**Delivers:**
- Ground truth benchmarks (Llama 7B LoRA, Llama 7B QLoRA, Llama 70B ZeRO-3)
- Cross-validation against Modal calculator, HuggingFace Accelerate estimator, DeepSpeed config generator
- Edge case testing (MoE models, GQA models, long sequences, large ranks)
- Documentation and user guidance (when to use LoRA vs QLoRA, gradient accumulation recommendations)
- "Report actual vs estimated" feedback mechanism

**Addresses:** Pitfall #14 (calculator cross-validation), all formula accuracy concerns.

**Research flag:** Validation phase — measure error rates, update formulas if >15% deviation.

---

### Phase Ordering Rationale

- **Phase 1 first:** Pure calculation engines with no UI dependencies enable independent testing and formula validation. Critical pitfalls (#1, #2, #3) must be addressed in formulas before UI complexity.
- **Phase 2 before 3:** State management and training mode selector must exist before advanced input components can conditionally render.
- **Phase 4 after 3:** Memory visualization requires training inputs to exist (can't show breakdown without calculation results).
- **Phase 5 after 4:** Framework presets and multi-GPU support build on single-GPU foundation. DeepSpeed ZeRO complexity requires core training to work first.
- **Phase 6 final:** Validation phase catches formula errors across all features before public release.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 5 (Multi-GPU):** DeepSpeed ZeRO-3 memory partitioning formulas inferred from documentation, not explicitly stated. Needs validation against real multi-GPU training runs. Communication overhead percentages vary by interconnect (10-25% range in sources).

Phases with standard patterns (skip research-phase):
- **Phase 1 (Engines):** Formulas well-documented in HuggingFace, DeepSpeed, QLoRA papers. No ambiguity.
- **Phase 2 (State/UI):** Follows existing Zustand store pattern, React component structure.
- **Phase 3 (Inputs):** Reuses Headless UI primitives (radio groups, dropdowns, toggles).
- **Phase 4 (Outputs):** Extends existing Recharts visualizations, standard table extension.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| **Stack** | HIGH | Existing stack verified sufficient. No new libraries needed (Decimal.js handles all calculations, Recharts handles stacked bar charts, Zod handles validation). |
| **Features** | HIGH | Table stakes and differentiators identified from authoritative sources (HuggingFace docs, DeepSpeed docs, QLoRA paper). MVP scope clear (12-15 days). |
| **Architecture** | HIGH | Follows established patterns (pure engines, Zod schemas, Zustand state, component-based UI). Integration points well-defined. Build order logical (engines → store → inputs → outputs). |
| **Pitfalls** | HIGH | 16 pitfalls identified from current 2026 sources. Critical pitfalls (#1-7) have clear prevention strategies. Phase-specific warnings mapped. Validation strategy defined (<15% error target). |

**Overall confidence:** HIGH

Research validates that fine-tuning estimation is well-understood with documented formulas and established patterns. No new libraries required reduces risk. Critical pitfalls identified early with clear prevention strategies.

### Gaps to Address

Areas where research was inconclusive or needs validation during implementation:

- **DeepSpeed ZeRO-3 exact memory partitioning:** Formulas inferred from documentation, not explicitly stated. Sources agree on 8-10x memory savings but don't provide exact per-component breakdown. Mitigation: Test against real ZeRO-3 training runs in Phase 5, compare to DeepSpeed config generator output.

- **Unsloth-specific optimizations:** Claims 70% memory reduction vs standard LoRA, but optimizations are proprietary. Sources validate claims against user reports, not source code. Mitigation: Use conservative 60% reduction estimate, add disclaimer "based on Unsloth benchmarks."

- **Training throughput estimation:** Deferred to v2+. Requires benchmarking data not available from web search. High variance across frameworks (Unsloth 2x faster, DeepSpeed ZeRO-3 20-40% slower). Mitigation: Focus v1.1 on memory estimation only, add throughput in future milestone.

- **Activation memory per-layer variation:** Simplified formula uses `10-12x hidden_size * batch * seq_len` per transformer block. Reality: MLP layers expand to intermediate_size (4x hidden), attention has seq_len² complexity without Flash Attention. Mitigation: Use conservative 12x multiplier, add Flash Attention toggle to address seq_len² issue.

- **CPU offloading performance penalties:** Sources range from "10-30x slower" to "15-30% slower" depending on PCIe bandwidth and offload strategy. Mitigation: Defer CPU offload to v1.2, focus v1.1 on GPU-only training.

## Sources

### Primary (HIGH confidence)

**Technology Stack:**
- [Decimal.js vs BigNumber.js](https://medium.com/@josephgathumbi/decimal-js-vs-c1471b362181) — validated Decimal.js sufficiency
- [npm-compare: JavaScript Arbitrary-Precision Libraries](https://npm-compare.com/big.js,bignumber.js,decimal.js,decimal.js-light) — Decimal.js most feature-complete
- [Top React Chart Libraries 2026](https://aglowiditsolutions.com/blog/react-chart-libraries/) — Recharts still recommended

**Fine-Tuning Formulas:**
- [Modal: How much VRAM do I need for LLM model fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning) — formula verification for full fine-tuning with AdamW
- [HuggingFace Model Memory Anatomy](https://huggingface.co/docs/transformers/main/en/model_memory_anatomy) — official memory breakdown
- [Memory Requirements (HBM, GPU RAM)](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/memory-requirements-hbm-gpu-ram) — optimizer states precision

**LoRA & QLoRA:**
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — original LoRA architecture
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — QLoRA 4-bit NF4 quantization
- [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — official QLoRA integration
- [LoRA fine-tuning Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — rank/alpha recommendations

**DeepSpeed ZeRO:**
- [Zero Redundancy Optimizer - DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) — official ZeRO stage explanations
- [Memory Requirements — DeepSpeed](https://deepspeed.readthedocs.io/en/latest/memory.html) — per-stage memory formulas
- [ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/) — CPU offloading mechanics

**Framework Documentation:**
- [Unsloth GitHub](https://github.com/unslothai/unsloth) — 2x faster, 70% less VRAM claims
- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180) — inference-only optimizations
- [HuggingFace TGI Documentation](https://huggingface.co/docs/text-generation-inference/en/index) — TGI optimization techniques

### Secondary (MEDIUM confidence)

**Gradient Accumulation & Checkpointing:**
- [Gradient Accumulation: Increase Batch Size Without Explicitly Increasing Batch Size](https://blog.dailydoseofds.com/p/gradient-accumulation-increase-batch) — activation memory only
- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/) — selective checkpointing
- [Small Batch Size Training for Language Models (2025)](https://arxiv.org/abs/2507.07101) — gradient accumulation research

**Flash Attention:**
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — 10-20x memory savings at 2K-4K seq lengths
- [Out of the box acceleration and memory savings](https://pytorch.org/blog/out-of-the-box-acceleration/) — PyTorch integration

**Mixed Precision:**
- [Mixed Precision Training in LLMs: FP16, BF16, FP8, and Beyond](https://medium.com/@dpratishraj7991/mixed-precision-training-in-llms-fp16-bf16-fp8-and-beyond-b4af13ca846f) — master weights in FP32
- [How can using FP16, BF16, or FP8 mixed precision speed up my model training?](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training) — 20-30% savings, not 50%

### Tertiary (LOW confidence, needs validation)

**Training Throughput:**
- Sources vary widely on speedup/slowdown percentages (Unsloth 2x, DeepSpeed ZeRO-3 20-40% slower, gradient checkpointing 25-40% slower). Defer to v2+ when benchmarking data available.

**CPU Offloading Performance:**
- Claims range from "10-30x slower" to "15-30% slower" depending on PCIe bandwidth. Needs hardware-specific validation.

**Unsloth Proprietary Optimizations:**
- 70% memory reduction claim validated against user reports, not source code. Use conservative estimate.

---
*Research completed: 2026-02-10*
*Ready for roadmap: yes*
