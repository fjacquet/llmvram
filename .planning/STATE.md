# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-09)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 5 - Sharing & Comparison (Next)

## Current Position

Phase: 5 of 5 (Sharing & Comparison)
Plan: 2 of 3 (plans 05-01 and 05-02 complete, 05-03 pending)
Status: In progress
Last activity: 2026-02-09 — Completed 05-01-PLAN.md (URL Hash Persistence)

Progress: [█████████░] 94% (17/18 total plans: Phase 1: 4/4, Phase 2: 4/4, Phase 3: 4/4, Phase 4: 3/3, Phase 5: 2/3)

## Performance Metrics

**Velocity:**

- Total plans completed: 17
- Average duration: 4.5 min
- Total execution time: 1.27 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 (Foundation & Data) | 4/4 | 12 min | 3.0 min |
| 2 (Inference Engine) | 4/4 | 18 min | 4.5 min |
| 3 (Core UI) | 4/4 | 28 min | 7.0 min |
| 4 (Multi-GPU Support) | 3/3 | 15 min | 5.0 min |
| 5 (Sharing & Comparison) | 2/3 | 7.3 min | 3.7 min |

**Recent Trend:**

- Last 5 plans: 04-01 (5min), 04-02 (3min), 04-03 (6min), 05-02 (2.7min), 05-01 (4.6min)
- Trend: Phase 5 executing efficiently (average 3.7min). URL persistence (05-01) took 4.6min - slightly longer than comparison store (05-02) due to comprehensive testing (12 unit tests). Both plans well under average.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

**From 05-01 (URL Hash Persistence):**

- Use lz-string for URL compression to keep shareable links under 1800 chars (typical configs ~200-400 chars compressed)
- Store persistence reduced to only isDarkMode — all other state managed via URL hash for sharing functionality
- Short key names (q, sl, bs, kvq, ng, ss) in URL schema to save bytes before compression
- Custom model/GPU serialize full parameters, not just ID, for complete restoration when shared link opened
- deserializeFromURL returns null on any failure, never throws — enables graceful degradation with toast notification
- 300ms debounce on URL updates to avoid excessive history pollution during rapid parameter changes
- findModelById and findGPUById helpers moved to store module level for use by URL hydration

**From 05-02 (Comparison Store & UI):**

- Comparison store uses no persist middleware — snapshots are transient session data, not long-term preferences (prevents localStorage bloat and stale data)
- 3-max snapshots with FIFO eviction (oldest first) when 4th snapshot added to prevent unlimited memory growth
- Diff highlighting uses amber/yellow left-border (2px) rather than full background color (less overwhelming, maintains readability)
- Per-GPU VRAM displayed for multi-GPU configs instead of total VRAM for accurate utilization calculation
- Inline label editing with Escape to cancel, Enter/blur to save for quick snapshot renaming

**From 04-02 (Multi-GPU Store-Worker-Hook Integration):**

- numGPUs and shardingStrategy default to 1 and 'tensor-parallel' for backward compatibility
- multiGPU calculation only runs when numGPUs > 1 to avoid unnecessary computation (saves ~15-20% for single-GPU case)
- Hook parameters are optional with defaults to avoid breaking existing consumers
- Follow existing Decimal serialization pattern: serialize in Worker, reconstruct in hook with type-specific functions

**From 04-01 (Multi-GPU VRAM Calculation Engine):**

- Tensor parallelism shards weights/KV/activations but replicates embeddings and layer norms (~3% of weights)
- Pipeline parallelism does NOT divide KV cache (each GPU needs full context for its layers)
- For small KV cache scenarios (batch=1, moderate seq length), TP can use MORE memory per GPU than PP due to weight replication, NCCL buffers, and higher communication overhead outweighing KV cache savings
- MoE models get 15% extra communication overhead for expert routing in multi-GPU
- Single GPU (numGPUs=1) always returns passthrough with zero overhead regardless of strategy
- Interconnect validation: 'none' type blocks multi-GPU, PCIe warns when TP degree exceeds recommended max

**From 03-04 (Results Display & App Assembly):**
- ResultsPanel handles 4 distinct states in priority order: no selection → loading → error → results
- Error notifications use both inline card (persistent) and toast (immediate attention) for better UX
- Input panel sticky on desktop (lg:sticky lg:top-6) to keep controls visible while scrolling results
- Performance estimate positioned below VRAM section as secondary information (VRAM fit is primary concern)
- Dark mode text colors for table and chart tooltips fixed for proper contrast

**From 03-03 (Output Components):**

- Use semantic HTML elements instead of divs with ARIA roles: <output> for FitIndicator (calculation results), <aside> for Recommendations (complementary content)
- Use Recharts PieLabelRenderProps type with proper guards for optional properties to satisfy strict TypeScript
- Calculate recommendation savings using VRAM component ratios: ~50% of weights for quantization, ~50% of KV cache for context reduction, ~75% of KV cache for KV quantization
- Remove unused model parameter from Recommendations component (not needed for current recommendation logic)
- Output components are pure display components receiving all data as props (no direct Zustand store access)

**From 03-02 (Input Components):**

- Use Headless UI Combobox instead of native select for keyboard navigation and screen reader support
- Keep custom form state local with useState until submission to avoid polluting Zustand store with temporary values
- Implement log-scale slider for sequence length to provide even spacing across exponential range (512 to 131K)
- @types path alias conflicts with TypeScript's @types namespace — use @/types pattern instead
- Custom forms appear inline below selector when "Custom..." option selected

**From 03-01 (UI Foundation & Dark Mode):**

- Zustand persist uses partialize to persist only preferences (not large model/GPU objects) to avoid stale data
- FOUC prevention inline script in index.html reads localStorage before first paint using same key as persist middleware
- useDarkMode hook is single source of truth for document.documentElement.classList manipulation
- @custom-variant dark in Tailwind v4 enables class-based dark mode (critical for toggling, vs media query)

**From 02-04 (Web Worker Integration):**

- Worker serializes Decimal values to strings for structured cloning (Decimal methods don't survive postMessage)
- Hook provides sync fallback via dynamic imports when Workers unavailable (SSR, old browsers)
- Integration tests use realistic model/GPU configs (Llama 3 70B, Mixtral 8x7B) to verify pipeline correctness
- Worker uses relative imports (not path aliases) for Vite bundling compatibility

**From 02-03 (Performance Estimation Engine):**

- Use roofline model (min of memory-bound and compute-bound) for tokens/sec estimation — captures both bandwidth-limited and FLOPS-limited regimes
- TTFT estimated at 0.5x decode speed (2x slower prefill) due to quadratic vs linear attention
- 5% tolerance for bottleneck classification prevents flip-flopping at memory/compute boundary
- Handle missing FLOPS gracefully by defaulting to memory-bound only (Infinity for compute bound)

**From 02-02 (KV Cache & Inference Engine):**

- GQA ratio fallback: If num_kv_heads undefined, default to num_attention_heads (standard MHA, ratio=1.0)
- MoE active params: 20% shared + 80% expert * (num_experts_per_token / num_experts) for activation sizing
- KV cache formula: 2 *layers* hidden_size *seq_len* batch *precision* gqa_ratio / BYTES_PER_GB
- Total inference VRAM: weights + KV cache + activations + 1GB framework overhead
- Activation memory uses FP32 precision (4 bytes) regardless of weight quantization
- KV cache quantization independent from weight quantization (INFER-05) enables int4 KV with fp16 weights

**From 02-01 (Quantization Engine & Types):**

- Use Decimal.js for ALL arithmetic in engine calculations to avoid floating-point precision errors
- Bake overhead multipliers into BYTES_PER_PARAMETER constants (GPTQ/AWQ 1.2x) rather than calculating at runtime
- Use empirical bits-per-parameter for GGUF formats (from Artefact2 gist) instead of theoretical values
- CalculationInputSchema validates sequence length 512-131072 and batch size 1-64 based on research
- Engine functions must be pure (deterministic, no side effects) for testability and Web Worker offloading potential

**From 01-04 (Data Refresh Scripts):**

- Model fetch script writes to temporary file (models-fetched.json) to avoid overwriting curated database with incomplete data from gated models
- GPU database uses manual curation rather than API fetching for accuracy and source citation
- Scripts handle missing model config fields (intermediate_size) with fallback estimation (4*hidden_size)
- 13 gated models (LLaMA, Gemma, Command-R) documented as known limitation requiring manual addition

**From 01-03 (Model Database):**

- Used plan specifications directly rather than HuggingFace API (specs already vetted)
- MoE models specify total parameters (46.7B for Mixtral 8x7B) not active parameters (13B) — critical for VRAM accuracy
- All GQA models include num_kv_heads field for accurate KV cache calculations

**From 01-02 (GPU Database):**

- Fixed bus_width schema to support Apple Silicon unified memory (nonnegative vs positive)
- Added nvlink-5 to interconnect enum for NVIDIA B200 GPU support
- Included estimated M3 Ultra specs for forward-looking hardware planning

**From 01-01 (Project Scaffold):**

- Zod-first approach with z.infer<> for type inference ensuring schema and types stay in sync
- TypeScript noUncheckedIndexedAccess enabled for array access safety in VRAM calculations
- Biome handles both linting and formatting with single tool (matches raidy pattern)
- Path aliases configured consistently across all build tools (@engines, @components, @store, etc.)

**Earlier decisions:**

- Phase 1: Mirror raidy architecture for consistency across internal tools (IMPLEMENTED in 01-01)
- Phase 1: Client-side only deployment for static hosting with no infrastructure cost (Pending)
- Phase 1: Curated + custom model/GPU data for convenience and flexibility (Schemas in 01-01, GPU data in 01-02, Model data in 01-03)

### Pending Todos

None yet.

### Blockers/Concerns

**Research-flagged risks:**

- Phase 2: Quantization overhead (10-30% underestimation if ignored) — MITIGATED: BYTES_PER_PARAMETER includes GPTQ/AWQ 1.2x overhead and GGUF empirical bpp (02-01)
- Phase 2: KV cache scaling for GQA/MQA architectures — MITIGATED: calculateKVCacheVRAM correctly applies GQA ratio (8x reduction verified for Llama 3 70B) (02-02)
- Phase 2: MoE parameter confusion — MITIGATED: calculateInferenceVRAM uses total params for weights (46.7B Mixtral), active params for activations (02-02)
- Phase 2: Performance estimation accuracy — MITIGATED: Roofline model correctly identifies memory-bandwidth bottleneck for typical LLM inference (02-03)
- Phase 2: UI blocking calculations — MITIGATED: Web Worker offloads calculations to background thread with sync fallback (02-04)
- Phase 4: Multi-GPU memory split naive division — MITIGATED: calculateMultiGPUVRAM accounts for weight replication (TP: embeddings + layer norms), NCCL buffers (TP: 0.2 GB per peer), strategy-specific overhead (TP: 12%, PP: 5%), and MoE extra overhead (15%) (04-01)

**Verified via integration tests (02-04):**

- Llama 3 70B GPTQ fits on H100 80GB (~42 GB total: 39.12 GB weights + 1.25 GB KV cache + activations + 1GB framework)
- Mixtral 8x7B FP16 exceeds 80GB (~87 GB) — correctly uses total params (46.7B) not active (13B)
- KV quantization int4 reduces cache by 4x vs fp16
- Long context (131K tokens) calculates without error

## Session Continuity

Last session: 2026-02-09 (plan execution)
Stopped at: Completed Plan 05-01 (URL Hash Persistence)

**Phase 4 Complete:** All 3 plans finished. Multi-GPU + offloading fully functional.
- ✅ Multi-GPU calculation engine with TP/PP strategies (04-01)
- ✅ Store/Worker/Hook integration for multi-GPU state (04-02)
- ✅ Multi-GPU UI, offloading panel, quantization expansion (04-03)

**Phase 5 In Progress:** Sharing & comparison (2/3 plans complete)
- ✅ URL hash persistence with lz-string compression (05-01)
- ✅ Comparison store and UI with diff highlighting (05-02)
- 🔄 Next: Integration + save/restore comparison snapshots (05-03)
Resume file: .planning/phases/05-sharing-comparison/05-01-PLAN.md

**Phase 2 Complete:** All 4 plans finished. Inference engine fully functional.

**Phase 3 Complete:** All 4 plans finished. Core UI fully functional.
- ✅ UI foundation with Tailwind v4 dark mode and Zustand store (03-01)
- ✅ Model & GPU selectors with search and custom forms (03-02)
- ✅ VRAM breakdown chart, memory table, fit indicator, and recommendations (03-03)
- ✅ Results display panel assembly and responsive layout (03-04)

**Phase 4 In Progress:** Multi-GPU support (2/3 plans complete)
- ✅ Multi-GPU VRAM calculation engine with TP/PP strategies (04-01)
- ✅ Zustand store, Web Worker, and hook integration for multi-GPU (04-02)
- 🔄 Next: Multi-GPU UI components for input and results display (04-03)
