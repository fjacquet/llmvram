# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-09)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 3 - Core UI (In Progress)

## Current Position

Phase: 3 of 5 (Core UI)
Plan: 1 of 4
Status: In progress
Last activity: 2026-02-09 — Completed 03-01-PLAN.md (UI Foundation & Dark Mode)

Progress: [████▓░░░░░] 11% (9/15 total plans: Phase 1: 4/4, Phase 2: 4/4, Phase 3: 1/4)

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 3.67 min
- Total execution time: 0.55 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 (Foundation & Data) | 4/4 | 12 min | 3.0 min |
| 2 (Inference Engine) | 4/4 | 18 min | 4.5 min |
| 3 (Core UI) | 1/4 | 3 min | 3.0 min |

**Recent Trend:**
- Last 5 plans: 02-03 (4min), 02-02 (5min), 02-04 (5min), 03-01 (3min)
- Trend: Phase 3 started with faster 3-minute plan, back to Phase 1 speed

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

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
- KV cache formula: 2 * layers * hidden_size * seq_len * batch * precision * gqa_ratio / BYTES_PER_GB
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
- Phase 4: Multi-GPU memory split naive division — must account for 10-20% replication and communication overhead

**Verified via integration tests (02-04):**
- Llama 3 70B GPTQ fits on H100 80GB (~42 GB total: 39.12 GB weights + 1.25 GB KV cache + activations + 1GB framework)
- Mixtral 8x7B FP16 exceeds 80GB (~87 GB) — correctly uses total params (46.7B) not active (13B)
- KV quantization int4 reduces cache by 4x vs fp16
- Long context (131K tokens) calculates without error

## Session Continuity

Last session: 2026-02-09 (plan execution)
Stopped at: Completed 03-01 (UI Foundation & Dark Mode)
Resume file: .planning/phases/03-core-ui/03-02-PLAN.md (next plan)

**Phase 2 Complete:** All 4 plans finished. Inference engine fully functional.

**Phase 3 Started (1/4 complete):**
- ✅ UI foundation with Tailwind v4 dark mode and Zustand store (03-01)
- ⏳ Model & GPU selectors (03-02)
- ⏳ Calculation parameter inputs (03-03)
- ⏳ Results display with breakdown visualization (03-04)

**Ready for Plan 03-02:** Zustand store has setSelectedModel/setSelectedGPU actions. @headlessui/react installed for Combobox components.
