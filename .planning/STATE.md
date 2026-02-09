# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-09)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 2 - Inference Engine

## Current Position

Phase: 2 of 5 (Inference Engine)
Plan: 3 of 4
Status: In progress
Last activity: 2026-02-09 — Completed 02-02-PLAN.md (KV Cache & Inference Engine)

Progress: [███▓░░░░░░] 75% (Phase 2: 3/4 plans complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 3.6 min
- Total execution time: 0.42 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 (Foundation & Data) | 4/4 | 12 min | 3.0 min |
| 2 (Inference Engine) | 3/4 | 13 min | 4.3 min |

**Recent Trend:**
- Last 5 plans: 01-04 (5min), 02-01 (4min), 02-03 (4min), 02-02 (5min)
- Trend: Consistent 4-5 min range, stable for engine implementation plans

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

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
- Phase 4: Multi-GPU memory split naive division — must account for 10-20% replication and communication overhead

## Session Continuity

Last session: 2026-02-09 (plan execution)
Stopped at: Completed 02-02 (KV Cache & Inference Engine)
Resume file: .planning/phases/02-inference-engine/02-04-PLAN.md (Multi-GPU Distribution)

**Phase 2 In Progress:** Quantization engine (02-01), performance estimation (02-03), and KV cache/inference engine (02-02) complete. Ready for 02-04 (Multi-GPU Distribution) to finish phase. Note: Plans executed out of order (02-01, 02-03, 02-02, then 02-04) due to independent dependencies.
