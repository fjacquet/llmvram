# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-09)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 1 - Foundation & Data

## Current Position

Phase: 1 of 5 (Foundation & Data)
Plan: 4 of 4
Status: Phase complete
Last activity: 2026-02-09 — Completed 01-04-PLAN.md (Data Refresh Scripts)

Progress: [██████████] 100% (Phase 1: 4/4 plans complete - PHASE COMPLETE)

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 3.0 min
- Total execution time: 0.20 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 (Foundation & Data) | 4/4 | 12 min | 3.0 min |

**Recent Trend:**
- Last 5 plans: 01-01 (2min), 01-02 (2min), 01-03 (3min), 01-04 (5min)
- Trend: Slight increase due to verification checkpoint in 01-04

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

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
- Phase 1: Quantization overhead (10-30% underestimation if ignored) — must apply 1.1-1.3x multipliers
- Phase 2: KV cache scaling for GQA/MQA architectures — MITIGATED: num_kv_heads now in model database (01-03)
- Phase 2: MoE parameter confusion — MITIGATED: Model database uses total params (46.7B for Mixtral 8x7B) per 01-03
- Phase 4: Multi-GPU memory split naive division — must account for 10-20% replication and communication overhead

## Session Continuity

Last session: 2026-02-09 (plan execution)
Stopped at: Completed 01-04 (Data Refresh Scripts) - PHASE 1 COMPLETE
Resume file: Ready for Phase 2 planning

**Phase 1 Complete:** All infrastructure, databases, and refresh scripts operational. Ready for Phase 2 (Inference Engine) development.
