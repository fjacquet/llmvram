# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-09)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 1 - Foundation & Data

## Current Position

Phase: 1 of 5 (Foundation & Data)
Plan: 0 of TBD
Status: Ready to plan
Last activity: 2026-02-09 — Roadmap created with 5 phases covering 36 v1 requirements

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: N/A
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: None yet
- Trend: N/A

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Phase 1: Mirror raidy architecture for consistency across internal tools (Pending implementation)
- Phase 1: Client-side only deployment for static hosting with no infrastructure cost (Pending implementation)
- Phase 1: Curated + custom model/GPU data for convenience and flexibility (Pending implementation)

### Pending Todos

None yet.

### Blockers/Concerns

**Research-flagged risks:**
- Phase 1: Quantization overhead (10-30% underestimation if ignored) — must apply 1.1-1.3x multipliers
- Phase 2: KV cache scaling for GQA/MQA architectures — must factor n_kv_heads/n_heads ratio to avoid 2-8x overestimation
- Phase 2: MoE parameter confusion — Mixtral loads ALL 46.7B params in VRAM, not just 13B active
- Phase 4: Multi-GPU memory split naive division — must account for 10-20% replication and communication overhead

## Session Continuity

Last session: 2026-02-09 (roadmap creation)
Stopped at: Roadmap written, ready for Phase 1 planning
Resume file: None
