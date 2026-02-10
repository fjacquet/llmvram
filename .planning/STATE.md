# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 6 - Fine-Tuning Calculation Engines

## Current Position

Phase: 6 of 10 (Fine-Tuning Calculation Engines)
Plan: 1 of 3 complete
Status: In progress
Last activity: 2026-02-10 — Completed 06-01-PLAN.md (Training Foundation)

Progress: [█████████░] 90% (19/21 plans complete: v1.0 18/18, v1.1 1/3)

## Performance Metrics

**Velocity (v1.0):**
- Total plans completed: 18
- Average duration: ~45 min
- Total execution time: ~13.5 hours

**By Phase (v1.0):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Foundation & Data | 4 | ~3h | ~45min |
| 2. Inference Engine | 4 | ~3h | ~45min |
| 3. Core UI | 4 | ~3h | ~45min |
| 4. Multi-GPU Support | 3 | ~2.25h | ~45min |
| 5. Sharing & Comparison | 3 | ~2.25h | ~45min |

**Velocity (v1.1 in progress):**
- Total plans completed: 1
- Average duration: 2 min
- Total execution time: 2 min

**By Phase (v1.1):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 6. Fine-Tuning Engines | 1/3 | 2min | 2min |

**Recent Trend:**
- v1.1 Phase 6: Fast start (type/schema foundation: 2min)
- Next plans will be calculation engines (likely longer)

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

**From v1.0 (architectural foundation):**

- Mirror raidy architecture — consistency across internal tools, proven patterns
- Client-side only — static deployment, no infrastructure cost
- Curated + custom model/GPU data — convenience + flexibility
- Decimal.js for all arithmetic — avoid floating-point precision errors
- Web Workers for calculations — prevent UI blocking with sync fallback
- Zod schemas as type source of truth — single source for validation + TypeScript types
- Total MoE params for VRAM — all expert weights must fit in memory
- URL hash (not localStorage) for config — shareable links with lz-string compression

**For v1.1 (from research and execution):**

- Separate training from inference calculation paths — avoid critical pitfall (30-60% underestimation)
- Optimizer states always FP32 — maintain numerical stability even in mixed precision training (06-01)
- LoRA optimizer states apply to adapters only — not full model params
- DeepSpeed ZeRO stage memory: 2x/4x/8x savings — not simple divide-by-N
- Training framework overhead is 1.5GB (higher than inference 1.0GB) due to autograd + optimizer buffers (06-01)
- Standard transformer has 7 targetable modules per layer for LoRA (4 attention + 3 MLP) (06-01)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-10
Stopped at: Completed 06-01-PLAN.md (Training Foundation: types, constants, schemas)
Resume file: .planning/phases/06-fine-tuning-calculation-engines/06-01-SUMMARY.md
Next action: Execute 06-02-PLAN.md (Full Fine-Tuning Engine) or create it if not yet planned
