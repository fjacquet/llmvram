# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Phase 8 - Memory Optimization Features

## Current Position

Phase: 8 of 10 (Memory Optimization Features)
Plan: 1 of TBD complete
Status: In progress
Last activity: 2026-02-10 — Completed 08-01-PLAN.md (optimization calculation engine)

Progress: [██████████] 100% (24/24 plans complete: v1.0 18/18, v1.1 6/6)

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
- Total plans completed: 6
- Average duration: ~3.6 min
- Total execution time: ~21.8 min

**By Phase (v1.1):**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 6. Fine-Tuning Engines | 3/3 | ~11.5min | ~3.8min |
| 7. Training State & Basic UI | 2/2 | ~6min | ~3min |
| 8. Memory Optimization Features | 1/TBD | ~6.3min | ~6.3min |

**Recent Trend:**
- Phase 8 Plan 1 complete: 6.3min execution (TDD optimization calculation engine)
- Phase 7 complete: 2 plans, ~6min total, zero deviations
- Phase 6 complete: 3 plans, ~11.5min total, rapid TDD execution
- All v1.1 plans completed with clean test coverage
- Zero deviations in Phase 7-8 Plans (all plans executed exactly as written)

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
- Training activations fundamentally different from inference — O(N^2) attention matrices vs O(N) KV cache (06-02)
- Mixed precision (FP16/BF16) requires FP32 master weights; pure FP32 training does not (06-02)
- MoE models scale activations by active parameter ratio (20% shared + 80% expert * active_ratio) (06-02)
- LoRA adapter params = 2 * rank * hiddenSize * targetModuleCount * layers (A + B matrices) (06-03)
- LoRA optimizer states/gradients apply ONLY to adapter parameters, not frozen base (PITFALLS.md #3) (06-03)
- QLoRA uses fixed three-precision architecture: NF4 base (0.5 bytes), FP16 adapters (2 bytes), FP32 optimizer (8 bytes) (06-03)
- 7B model QLoRA fits in 6-12GB range (research validation: ~10 GB total) (06-03)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-10
Stopped at: Phase 8 Plan 1 complete (optimization calculation engine)
Resume file: .planning/phases/08-memory-optimization-features/08-01-SUMMARY.md
Next action: Continue Phase 8 with remaining plans (optimization UI components and state management)
