# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.
**Current focus:** Milestone v1.1 Fine-Tuning Estimation

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-10 — Milestone v1.1 started

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

**From v1.0 (carried forward):**

- Zod-first approach with z.infer<> for type inference ensuring schema and types stay in sync
- Decimal.js for ALL arithmetic in engine calculations to avoid floating-point precision errors
- Engines are pure functions — no React/DOM dependencies, designed for testability and Web Worker offloading
- Web Worker offloads calculations to background thread with sync fallback
- Zustand persist uses partialize to persist only preferences (not large model/GPU objects)
- MoE models use total parameters for VRAM, active parameters for compute
- URL hash sharing with lz-string compression for shareable configurations

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-10 (milestone initialization)
Status: DEFINING REQUIREMENTS
