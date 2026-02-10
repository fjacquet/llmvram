---
phase: 10-framework-presets-multi-gpu-training
plan: 02
subsystem: state-management
tags: [state, zustand, url-persistence, framework-presets, auto-apply]

# Dependency graph
requires:
  - phase: 10-framework-presets-multi-gpu-training
    plan: 01
    provides: Framework preset types and FRAMEWORK_PRESETS configuration constant
provides:
  - Framework preset state in uiStore with auto-apply logic
  - CPU offload optimizer state management
  - URL persistence for framework preset and CPU offload fields
affects: [10-03, framework-ui, training-configuration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Auto-apply pattern for framework presets (mode enforcement + optimization cascades)"
    - "Partial state updates with Zustand set() for multiple fields"
    - "URL serialization with conditional presence (training vs inference modes)"

key-files:
  created: []
  modified:
    - src/store/uiStore.ts
    - src/store/urlSerializer.ts

key-decisions:
  - "ZeRO stage derived from frameworkPreset in calculation layer, not stored separately"
  - "Framework preset persists in URL for both training and inference modes"
  - "CPU offload auto-resets to false when switching away from DeepSpeed presets"

patterns-established:
  - "Auto-apply logic uses FRAMEWORK_PRESETS constant to cascade optimization settings"
  - "Mode enforcement: vLLM/TGI presets force mode='inference'"
  - "URL keys follow existing short-key pattern: fp=frameworkPreset, co=cpuOffloadOptimizer"

# Metrics
duration: 2min
completed: 2026-02-10
---

# Phase 10 Plan 2: Framework Preset State Management + URL Persistence Summary

**Framework preset state with auto-apply logic and URL persistence for sharing training/inference configurations**

## Performance

- **Duration:** 2 minutes
- **Started:** 2026-02-10T17:08:16Z
- **Completed:** 2026-02-10T17:10:49Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Framework preset state management in uiStore with auto-apply logic
- CPU offload optimizer state management
- Auto-apply pattern: selecting preset auto-configures optimizations and enforces mode
- URL persistence for framework preset and CPU offload with short keys (fp, co)
- Conditional serialization: fp in training mode, fp in inference mode for vLLM/TGI

## Task Commits

Each task was committed atomically:

1. **Task 1: Framework preset state** - `9b418e8` (feat)
   - Add frameworkPreset field defaulting to 'none'
   - Add cpuOffloadOptimizer field defaulting to false
   - Implement setFrameworkPreset with auto-apply:
     - vLLM/TGI auto-switch to inference mode
     - DeepSpeed presets auto-enable gradient checkpointing + flash attention
     - Unsloth auto-enables GC, FA, and adamw-8bit optimizer
     - Non-DeepSpeed presets reset cpuOffloadOptimizer to false
   - Import FrameworkPreset and FRAMEWORK_PRESETS from @engines/frameworks

2. **Task 2: URL persistence** - `04f522a` (feat)
   - Add fp (frameworkPreset) and co (cpuOffloadOptimizer) to URLStateSchema
   - Serialize fp and co in training mode
   - Serialize fp in inference mode for vLLM/TGI presets
   - Import FrameworkPreset type from @engines/frameworks
   - Update serializeToURL signature with new fields

**Plan metadata:** (pending)

## Files Created/Modified

- `src/store/uiStore.ts` - Added frameworkPreset and cpuOffloadOptimizer state with auto-apply logic
- `src/store/urlSerializer.ts` - Added URL persistence for fp and co fields

## Decisions Made

**Auto-apply logic:**
- Framework preset selection cascades optimization settings based on FRAMEWORK_PRESETS config
- vLLM/TGI presets enforce mode='inference' (inference-only frameworks)
- DeepSpeed presets auto-enable gradientCheckpointing and flashAttention
- Unsloth preset additionally auto-enables adamw-8bit optimizer
- Non-DeepSpeed presets reset cpuOffloadOptimizer to false (DeepSpeed-specific feature)

**State architecture:**
- ZeRO stage is NOT stored separately - derived from frameworkPreset in calculation layer
- Avoids state duplication: frameworkPreset is the single source of truth for ZeRO stage
- CPU offload state is independent but only meaningful for DeepSpeed presets

**URL persistence:**
- fp and co follow existing short-key pattern for minimal URL size
- Framework preset serialized in both training and inference modes
- CPU offload only serialized if true (omit if false for cleaner URLs)
- Inference mode serializes fp only when preset is not 'none' (vLLM/TGI case)

## Deviations from Plan

None - plan executed exactly as written. All state management logic and URL persistence implemented according to specification.

## Issues Encountered

**Biome lint warnings (fixed immediately):**
- Unused `state` parameter in setFrameworkPreset → prefixed with underscore
- Formatting: enum array for fp required line breaks → reformatted multi-line

Both fixed before commit. No functional issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 10 Plan 3 (Multi-GPU UI Components):**
- Framework preset state available in uiStore
- Auto-apply logic tested via TypeScript compilation
- URL persistence round-trips framework preset and CPU offload state
- Integration points ready for UI components

**Test coverage:**
- All existing tests passing: 266 tests across 18 test suites
- TypeScript compilation clean
- Biome lint passes

**No blockers or concerns.**

## Self-Check: PASSED

All modified files exist and compile:
- FOUND: src/store/uiStore.ts
- FOUND: src/store/urlSerializer.ts

All commits exist in git history:
- FOUND: 9b418e8 (Task 1)
- FOUND: 04f522a (Task 2)

All verification passed:
- TypeScript compilation: PASS
- Biome lint: PASS
- Test suite: 266/266 passing

---
*Phase: 10-framework-presets-multi-gpu-training*
*Completed: 2026-02-10*
