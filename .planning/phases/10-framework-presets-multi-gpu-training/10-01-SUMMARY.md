---
phase: 10-framework-presets-multi-gpu-training
plan: 01
subsystem: calculation-engines
tags: [deepspeed, zero, multi-gpu, training, framework-presets, cpu-offload]

# Dependency graph
requires:
  - phase: 06-fine-tuning-calculation-engines
    provides: Training VRAM breakdown types and calculation functions
  - phase: 08-memory-optimization-features
    provides: Training optimization patterns and memory reduction techniques
provides:
  - Framework preset types and FRAMEWORK_PRESETS configuration constant
  - DeepSpeed ZeRO calculation engine with stage-specific partitioning logic
  - CPU offload memory calculation for optimizer states and parameters
  - ZeROResult type for per-GPU memory breakdown
affects: [10-02, 10-03, multi-gpu-ui, training-presets]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ZeRO stage partitioning (NOT simple divide-by-N)"
    - "CPU offload calculation for optimizer states and parameters"
    - "Framework preset configurations with mode and auto-optimizations"

key-files:
  created:
    - src/engines/frameworks.ts
    - src/engines/deepspeed.ts
    - src/engines/deepspeed.test.ts
  modified:
    - src/engines/types.ts
    - src/engines/index.ts

key-decisions:
  - "ZeRO-3 framework overhead increased by 15% for parameter gather/scatter"
  - "Single GPU passthrough returns original breakdown unchanged (no partitioning overhead)"
  - "CPU offload operates on single GPU breakdown (apply after ZeRO partitioning)"

patterns-established:
  - "ZeRO stages partition different components: ZeRO-1 (optimizer only), ZeRO-2 (optimizer+gradients), ZeRO-3 (all training state)"
  - "Framework presets map to mode (training/inference) and auto-optimizations (gradient checkpointing, flash attention, optimizer)"
  - "Reduction factor calculated as singleGPU.total / perGPU.total"

# Metrics
duration: 5min
completed: 2026-02-10
---

# Phase 10 Plan 1: Framework Preset Types + DeepSpeed ZeRO Engine Summary

**DeepSpeed ZeRO-1/2/3 calculation engine with stage-specific partitioning (2x/4x/8x reduction) and framework preset configurations for 7 training/inference frameworks**

## Performance

- **Duration:** 5 minutes
- **Started:** 2026-02-10T16:56:23Z
- **Completed:** 2026-02-10T17:02:08Z
- **Tasks:** 1 (TDD with 3 commits)
- **Files modified:** 5

## Accomplishments

- Framework preset types with FRAMEWORK_PRESETS constant mapping 7 presets (none, deepspeed-zero1/2/3, unsloth, vllm, tgi)
- DeepSpeed ZeRO calculation engine implementing stage-specific partitioning logic (NOT simple divide-by-N)
- CPU offload memory calculation for moving optimizer states and/or parameters to CPU RAM
- Comprehensive test coverage with 13 tests validating ZeRO-1/2/3 partitioning and CPU offload

## Task Commits

Each TDD phase was committed atomically:

1. **RED: Failing tests** - `3c6b911` (test)
   - Framework preset types and FRAMEWORK_PRESETS constant
   - ZeROResult interface for per-GPU memory breakdown
   - Comprehensive test suite for ZeRO-1/2/3 partitioning logic
   - CPU offload memory calculation tests

2. **GREEN: Implementation** - `4d32106` (feat)
   - calculateZeROMemoryPerGPU: partition training state per ZeRO stage
   - calculateCPUOffloadMemory: split memory between GPU and CPU
   - ZeRO-1: optimizer states + activations partitioned, rest replicated
   - ZeRO-2: optimizer states + gradients + activations partitioned
   - ZeRO-3: ALL components partitioned, 15% framework overhead for gather
   - Single GPU passthrough with no partitioning

3. **REFACTOR: Barrel export** - `7afb590` (refactor)
   - Export calculateZeROMemoryPerGPU, calculateCPUOffloadMemory from deepspeed
   - Export FRAMEWORK_PRESETS and types from frameworks
   - Export ZeROResult type

**Plan metadata:** (pending)

## Files Created/Modified

- `src/engines/frameworks.ts` - Framework preset types and FRAMEWORK_PRESETS configuration constant
- `src/engines/deepspeed.ts` - DeepSpeed ZeRO calculation engine with stage-specific partitioning
- `src/engines/deepspeed.test.ts` - Comprehensive test suite (13 tests)
- `src/engines/types.ts` - Added ZeROResult interface
- `src/engines/index.ts` - Added barrel exports for DeepSpeed engine and framework presets

## Decisions Made

**ZeRO partitioning logic:**
- ZeRO-1: Only optimizer states and activations partitioned (model weights, master weights, gradients replicated)
- ZeRO-2: Optimizer states, gradients, and activations partitioned (model weights, master weights replicated)
- ZeRO-3: ALL components partitioned with 15% framework overhead increase for parameter gather/scatter
- Single GPU passthrough: Returns original breakdown unchanged (no partitioning overhead)

**Framework presets:**
- vLLM and TGI presets have mode 'inference', all others have mode 'training'
- DeepSpeed presets auto-enable gradientCheckpointing and flashAttention
- Unsloth preset additionally auto-enables adamw-8bit optimizer

**CPU offload calculation:**
- Operates on single GPU breakdown (not multi-GPU)
- Apply CPU offload AFTER ZeRO partitioning when using both
- Supports independent offload of optimizer states and/or model parameters

## Deviations from Plan

None - plan executed exactly as written. All ZeRO partitioning logic, CPU offload calculations, and framework preset configurations implemented according to specification.

## Issues Encountered

None - TDD flow proceeded smoothly with failing tests, implementation, and refactoring all passing verification.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 10 Plan 2 (Multi-GPU State Management):**
- Framework preset types and constants available
- DeepSpeed ZeRO calculation engine tested and ready for integration
- CPU offload calculation ready for UI integration
- FRAMEWORK_PRESETS constant provides preset configurations for UI

**Test coverage:**
- 13 DeepSpeed-specific tests
- Full test suite: 266 tests passing
- TypeScript compilation clean
- Biome lint passes

**No blockers or concerns.**

## Self-Check: PASSED

All created files exist:
- FOUND: src/engines/frameworks.ts
- FOUND: src/engines/deepspeed.ts
- FOUND: src/engines/deepspeed.test.ts

All commits exist in git history:
- FOUND: 3c6b911 (RED phase)
- FOUND: 4d32106 (GREEN phase)
- FOUND: 7afb590 (REFACTOR phase)
- FOUND: ebbd0cf (metadata commit)

---
*Phase: 10-framework-presets-multi-gpu-training*
*Completed: 2026-02-10*
