---
phase: 06-fine-tuning-calculation-engines
plan: 01
subsystem: engines
tags: [training, fine-tuning, types, schemas, zod, typescript, vram-calculation, optimizer, lora, qlora]

# Dependency graph
requires:
  - phase: 02-inference-engine
    provides: Base types (QuantizationFormat, Decimal), constants pattern, schema validation pattern
provides:
  - Training type definitions (OptimizerType, FineTuningMethod, TrainingPrecision, TrainingVRAMBreakdown, LoRAVRAMBreakdown)
  - Training constants (OPTIMIZER_STATE_BYTES, TRAINING_FRAMEWORK_OVERHEAD_GB, GRADIENT_BYTES, WEIGHT_BYTES, MASTER_WEIGHT_BYTES, TOTAL_TARGETABLE_MODULES_PER_LAYER)
  - TrainingInputSchema for training configuration validation
affects: [06-fine-tuning-calculation-engines, fine-tuning-ui, training-estimates]

# Tech tracking
tech-stack:
  added: []
  patterns: [Training type extensions, Training-specific Zod schemas, FP32 optimizer state constants]

key-files:
  created: []
  modified: [src/engines/types.ts, src/engines/constants.ts, src/utils/schemas.ts]

key-decisions:
  - "Optimizer states always FP32 bytes (8 for AdamW, 4 for SGD, 2 for 8-bit, 4 for Adafactor) even in mixed precision"
  - "Training framework overhead is 1.5GB (higher than inference 1.0GB) due to autograd + optimizer buffers"
  - "Standard transformer has 7 targetable modules per layer for LoRA (4 attention + 3 MLP)"

patterns-established:
  - "Training types follow inference pattern: type unions → interfaces → breakdown structures"
  - "Constants use Record<Type, Decimal> for type-safe mapping with precision arithmetic"
  - "Zod schemas with defaults and validation ranges (batchSize 1-128, sequenceLength 512-131072)"

# Metrics
duration: 2min
completed: 2026-02-10
---

# Phase 06 Plan 01: Training Foundation Summary

**Training types and schemas with FP32 optimizer states, mixed precision constants, and LoRA parameter validation**

## Performance

- **Duration:** 2 minutes
- **Started:** 2026-02-10T13:57:09Z
- **Completed:** 2026-02-10T13:59:24Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Comprehensive training type system (OptimizerType, FineTuningMethod, TrainingPrecision, TrainingVRAMBreakdown, LoRAVRAMBreakdown)
- Training-specific constants with correct FP32 optimizer state sizes (AdamW: 8 bytes/param, SGD: 4 bytes/param, 8-bit: 2 bytes/param, Adafactor: 4 bytes/param)
- TrainingInputSchema for validating fine-tuning configuration (method, optimizer, precision, batch/sequence, LoRA params)
- Critical pitfall documented: Optimizer states always FP32, even in mixed precision training

## Task Commits

Each task was committed atomically:

1. **Task 1: Add training types and constants** - `a2725d4` (feat)
2. **Task 2: Add TrainingInputSchema to schemas.ts** - `9f5fcf5` (feat)

## Files Created/Modified
- `src/engines/types.ts` - Added OptimizerType, FineTuningMethod, TrainingPrecision unions; TrainingVRAMBreakdown and LoRAVRAMBreakdown interfaces with JSDoc referencing PITFALLS.md
- `src/engines/constants.ts` - Added OPTIMIZER_STATE_BYTES (Record<OptimizerType, Decimal>), TRAINING_FRAMEWORK_OVERHEAD_GB (1.5GB), GRADIENT_BYTES, WEIGHT_BYTES, MASTER_WEIGHT_BYTES, TOTAL_TARGETABLE_MODULES_PER_LAYER (7)
- `src/utils/schemas.ts` - Added TrainingInputSchema validating method/optimizer/precision/batch/sequence/LoRA parameters; exported TrainingInput type and validateTrainingInput helper

## Decisions Made
None - followed plan as specified

## Deviations from Plan

None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for next plan (06-02):** Training calculation engines foundation is complete. All types, constants, and schemas in place for implementing full fine-tuning, LoRA, and QLoRA VRAM calculation engines.

**Blockers:** None

**Next steps:** Implement full fine-tuning engine using TrainingVRAMBreakdown with correct optimizer state calculation (8B params × 8 bytes = 64GB just for AdamW states).

## Self-Check: PASSED

All modified files verified:
- FOUND: src/engines/types.ts
- FOUND: src/engines/constants.ts
- FOUND: src/utils/schemas.ts

All commits verified:
- FOUND: a2725d4 (Task 1)
- FOUND: 9f5fcf5 (Task 2)

---
*Phase: 06-fine-tuning-calculation-engines*
*Completed: 2026-02-10*
