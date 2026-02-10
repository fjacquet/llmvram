---
phase: 06-fine-tuning-calculation-engines
plan: 03
subsystem: engines
tags: [lora, qlora, fine-tuning, vram-calculation, parameter-efficient, tdd]

# Dependency graph
requires:
  - phase: 06-fine-tuning-calculation-engines
    plan: 02
    provides: Full fine-tuning engine (calculateOptimizerStateMemory, calculateTrainingActivationMemory, calculateFullFineTuningVRAM)
provides:
  - LoRA/QLoRA VRAM calculation engine
  - calculateLoRAAdapterParams function (adapter parameter count)
  - calculateLoRAFineTuningVRAM function (LoRA breakdown with frozen base)
  - calculateQLoRAFineTuningVRAM function (three-precision architecture)
  - Updated barrel exports with all training/LoRA functions and types
affects: [fine-tuning-ui, training-estimates, 06-fine-tuning-calculation-engines]

# Tech tracking
tech-stack:
  added: []
  patterns: [TDD execution (RED-GREEN), LoRA adapter parameter calculation, QLoRA three-precision architecture, Optimizer states only on trainable params]

key-files:
  created: [src/engines/lora.ts, src/engines/lora.test.ts]
  modified: [src/engines/index.ts, src/engines/training.ts, src/engines/training.test.ts]

key-decisions:
  - "LoRA adapter params = 2 * rank * hiddenSize * targetModuleCount * layers (A + B matrices)"
  - "Optimizer states and gradients apply ONLY to adapter parameters, not frozen base (PITFALLS.md #3)"
  - "QLoRA uses fixed three-precision architecture: NF4 base (0.5 bytes), FP16 adapters (2 bytes), FP32 optimizer (8 bytes for AdamW)"
  - "7B model QLoRA fits in 6-12GB range (research validation: ~10 GB total)"

patterns-established:
  - "LoRA engine follows existing patterns: pure functions, Decimal.js, comprehensive JSDoc"
  - "TDD execution: RED (failing tests) → GREEN (minimal implementation)"
  - "Test coverage: adapter param counts, optimizer state targeting, QLoRA three-precision validation"

# Metrics
duration: 5min 47sec
completed: 2026-02-10
---

# Phase 06 Plan 03: LoRA/QLoRA Engine Summary

**TDD implementation of parameter-efficient fine-tuning VRAM calculations with adapter-only optimizer states and QLoRA three-precision architecture**

## Performance

- **Duration:** 5 minutes 47 seconds
- **Started:** 2026-02-10T14:07:50Z
- **Completed:** 2026-02-10T14:13:37Z
- **Tasks:** 3 (TDD: RED + GREEN + barrel exports)
- **Files created:** 2
- **Files modified:** 3
- **Tests:** 13 (all passing)

## Accomplishments

- **calculateLoRAAdapterParams**: Computes adapter parameter count based on rank, target modules (% of 7 per layer), hidden size, and layers. Validates 7B model with rank=16, 50% targets → ~16.8M params (0.24% of base).

- **calculateLoRAFineTuningVRAM**: Complete LoRA breakdown with frozen base at training precision, trainable FP16 adapters, FP32 master weights (mixed precision), and optimizer states/gradients ONLY on adapters. 7B model total: ~19.8 GB (fits 24GB GPU).

- **calculateQLoRAFineTuningVRAM**: QLoRA three-precision architecture (NF4 base, FP16 adapters, FP32 optimizer) with no precision parameter (fixed architecture for stability). 7B model total: ~10 GB (within 6-12GB research range, fits 12GB GPU).

- **Barrel exports updated**: All training and LoRA functions, types, and constants now exported from `src/engines/index.ts` for clean imports.

- **Critical validation**: Optimizer states apply ONLY to adapter parameters (~16.8M), NOT frozen base (7B) - avoiding PITFALL #3 (30-60% underestimation).

## Task Commits

TDD execution with atomic commits:

1. **Task 1 (RED): Add failing tests** - `45e1be2` (test)
   - 13 failing tests for all three functions
   - Validates adapter params, LoRA breakdown, QLoRA three-precision

2. **Task 2 (GREEN): Implement passing code** - `b792588` (feat)
   - Full implementation with comprehensive JSDoc
   - Fixed test models to use correct schema fields (num_hidden_layers, num_attention_heads)
   - Fixed training.ts to use correct field names

3. **Task 3: Update barrel exports** - `5b484ca` (feat)
   - Export all LoRA/training functions and types
   - Alphabetical ordering maintained

## Files Created/Modified

**Created:**
- `src/engines/lora.ts` - Core LoRA/QLoRA VRAM calculation engine with 3 exported functions
- `src/engines/lora.test.ts` - Comprehensive test suite with 13 tests

**Modified:**
- `src/engines/index.ts` - Added training and LoRA exports (functions, types, constants)
- `src/engines/training.ts` - Fixed num_layers → num_hidden_layers
- `src/engines/training.test.ts` - Fixed test models to use correct schema fields

## Decisions Made

None - followed plan exactly as specified with TDD methodology

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect Model schema field names**
- **Found during:** GREEN phase implementation
- **Issue:** Test models and training.ts used `num_layers`, `num_heads` but Model schema defines `num_hidden_layers`, `num_attention_heads`. TypeScript build failed.
- **Fix:** Updated all test models in lora.test.ts and training.test.ts to use correct schema fields. Updated training.ts calculateTrainingActivationMemory to use `model.num_hidden_layers`.
- **Files modified:** src/engines/lora.test.ts, src/engines/training.test.ts, src/engines/training.ts
- **Commit:** b792588 (included in GREEN commit)

No other deviations - plan executed exactly as written with TDD methodology.

## Issues Encountered

None - clean TDD execution with all tests passing after schema field fix.

## User Setup Required

None - no external service configuration required

## Next Phase Readiness

**Ready for UI integration:** Complete LoRA/QLoRA calculation engine with all functions exported. Ready for Phase 07 (Fine-Tuning UI) to build input components and integrate with calculation engine.

**Blockers:** None

**Next steps:** Build fine-tuning UI with training method selector (Full/LoRA/QLoRA), optimizer selector, LoRA rank/target module inputs, and VRAM breakdown display.

## Self-Check: PASSED

All created files verified:
- FOUND: src/engines/lora.ts
- FOUND: src/engines/lora.test.ts

All commits verified:
- FOUND: 45e1be2 (RED - failing tests)
- FOUND: b792588 (GREEN - passing implementation + schema fixes)
- FOUND: 5b484ca (barrel exports)

All tests verified:
```
Test Files  16 passed (16)
     Tests  213 passed (213)
  Duration  1.73s
```

TypeScript check: PASSED
Biome lint: PASSED
Production build: PASSED

Key validations:
- ✅ 7B LoRA adapter params (rank=16, 50% targets): ~16.8M (0.24% of base)
- ✅ 7B LoRA total VRAM: ~19.8 GB (fits 24GB GPU)
- ✅ 7B QLoRA total VRAM: ~10 GB (within 6-12GB research range)
- ✅ Optimizer states only count adapter params (~0.125 GB), not frozen base (~52 GB)
- ✅ QLoRA three-precision architecture validated (NF4 base, FP16 adapters, FP32 optimizer)
- ✅ All training/LoRA functions importable from @engines/index

---
*Phase: 06-fine-tuning-calculation-engines*
*Completed: 2026-02-10*
