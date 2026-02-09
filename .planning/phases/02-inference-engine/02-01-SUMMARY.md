---
phase: 02-inference-engine
plan: 01
subsystem: engine
tags: [decimal.js, zod, quantization, vram-calculation, type-system]

# Dependency graph
requires:
  - phase: 01-foundation-data
    provides: Project scaffold, TypeScript config, Zod schemas pattern, path aliases
provides:
  - QuantizationFormat type union (13 formats)
  - KVCachePrecision type union (4 formats)
  - InferenceVRAMBreakdown and PerformanceEstimate interfaces
  - CalculationInputSchema with validation
  - BYTES_PER_PARAMETER constants with overhead multipliers
  - getBytesPerParameter and calculateModelWeightVRAM pure functions
affects: [02-02-kv-cache, 02-03-multi-gpu, 02-04-performance]

# Tech tracking
tech-stack:
  added: [Decimal.js for arithmetic precision]
  patterns:
    - Pure functions in engines/ (no React/DOM dependencies)
    - Zod-first validation for calculation inputs
    - Type-only imports for tree-shaking optimization
    - Comprehensive test coverage (100% on engine modules)

key-files:
  created:
    - src/engines/types.ts: Core type system for all engine modules
    - src/engines/constants.ts: Quantization constants with research citations
    - src/engines/quantization.ts: Pure calculation functions
    - src/engines/quantization.test.ts: 21 comprehensive tests
  modified: []

key-decisions:
  - "Use Decimal.js for ALL arithmetic to avoid floating-point precision errors in VRAM calculations"
  - "Bake overhead multipliers into BYTES_PER_PARAMETER constants (GPTQ/AWQ 1.2x, GGUF empirical bpp) rather than calculating at runtime"
  - "Use empirical bits-per-parameter for GGUF formats (from Artefact2 gist) instead of theoretical values"
  - "CalculationInputSchema validates sequence length 512-131072 and batch size 1-64 based on research"

patterns-established:
  - "Engine functions are pure (deterministic, no side effects) for testability and potential Web Worker offloading"
  - "All numeric constants use Decimal.js constructors (new Decimal(1.0)) for consistency"
  - "JSDoc comments include research citations (INFER-XX references) for traceability"

# Metrics
duration: 4min
completed: 2026-02-09
---

# Phase 2 Plan 1: Quantization Engine & Types Summary

**Foundational type system and quantization engine with Decimal.js precision supporting all 13 formats including GPTQ/AWQ 1.2x overhead and GGUF empirical bits-per-parameter**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-09T14:56:32Z
- **Completed:** 2026-02-09T15:00:54Z
- **Tasks:** 2 completed
- **Files created:** 4

## Accomplishments

- Created complete engine type system with QuantizationFormat (13 formats), KVCachePrecision (4 formats), and all calculation interfaces
- Implemented BYTES_PER_PARAMETER constants with proper overhead multipliers: GPTQ/AWQ at 0.6 bytes (1.2x overhead), GGUF formats using empirical bits-per-parameter (Q4_K_M=4.8bpp, Q8_0=8.5bpp)
- Built quantization engine with 100% test coverage validating all formats against known references (7B FP16=~13.04GB, 70B GPTQ=~39.12GB, Mixtral 46.7B FP16=~86.97GB)
- Established pure function pattern for engines using Decimal.js arithmetic throughout

## Task Commits

Each task was committed atomically:

1. **Task 1: Create engine types and constants** - `f149543` (feat)
   - Added QuantizationFormat union type with all 13 formats
   - Added KVCachePrecision, InferenceVRAMBreakdown, PerformanceEstimate interfaces
   - Added CalculationInputSchema with validation (seq len 512-131072, batch 1-64)
   - Added BYTES_PER_PARAMETER, KV_PRECISION_BYTES, FRAMEWORK_OVERHEAD_GB constants

2. **Task 2: Create quantization engine with comprehensive tests** - `6ba1fc0` (test)
   - Created getBytesPerParameter lookup function
   - Created calculateModelWeightVRAM with Decimal.js precision
   - 21 comprehensive tests covering all 13 formats, overhead verification, known references
   - Verified GPTQ/AWQ 1.2x overhead, GGUF empirical bpp, MoE total params usage
   - 100% test coverage on quantization.ts

## Files Created/Modified

**Created:**
- `src/engines/types.ts` - Core engine type system: QuantizationFormat (13 formats), KVCachePrecision (4 formats), InferenceVRAMBreakdown, PerformanceEstimate interfaces, CalculationInputSchema Zod validation
- `src/engines/constants.ts` - Quantization constants: BYTES_PER_PARAMETER map with overhead multipliers (GPTQ/AWQ 0.6, GGUF empirical bpp), KV_PRECISION_BYTES, FRAMEWORK_OVERHEAD_GB (1.0), BYTES_PER_GB conversion constant
- `src/engines/quantization.ts` - Pure calculation functions: getBytesPerParameter (format lookup), calculateModelWeightVRAM (params * bytes / GB with Decimal.js)
- `src/engines/quantization.test.ts` - 21 comprehensive tests: all formats, overhead verification (GPTQ/AWQ >0.5, GGUF empirical), known references (7B/70B/46.7B models), precision verification

**Modified:**
- None

## Decisions Made

**1. Decimal.js for ALL arithmetic**
- Rationale: Native JavaScript number precision errors unacceptable for VRAM calculations (e.g., 13.0000000001 GB)
- Impact: All engine constants use `new Decimal(...)`, all calculations use Decimal methods (.mul, .div, etc.)

**2. Baked overhead multipliers in constants**
- Rationale: GPTQ/AWQ overhead (1.2x) and GGUF empirical bpp are fixed per format, no runtime calculation needed
- Impact: BYTES_PER_PARAMETER is a simple lookup map, getBytesPerParameter is trivial
- Alternative considered: Calculate overhead at runtime → Rejected (unnecessary complexity)

**3. Empirical bits-per-parameter for GGUF**
- Rationale: Real-world GGUF files don't match theoretical bit counts (Q4_K_M is 4.8bpp, not 4.0bpp)
- Source: Artefact2 gist (github.com/Artefact2/d7dc977a6c7288ac784cab80e3f3a700)
- Impact: More accurate VRAM estimates for GGUF quantized models

**4. CalculationInputSchema validation ranges**
- Rationale: Based on research (INFER-06): typical inference uses 512-131072 seq len, 1-64 batch size
- Impact: Catches unrealistic inputs early (e.g., 1M token sequence, 1000 batch size)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**1. Biome import ordering**
- Issue: Biome's organizeImports rule required alphabetical import ordering
- Resolution: Ran `npm run lint:fix` to auto-organize imports in quantization.ts and test file
- Impact: 2 files auto-fixed, no functional change

**2. Test precision expectations**
- Issue: Initial test expected <15 decimal places, but Decimal.js defaults to 18 (which is correct)
- Resolution: Updated test to expect <25 decimal places (accommodates Decimal.js default precision)
- Impact: Test now validates precision without false negatives

**3. 405B calculation reference**
- Issue: Initial test expected 754.19 GB, but correct value is 754.37 GB (405B * 2 / 1024^3)
- Resolution: Recalculated expected value and updated test
- Impact: Test now validates against correct reference

## Next Phase Readiness

**Ready for:**
- 02-02: KV cache calculation (depends on QuantizationFormat, KVCachePrecision types)
- 02-03: Multi-GPU memory split (depends on calculateModelWeightVRAM)
- 02-04: Performance estimation (depends on InferenceVRAMBreakdown, PerformanceEstimate interfaces)

**Provided:**
- Complete type system for all engine modules
- Quantization calculation foundation (other modules will call calculateModelWeightVRAM)
- Pure function pattern established (other engines should follow)
- 100% test coverage standard set

**No blockers.** All dependencies from Phase 1 (project structure, schemas pattern, path aliases) were available and working as expected.

## Self-Check: PASSED

**Files created (4/4):**
- ✓ src/engines/types.ts
- ✓ src/engines/constants.ts
- ✓ src/engines/quantization.ts
- ✓ src/engines/quantization.test.ts

**Commits (2/2):**
- ✓ f149543 (Task 1: feat)
- ✓ 6ba1fc0 (Task 2: test)

All claims in this summary have been verified.

---
*Phase: 02-inference-engine*
*Completed: 2026-02-09*
