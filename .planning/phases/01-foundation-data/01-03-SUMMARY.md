---
phase: 01-foundation-data
plan: 03
subsystem: database
tags: [zod, vitest, llm, models, huggingface, moe, gqa]

# Dependency graph
requires:
  - phase: 01-01
    provides: Zod ModelSchema and validation helpers
provides:
  - Curated model database with 30 popular LLMs (LLaMA 2/3, Mistral, Mixtral, Qwen, Phi, DeepSeek, Gemma, Command-R)
  - Accurate architecture parameters from HuggingFace specs (hidden_size, layers, attention heads, intermediate_size)
  - Correct MoE total parameters (Mixtral 8x7B = 46.7B, not 13B active)
  - GQA num_kv_heads specifications for all models using Grouped Query Attention
  - Comprehensive validation test suite ensuring data integrity
affects: [01-04, 02-01, engines, ui, calculations]

# Tech tracking
tech-stack:
  added: []
  patterns: [curated-database, zod-validation, comprehensive-test-coverage]

key-files:
  created:
    - src/data/models.json
    - src/utils/models.test.ts
  modified: []

key-decisions:
  - "Used plan specifications directly rather than fetching from HuggingFace API (specs already vetted)"
  - "MoE models specify total parameters (46.7B for Mixtral 8x7B) not active parameters (13B)"
  - "All GQA models include num_kv_heads field for accurate KV cache calculations"

patterns-established:
  - "Model database follows HuggingFace config.json field naming conventions"
  - "Test suite validates both structural integrity and critical domain requirements (MoE params, GQA heads)"

# Metrics
duration: 3min
completed: 2026-02-09
---

# Phase 1 Plan 3: Model Database Summary

**Curated database of 30 popular LLMs with accurate MoE total parameters (46.7B for Mixtral 8x7B) and GQA num_kv_heads specifications**

## Performance

- **Duration:** 3 minutes (181 seconds)
- **Started:** 2026-02-09T14:05:57Z
- **Completed:** 2026-02-09T14:08:58Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created comprehensive model database with 30 LLMs across 8 major families (LLaMA 2/3, Mistral, Mixtral, Qwen, Phi, DeepSeek, Gemma, Command-R)
- Correctly specified total parameters for MoE models (Mixtral 8x7B = 46.7B, avoiding research pitfall #1)
- Included num_kv_heads for all GQA models (LLaMA 3.1, Mistral, Qwen, etc.) to prevent KV cache overestimation (research pitfall #2)
- Comprehensive test suite with 19 tests validating model families, architecture types, parameter accuracy, and schema compliance

## Task Commits

Each task was committed atomically:

1. **Task 1: Create model database with 30+ validated entries** - `7b554cc` (feat)
   - Created src/data/models.json with 30 popular LLM entries
   - Includes LLaMA 2 (3 models), LLaMA 3.1 (4 models), Mistral (2), Mixtral (3)
   - Includes Qwen (4), Phi (3), DeepSeek (3), Gemma (3), Command-R (2)
   - Includes Yi, Falcon, and MPT models
   - MoE models specify total parameters (Mixtral 8x7B = 46.7B, not 13B active)
   - GQA models correctly specify num_kv_heads (LLaMA 3.1, Mistral, Qwen)
   - All entries conform to ModelSchema structure

2. **Task 2: Write validation tests for model data** - `e904b52` (test)
   - Created src/utils/models.test.ts with 19 comprehensive tests
   - Validates minimum 30 models in database
   - Verifies all required model families present
   - Tests MoE models use total parameters (Mixtral 8x7B = 46.7B)
   - Tests GQA models specify num_kv_heads correctly
   - Validates all entries pass Zod schema validation
   - Verifies unique model IDs and valid architecture configurations
   - Fixed test logic to allow intermediate_size exactly 8x hidden_size

## Files Created/Modified

**Database:**
- `src/data/models.json` - Curated database of 30 popular LLM models with accurate architecture parameters

**Tests:**
- `src/utils/models.test.ts` - Comprehensive validation test suite with 19 tests covering model families, MoE parameters, GQA specifications, and schema validation

## Decisions Made

**1. Used plan specifications directly instead of HuggingFace API**
- Rationale: Plan already provided accurate, vetted specifications from HuggingFace config.json files. Direct API calls would add complexity and potential for rate limiting without adding value since specs were pre-verified.

**2. MoE models use total parameters, not active parameters**
- Rationale: Research pitfall #1 documented that MoE models load ALL parameters into VRAM (Mixtral 8x7B = 46.7B total), not just active parameters (13B). Critical for accurate VRAM calculations.

**3. All GQA models include num_kv_heads field**
- Rationale: Research pitfall #2 documented that GQA models have fewer KV heads than query heads, affecting KV cache size by 2-8x. Essential for accurate memory estimation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test logic for intermediate_size ratio validation**
- **Found during:** Task 2 verification (npm run test failed)
- **Issue:** Test expected intermediate_size/hidden_size ratio to be less than 8, but Gemma 2B has exactly 8x ratio, causing test failure
- **Fix:** Changed `expect(ratio).toBeLessThan(8)` to `expect(ratio).toBeLessThanOrEqual(8)` to allow exactly 8x ratio
- **Files modified:** src/utils/models.test.ts
- **Verification:** All 19 tests pass after fix
- **Committed in:** e904b52 (Task 2 commit)

**2. [Rule 3 - Blocking] Auto-fixed Biome import organization**
- **Found during:** Task 2 verification (npm run lint failed)
- **Issue:** Imports in models.test.ts not organized alphabetically per Biome rules
- **Fix:** Ran `npm run lint:fix` to auto-organize imports
- **Files modified:** src/utils/models.test.ts
- **Verification:** npm run lint passed with zero errors after fix
- **Committed in:** e904b52 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both auto-fixes essential for test correctness and code quality standards. No scope creep.

## Issues Encountered

None - all tasks executed as planned after auto-fixes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready:**
- Model database complete with 30+ popular LLMs
- All critical architecture parameters captured (MoE total params, GQA num_kv_heads)
- Comprehensive test coverage ensures data integrity
- Database ready for use in VRAM calculation engines

**Blockers:**
None

**Concerns:**
None

## Self-Check

Verifying all claimed files and commits exist:

**Files created (2):**
- src/data/models.json: EXISTS (342 lines)
- src/utils/models.test.ts: EXISTS (148 lines)

**Commits:**
- 7b554cc: EXISTS (Task 1 - model database)
- e904b52: EXISTS (Task 2 - validation tests)

**Verification commands:**
- npm run test: PASSED (30 tests, all passing)
- npm run typecheck: PASSED (no type errors)
- npm run lint: PASSED (no linting errors)

**Model database validation:**
- Total models: 30 ✓
- LLaMA 2 models: 3 ✓
- LLaMA 3.1 models: 4 ✓
- Mistral models: 2 ✓
- Mixtral MoE models: 3 ✓
- Qwen models: 4 ✓
- Phi models: 3 ✓
- DeepSeek models: 3 ✓
- Gemma models: 3 ✓
- Command-R models: 2 ✓
- Additional models: 3 (Yi, Falcon, MPT) ✓

**Critical validations:**
- Mixtral 8x7B total params: 46.7B (not 13B) ✓
- GQA models have num_kv_heads: ✓
  - LLaMA 3.1 8B: num_kv_heads=8, num_attention_heads=32 ✓
  - Mistral 7B: num_kv_heads=8, num_attention_heads=32 ✓
  - Qwen 2.5 7B: num_kv_heads=4, num_attention_heads=28 ✓

## Self-Check: PASSED

All files exist, all commits present, all verifications passed, all critical validations confirmed.

---
*Phase: 01-foundation-data*
*Completed: 2026-02-09*
