---
phase: 02-inference-engine
plan: 02
subsystem: engine
tags: [kv-cache, gqa, mqa, moe, inference-vram, activation-memory, decimal.js]

# Dependency graph
requires:
  - phase: 01-foundation-data
    provides: Model schema with GQA/MoE fields (num_kv_heads, num_experts, num_experts_per_token)
  - phase: 02-01
    provides: QuantizationFormat, KVCachePrecision types, BYTES_PER_GB constant, calculateModelWeightVRAM function
provides:
  - calculateKVCacheVRAM with GQA/MQA ratio handling (8x reduction for Llama 3 70B)
  - calculateInferenceVRAM with full VRAM breakdown (weights + KV + activations + overhead)
  - calculateMoEActiveParams for MoE activation sizing
  - calculateActivationMemory with MoE support
affects: [02-03-multi-gpu, 02-04-performance, 03-ui-components]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - GQA/MQA ratio calculation via (num_kv_heads ?? num_attention_heads) / num_attention_heads
    - MoE models use TOTAL params for weights, ACTIVE params for activations
    - KV cache quantization independent from weight quantization (INFER-05)
    - Full VRAM breakdown returned as structured object (InferenceVRAMBreakdown)

key-files:
  created:
    - src/engines/kv-cache.ts: KV cache calculation with GQA/MQA support
    - src/engines/kv-cache.test.ts: 8 comprehensive tests for GQA reduction, scaling, precision
    - src/engines/inference.ts: Main VRAM calculator with MoE support
    - src/engines/inference.test.ts: 17 comprehensive tests for dense/MoE models, all quantizations
  modified: []

key-decisions:
  - "GQA ratio fallback: If num_kv_heads undefined, default to num_attention_heads (standard MHA, ratio=1.0)"
  - "MoE active params: 20% shared + 80% expert * (num_experts_per_token / num_experts) for activation sizing"
  - "KV cache formula: 2 * layers * hidden_size * seq_len * batch * precision * gqa_ratio / BYTES_PER_GB"
  - "Total inference VRAM: weights + KV cache + activations + 1GB framework overhead"

patterns-established:
  - "KV cache scales linearly with sequence length and batch size (verified in tests)"
  - "Activation memory uses FP32 precision (4 bytes) regardless of weight quantization"
  - "All VRAM breakdown components returned as Decimal instances for precision"

# Metrics
duration: 5min
completed: 2026-02-09
---

# Phase 2 Plan 2: KV Cache & Inference Engine Summary

**KV cache engine with GQA/MQA support achieving 8x reduction for Llama 3 70B, and full inference VRAM calculator correctly handling MoE models using total params for weights (46.7B Mixtral) and active params for activations**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-09T15:04:26Z
- **Completed:** 2026-02-09T15:10:21Z
- **Tasks:** 2 completed
- **Files created:** 4

## Accomplishments

- Implemented KV cache calculation with proper GQA/MQA ratio handling, achieving verified 8x reduction for Llama 3 70B (8 KV heads / 64 query heads) and 32x for MQA models
- Built complete inference VRAM engine that correctly uses TOTAL parameters for MoE model weights (46.7B for Mixtral, not 13B active) while using active parameters for activation memory sizing
- Enabled independent KV cache quantization (INFER-05) allowing users to optimize memory with int4/fp8 KV cache while maintaining fp16/fp32 weights
- Achieved 100% test coverage on both engines with 25 comprehensive test scenarios covering dense/MoE models, all quantization formats, and edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Create KV cache engine with GQA/MQA support and tests** - `0fa3c19` (feat)
   - Implemented calculateKVCacheVRAM with GQA ratio: (num_kv_heads ?? num_attention_heads) / num_attention_heads
   - Verified 8x KV cache reduction for Llama 3 70B GQA vs MHA equivalent
   - Tested sequence length scaling (2x seq = 2x cache), batch scaling (4x batch = 4x cache)
   - Tested KV precision independence (int4 is 4x smaller than fp16)
   - Tested MQA with single KV head (32x reduction)
   - 8 comprehensive tests with 100% coverage

2. **Task 2: Create main inference VRAM engine with tests** - `b821350` (feat)
   - Implemented calculateMoEActiveParams for MoE activation sizing (20% shared + 80% expert * active_ratio)
   - Implemented calculateActivationMemory with MoE support (uses active params, not total)
   - Implemented calculateInferenceVRAM combining weights + KV + activations + 1GB overhead
   - Verified Llama 3 70B GPTQ fits on H100 80GB (~41.83 GB total)
   - Verified Mixtral 8x7B uses 46.7B for weights (~86.986 GB FP16), not 13B active
   - 17 comprehensive tests with 100% coverage

## Files Created/Modified

**Created:**
- `src/engines/kv-cache.ts` - KV cache calculation: calculateKVCacheVRAM with GQA/MQA ratio handling. Formula: 2 * layers * hidden_size * seq_len * batch * precision * gqa_ratio. Supports all 4 KV precision formats (fp16, fp8, int8, int4).
- `src/engines/kv-cache.test.ts` - 8 comprehensive tests: GQA 8x reduction verification, known reference value (Llama 3 70B = 1.25 GB), sequence/batch scaling, KV precision (int4 vs fp16), MQA 32x reduction, MHA fallback
- `src/engines/inference.ts` - Main VRAM calculator: calculateInferenceVRAM (full breakdown), calculateMoEActiveParams (20/80 split), calculateActivationMemory (FP32, uses active params for MoE). Returns InferenceVRAMBreakdown with all Decimal components.
- `src/engines/inference.test.ts` - 17 comprehensive tests: activation memory (dense/MoE scaling), MoE active params (Mixtral 18.68B active from 46.7B total), inference VRAM (Llama 70B GPTQ on H100, Mixtral total vs active params, KV quantization independence, breakdown structure, all quantization formats)

**Modified:**
- None

## Decisions Made

**1. GQA ratio fallback behavior**
- Rationale: Models without num_kv_heads field should work (standard MHA behavior)
- Implementation: `gqaRatio = (model.num_kv_heads ?? model.num_attention_heads) / model.num_attention_heads`
- Impact: Llama 2 models (no GQA) correctly calculate with ratio=1.0

**2. MoE active parameter calculation**
- Rationale: Activation memory depends on active experts, not total params (only some experts fire per token)
- Formula: `sharedParams (20%) + expertParams (80%) * (num_experts_per_token / num_experts)`
- Impact: Mixtral 8x7B activations use ~18.68B active, not 46.7B total (more accurate memory estimate)

**3. Activation memory uses FP32 regardless of weight quantization**
- Rationale: Activations are always computed in higher precision (FP32) even if weights are quantized
- Impact: Factor of 4 bytes per activation element, not tied to weight quantization format

**4. Independent KV cache quantization (INFER-05)**
- Rationale: Modern frameworks support per-layer KV quantization separate from weight quantization
- Implementation: `kvQuantization` parameter defaults to 'fp16', independent from `quantization`
- Impact: Users can optimize memory with int4 KV cache while maintaining fp16 weights

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed plan calculation error in KV cache reference test**
- **Found during:** Task 1 (KV cache test implementation)
- **Issue:** Plan specified expected value of 0.9375 GB for Llama 3 70B KV cache, but correct calculation is 1.25 GB (2 * 80 * 8192 * 4096 * 1 * 2 * 0.125 / 1024^3 = 1,342,177,280 / 1,073,741,824 = 1.25)
- **Fix:** Updated test expectation from 0.9375 GB to 1.25 GB with correct calculation comment
- **Files modified:** src/engines/kv-cache.test.ts (lines 70, 89)
- **Verification:** All 8 KV cache tests pass, GQA 8x reduction test confirms correctness
- **Committed in:** 0fa3c19 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed precision in MoE weight test expectation**
- **Found during:** Task 2 (Inference test implementation)
- **Issue:** Test expected 86.97 GB but actual calculation is 86.986 GB (46.7 * 1e9 * 2 / 1024^3)
- **Fix:** Updated test expectation from 86.97 to 86.986 GB with corrected comment
- **Files modified:** src/engines/inference.test.ts (lines 216, 220)
- **Verification:** All 17 inference tests pass, total equals sum of components
- **Committed in:** b821350 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs in plan test expectations)
**Impact on plan:** Both fixes corrected arithmetic errors in expected test values. No functional changes to engine logic. Plan's formulas were correct; only manual calculations had errors.

## Issues Encountered

**1. Biome import ordering and formatting**
- Issue: Biome required import organization and formatting fixes after file creation
- Resolution: Ran `npm run lint:fix` to auto-organize imports and format code
- Impact: 2 files auto-fixed (inference.test.ts), no functional change

## Next Phase Readiness

**Ready for:**
- 02-03: Multi-GPU memory distribution (depends on calculateInferenceVRAM for per-GPU VRAM allocation)
- 02-04: Performance estimation (depends on InferenceVRAMBreakdown, can estimate tokens/sec based on memory bandwidth)
- 03-XX: UI components (can call calculateInferenceVRAM to display VRAM breakdown in real-time)

**Provided:**
- Complete KV cache calculation with GQA/MQA support (8x reduction validated)
- Full inference VRAM breakdown with MoE handling (total params for weights, active for activations)
- Independent KV cache quantization capability (INFER-05 implemented)
- 100% test coverage on all calculation functions

**Critical validations:**
- ✓ Llama 3 70B GPTQ fits on H100 80GB (41.83 GB total)
- ✓ GQA produces exact 8x reduction (64/8 = 8.0)
- ✓ MQA produces exact 32x reduction (32/1 = 32.0)
- ✓ Mixtral weights use 46.7B total, not 13B active (~87 GB vs ~24 GB)
- ✓ KV cache int4 is exactly 4x smaller than fp16
- ✓ Sequence/batch scaling is perfectly linear

**No blockers.** All dependencies from Phase 1 (Model schema with num_kv_heads, num_experts fields) and 02-01 (types, constants, quantization engine) were available and working as expected.

## Self-Check: PASSED

**Files created (4/4):**
- ✓ src/engines/kv-cache.ts
- ✓ src/engines/kv-cache.test.ts
- ✓ src/engines/inference.ts
- ✓ src/engines/inference.test.ts

**Commits (2/2):**
- ✓ 0fa3c19 (Task 1: feat - KV cache engine)
- ✓ b821350 (Task 2: feat - Inference engine)

All claims in this summary have been verified.

---
*Phase: 02-inference-engine*
*Completed: 2026-02-09*
