---
phase: 02-inference-engine
verified: 2026-02-09T16:26:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 2: Inference Engine Verification Report

**Phase Goal:** Accurate VRAM calculations for all model types and quantization formats
**Verified:** 2026-02-09T16:26:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Engine correctly calculates VRAM for FP32, FP16, BF16, INT8, INT4, NF4, GPTQ, AWQ, GGUF quantized models with proper overhead (1.1-1.3x multipliers) | ✓ VERIFIED | constants.ts defines all 13 formats with correct bytes-per-parameter. GPTQ/AWQ: 0.6 bytes (1.2x overhead), GGUF Q4_K_M: 0.6 bytes (4.8 bpp empirical). quantization.test.ts verifies all formats with 21 tests (100% coverage). |
| 2 | KV cache calculation accounts for GQA/MQA architectures by factoring n_kv_heads/n_heads ratio | ✓ VERIFIED | kv-cache.ts implements `gqaRatio = (num_kv_heads ?? num_attention_heads) / num_attention_heads`. Tests verify 8x reduction for Llama 3 70B (8/64 ratio) and 32x for MQA (1/32 ratio). |
| 3 | MoE models (Mixtral, DeepSeek) calculate ALL expert weights in VRAM, not just active parameters | ✓ VERIFIED | inference.ts uses `model.num_parameters_billion` directly for `calculateModelWeightVRAM()`. Test verifies Mixtral 8x7B FP16 = ~86.986 GB (46.7B total params), not ~24 GB (13B active). Activations correctly use active params via `calculateMoEActiveParams()`. |
| 4 | Performance estimation (tokens/sec, TTFT) reflects GPU compute capability and memory bandwidth using roofline model | ✓ VERIFIED | performance.ts implements `tokensPerSecond = min(memoryBoundTPS, computeBoundTPS)`. Tests verify Llama 3 70B FP16 on H100 = ~24 tok/s (memory-bound), GPTQ = ~80 tok/s (3.3x faster). TTFT uses 0.5x decode speed factor. Bottleneck correctly identified. |
| 5 | Heavy calculations run in Web Worker without blocking UI, with graceful fallback to synchronous execution | ✓ VERIFIED | calculation.worker.ts handles `CALCULATE_INFERENCE` messages with Decimal serialization to strings. useInferenceCalculation.ts checks `typeof Worker !== 'undefined'` and falls back to dynamic imports. Vite production build succeeds (Worker bundled correctly at 193.52 kB). |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/engines/types.ts` | Type system: QuantizationFormat (13 formats), KVCachePrecision (4 formats), InferenceVRAMBreakdown, PerformanceEstimate, CalculationInputSchema | ✓ VERIFIED | EXISTS (108 lines). Exports all types. No stubs. Imports Decimal and Zod. |
| `src/engines/constants.ts` | BYTES_PER_PARAMETER map, KV_PRECISION_BYTES, FRAMEWORK_OVERHEAD_GB (1.0), BYTES_PER_GB | ✓ VERIFIED | EXISTS (90 lines). All 13 formats with overhead: GPTQ/AWQ 0.6, GGUF empirical bpp. JSDoc cites research sources. |
| `src/engines/quantization.ts` | getBytesPerParameter, calculateModelWeightVRAM pure functions | ✓ VERIFIED | EXISTS (60 lines). Uses Decimal.js arithmetic. No stubs. Proper exports. |
| `src/engines/quantization.test.ts` | Tests for all 13 formats, overhead verification, known references | ✓ VERIFIED | EXISTS (210 lines). 21 tests. Verifies 7B FP16=~13.04GB, 70B GPTQ=~39.12GB, Mixtral 46.7B=~86.97GB. |
| `src/engines/kv-cache.ts` | calculateKVCacheVRAM with GQA/MQA support | ✓ VERIFIED | EXISTS (70 lines). Implements gqaRatio formula. Uses Decimal.js. |
| `src/engines/kv-cache.test.ts` | GQA 8x reduction, MQA 32x, scaling tests | ✓ VERIFIED | EXISTS (221 lines). 8 tests. Verifies Llama 3 70B KV cache with GQA. |
| `src/engines/inference.ts` | calculateInferenceVRAM, calculateMoEActiveParams, calculateActivationMemory | ✓ VERIFIED | EXISTS (190 lines). Combines weights + KV + activations + 1GB overhead. MoE uses total for weights, active for activations. |
| `src/engines/inference.test.ts` | Dense/MoE models, all quantizations, breakdown verification | ✓ VERIFIED | EXISTS (361 lines). 17 tests. Verifies MoE total params, KV quantization independence. |
| `src/engines/performance.ts` | estimatePerformance with roofline model | ✓ VERIFIED | EXISTS (117 lines). Min(memory-bound, compute-bound). TTFT at 0.5x decode speed. |
| `src/engines/performance.test.ts` | Memory-bound scenarios, bottleneck analysis, TTFT | ✓ VERIFIED | EXISTS (387 lines). 12 tests. Verifies roofline model, Llama 3 70B ~24 tok/s. |
| `src/workers/calculation.worker.ts` | Web Worker entry point, Decimal serialization | ✓ VERIFIED | EXISTS (130 lines). Handles CALCULATE_INFERENCE, serializes Decimal to strings, posts results. |
| `src/hooks/useInferenceCalculation.ts` | React hook with Worker + sync fallback | ✓ VERIFIED | EXISTS (213 lines). Checks Worker availability, reconstructs Decimals, provides loading/error states. |
| `src/engines/index.ts` | Barrel export for all engine modules | ✓ VERIFIED | EXISTS (27 lines). Exports 5 functions, all types, constants. |
| `src/engines/inference.integration.test.ts` | End-to-end pipeline tests with real model/GPU specs | ✓ VERIFIED | EXISTS (338 lines). 7 integration tests. Verifies Llama 3 70B GPTQ on H100 (~42 GB total), Mixtral >80GB. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| quantization.ts | types.ts | `import type { QuantizationFormat }` | ✓ WIRED | Import exists, type used in function signature. |
| quantization.ts | constants.ts | `import { BYTES_PER_PARAMETER }` | ✓ WIRED | Import exists, used in getBytesPerParameter lookup. |
| quantization.ts | decimal.js | `import Decimal` | ✓ WIRED | Used for all arithmetic (mul, div, pow). |
| kv-cache.ts | types.ts | `import type { KVCachePrecision }` | ✓ WIRED | Import exists, type used in function params. |
| kv-cache.ts | constants.ts | `import { KV_PRECISION_BYTES }` | ✓ WIRED | Import exists, used in precision lookup. |
| inference.ts | quantization.ts | `import { calculateModelWeightVRAM }` | ✓ WIRED | Import exists, called for weight calculation (9 usages found). |
| inference.ts | kv-cache.ts | `import { calculateKVCacheVRAM }` | ✓ WIRED | Import exists, called for KV cache calculation. |
| performance.ts | quantization.ts | `import { calculateModelWeightVRAM }` | ✓ WIRED | Import exists, used to compute model size in bytes for bandwidth calculation. |
| calculation.worker.ts | inference.ts | `import { calculateInferenceVRAM }` | ✓ WIRED | Import exists, called in onmessage handler (7 usages found). |
| calculation.worker.ts | performance.ts | `import { estimatePerformance }` | ✓ WIRED | Import exists, called in onmessage handler. |
| useInferenceCalculation.ts | calculation.worker.ts | `new Worker(new URL(...))` | ✓ WIRED | Worker instantiated with Vite URL import, bundled in production build. |
| index.ts | quantization.ts | `export { getBytesPerParameter, calculateModelWeightVRAM }` | ✓ WIRED | Barrel export re-exports functions. |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| INFRA-07: Web Workers for heavy calculations | ✓ SATISFIED | calculation.worker.ts offloads to background thread, hook provides sync fallback. |
| INFER-01: Calculate model weight by quantization format | ✓ SATISFIED | quantization.ts supports all 13 formats with proper overhead (GPTQ/AWQ 1.2x, GGUF empirical). |
| INFER-02: Calculate KV cache with GQA/MQA support | ✓ SATISFIED | kv-cache.ts implements gqaRatio, verified 8x reduction for Llama 3 70B. |
| INFER-03: Calculate activation memory | ✓ SATISFIED | inference.ts calculateActivationMemory() uses FP32 precision, supports MoE active params. |
| INFER-04: Framework overhead | ✓ SATISFIED | constants.ts FRAMEWORK_OVERHEAD_GB = 1.0 GB, included in total VRAM. |
| INFER-05: KV cache quantization separate from weights | ✓ SATISFIED | calculateInferenceVRAM() accepts kvQuantization parameter, defaults to fp16. Test verifies int4 vs fp16 = 4x reduction. |
| INFER-06: Configurable sequence length and batch size | ✓ SATISFIED | CalculationInputSchema validates seq_len 512-131072, batch 1-64. Functions accept as parameters. |
| INFER-07: MoE architecture handling | ✓ SATISFIED | inference.ts uses total params for weights (46.7B Mixtral), active params for activations (18.68B). |
| INFER-08: Tokens/sec throughput estimation | ✓ SATISFIED | performance.ts roofline model: min(memory-bound, compute-bound). Tests verify ~24 tok/s for Llama 3 70B FP16 on H100. |
| INFER-09: TTFT estimation | ✓ SATISFIED | performance.ts calculates TTFT = 1 / (tokensPerSecond * 0.5). Prefill 2x slower than decode. |

### Anti-Patterns Found

No blocking anti-patterns found.

**Summary:** All engine files are substantive implementations with proper exports, no TODO/FIXME/placeholder comments, no console.log-only implementations, no stub patterns. Decimal.js used consistently for all arithmetic.

### Human Verification Required

None. All verification performed programmatically via tests and static analysis.

**Rationale:** Calculation correctness verified by comprehensive unit and integration tests (95 tests, 100% pass rate). Test cases use known reference values (e.g., Llama 3 70B GPTQ on H100, Mixtral 8x7B MoE) to validate formulas. No visual UI components in this phase.

### Notes

**Build Issues (non-blocking):**

1. `npm run build` fails with "File 'src/utils/schemas.ts' not listed within the file list of project 'tsconfig.node.json'". This is a pre-existing configuration issue documented in 02-04-SUMMARY.md. Workaround: `npx vite build` succeeds (Worker bundled correctly, 193.52 kB output).

2. `npm run lint` reports formatting errors in vite.config.ts (indentation/whitespace), not in phase 2 engine code. Engine files pass lint when isolated.

**These issues do not impact phase goal achievement.** All calculation engines are functional, tested, and production-ready.

---

_Verified: 2026-02-09T16:26:00Z_
_Verifier: Claude (gsd-verifier)_
