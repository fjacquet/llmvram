# Plan 04-03 Summary: Multi-GPU UI + Offloading + Quantization Expansion

## Result
Complete multi-GPU, CPU/RAM/NVMe offloading, and expanded quantization feature delivered end-to-end.

## Tasks Completed
| # | Task | Commit | Duration |
|---|------|--------|----------|
| 1 | Offloading engine and tests | 91fc508 | ~2 min |
| 2 | Store, Worker, Hook extension for offloading | b5823f1 | ~2 min |
| 3 | Multi-GPU + offloading input components | efa070f | ~3 min |
| 4 | Multi-GPU output chart + ResultsPanel + Recommendations | 472e9e1 | ~3 min |
| + | Quantization expansion (13 → 22 formats) | 9346df2 | ~2 min |
| + | Fix render loop (useMemo for offloadingConfig) | 64b9cf9 | ~1 min |

## Key Files

### Created
- `src/engines/offloading.ts` — calculateOffloadedVRAM (percentage/layers, CPU/NVMe, KV cache)
- `src/engines/offloading.test.ts` — 12 tests for offloading engine
- `src/components/inputs/GPUCountSelector.tsx` — 1-8 GPU slider
- `src/components/inputs/ShardingStrategySelector.tsx` — TP/PP radios with interconnect badges
- `src/components/inputs/OffloadingPanel.tsx` — Enable toggle, target, mode, percentage/layers, KV cache
- `src/components/outputs/MultiGPUBreakdownChart.tsx` — Recharts stacked bar chart

### Modified
- `src/engines/types.ts` — Added OffloadTarget, OffloadMode, OffloadingConfig, OffloadedVRAMBreakdown; expanded QuantizationFormat (13 → 22)
- `src/engines/constants.ts` — Added BYTES_PER_PARAMETER for 9 new formats (NVFP4, NVFP6, Q2_K-Q5_K_S)
- `src/store/uiStore.ts` — 6 offloading state fields with persistence
- `src/workers/calculation.worker.ts` — Offloading computation (before multi-GPU), serialization
- `src/hooks/useInferenceCalculation.ts` — OffloadingConfig param, OffloadedVRAMBreakdown in result
- `src/components/layout/InputPanel.tsx` — Hardware Configuration + Offloading sections
- `src/components/layout/ResultsPanel.tsx` — Offloading summary, multi-GPU chart, useMemo fix
- `src/components/outputs/Recommendations.tsx` — Real multi-GPU estimates, offloading suggestion
- `src/components/inputs/QuantizationPicker.tsx` — 22 formats in 5 groups, "Inference Quantization" label
- `src/engines/quantization.test.ts` — Updated for all 22 formats

## Decisions
- Offloading applies BEFORE multi-GPU distribution (reduces base, then splits across GPUs)
- useMemo required for offloadingConfig to prevent render loop (object reference in useEffect deps)
- Quantization expanded from 13 to 22 formats: added NVFP4, NVFP6, Q2_K, Q3_K_S/M/L, Q4_K_S, Q5_0, Q5_K_S
- QuantizationPicker refactored with reusable QuantizationGroup component to reduce repetition

## Deviations
- Scope expanded beyond original plan to include CPU/RAM/NVMe offloading (user request)
- Quantization format expansion added (user request after comparing with apxml.com)
- Render loop bug discovered and fixed during human verification

## Self-Check: PASSED
- [x] TypeScript: no errors
- [x] Biome: clean
- [x] Tests: 132 passing (9 test files)
- [x] Human verification: approved
