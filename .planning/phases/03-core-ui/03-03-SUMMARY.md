---
phase: 03-core-ui
plan: 03
subsystem: ui
tags: [react, recharts, tailwind, accessibility, visualization]

# Dependency graph
requires:
  - phase: 03-01
    provides: UI foundation with Tailwind v4 dark mode and component structure
provides:
  - VRAMBreakdownChart: Recharts donut chart showing 4 VRAM components with tooltip and legend
  - MemoryBreakdownTable: Accessible table alternative with GB values and percentages
  - FitIndicator: Color-coded status (green/yellow/red) with progress bar and percentage
  - Recommendations: Context-aware suggestions when model exceeds GPU VRAM
affects: [03-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure display components receiving all data as props (no Zustand access)"
    - "Semantic HTML elements (output, aside) instead of divs with ARIA roles"
    - "Recharts PieLabelRenderProps type for custom label renderers"
    - "Decimal.js precision maintained through toNumber() conversion at display boundary"

key-files:
  created:
    - src/components/outputs/VRAMBreakdownChart.tsx
    - src/components/outputs/MemoryBreakdownTable.tsx
    - src/components/outputs/FitIndicator.tsx
    - src/components/outputs/Recommendations.tsx
  modified: []

key-decisions:
  - "Use Recharts PieLabelRenderProps type with proper guards for optional properties"
  - "Use semantic <output> element for FitIndicator (calculation result display)"
  - "Use semantic <aside> element for Recommendations (complementary content)"
  - "Remove unused model parameter from Recommendations (not needed for current logic)"
  - "Calculate recommendation savings based on VRAM component ratios (50% for quantization, 50% for context)"

patterns-established:
  - "Output components are pure display - receive breakdown/config as props, no store access"
  - "Dark mode uses currentColor for SVG fills and Tailwind dark: variants for backgrounds"
  - "Accessibility: sr-only text for charts, semantic elements, aria-live for status updates"
  - "Color-coded status tiers: green (<=80%), yellow (81-95%), red (>95%) VRAM utilization"

# Metrics
duration: 6min
completed: 2026-02-09
---

# Phase 03 Plan 03: Output Components Summary

**VRAM breakdown donut chart, memory table, color-coded fit indicator, and actionable recommendations using Recharts and semantic HTML**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-09T15:57:30Z
- **Completed:** 2026-02-09T16:03:30Z
- **Tasks:** 2
- **Files modified:** 4 created

## Accomplishments

- Recharts donut chart with 4 VRAM components (weights, KV cache, activations, overhead) including tooltip, legend, and center total label
- Accessible memory breakdown table with colored dots, GB values, and percentages calculated using Decimal precision
- Color-coded fit indicator (green/yellow/red) with progress bar showing VRAM utilization percentage
- Context-aware recommendations generating up to 5 actionable suggestions (lower quantization, reduce context, KV cache quantization, multi-GPU, GPU upgrade) with estimated GB savings

## Task Commits

Each task was committed atomically:

1. **Task 1: VRAM Breakdown Chart and Memory Table** - `db9e963` (feat)
2. **Task 2: Fit Indicator and Recommendations** - `b4d3bc4` (feat)

**Bug fixes:** `66f5608` (fix - Recharts type annotations)

## Files Created/Modified

- `src/components/outputs/VRAMBreakdownChart.tsx` - Recharts donut chart with 4 VRAM components, custom labels, tooltip, legend, and screen reader text
- `src/components/outputs/MemoryBreakdownTable.tsx` - Accessible table showing VRAM components with GB values and percentages
- `src/components/outputs/FitIndicator.tsx` - Color-coded status card (green/yellow/red) with progress bar and percentage using semantic <output> element
- `src/components/outputs/Recommendations.tsx` - Generates context-aware suggestions when model doesn't fit, estimates GB savings for each recommendation

## Decisions Made

**Use semantic HTML elements instead of divs with ARIA roles:**
- FitIndicator uses `<output>` (designed for calculation results) instead of `<div role="status">`
- Recommendations uses `<aside>` (semantic complementary content) instead of `<div role="complementary">`
- Rationale: Better accessibility and simpler markup per Biome linter suggestions

**Remove unused model parameter from Recommendations:**
- Initially included Model type in props but never used it
- Current recommendation logic only needs GPU, breakdown, quantization format, and sequence length
- Removed to satisfy noUnusedFunctionParameters lint rule

**Calculate recommendation savings using component ratios:**
- Quantization savings: ~50% of model weights (fp16→4-bit), ~35% (int8→int4), ~15% (GPTQ→GGUF Q4_0)
- Context reduction savings: ~50% of KV cache (halving context)
- KV quantization savings: ~75% of KV cache (fp16→int4)
- Rationale: Provides users with concrete GB estimates rather than abstract percentages

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] TypeScript type errors in VRAMBreakdownChart**
- **Found during:** Task 1 verification (npm run build)
- **Issue:** Recharts PieLabelRenderProps type mismatch - renderLabel function used inline object type instead of proper Recharts type, causing strict TypeScript errors about optional properties (midAngle?: number incompatible with required number)
- **Fix:** Imported PieLabelRenderProps type from recharts, used it for renderLabel parameter, added typeof guards for all optional properties (value, cx, cy, midAngle, innerRadius, outerRadius), updated Tooltip formatter to handle number | undefined properly
- **Files modified:** src/components/outputs/VRAMBreakdownChart.tsx
- **Verification:** npm run build succeeded, TypeScript compilation passed
- **Committed in:** `66f5608` (separate fix commit after Task 1)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix necessary for production build. Recharts type system requires proper PieLabelRenderProps type - inline object type was insufficient for strict TypeScript. No scope creep.

## Issues Encountered

**Biome linter suggested semantic HTML elements:**
- Initial implementation used `<div role="status">` for FitIndicator and `<div role="complementary">` for Recommendations
- Biome lint rules (useSemanticElements) suggested using `<output>` and `<aside>` respectively
- Resolution: Adopted semantic elements - better accessibility and simpler markup. `<output>` is perfect for FitIndicator (shows calculation result), `<aside>` is ideal for Recommendations (complementary content).

**SVG accessibility warnings:**
- Biome noSvgWithoutTitle rule flagged inline SVG icons without <title> elements
- Resolution: Added descriptive <title> elements to all three status icons (Success, Warning, Error)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 03-04 (Results Panel Assembly):**
- All 4 output components complete and tested (typecheck + lint + build passing)
- Components are pure display - receive data as props, ready to be integrated into ResultsPanel
- VRAMBreakdownChart and MemoryBreakdownTable visualize InferenceVRAMBreakdown
- FitIndicator shows utilization status given totalVRAM and availableVRAM
- Recommendations generates suggestions given GPU, breakdown, quantization, and sequence length

**Integration requirements for 03-04:**
- ResultsPanel will need to import these 4 components
- Pass InferenceVRAMBreakdown from useInferenceCalculation hook
- Pass selected GPU, current quantization format, and sequence length for Recommendations
- Conditionally render Recommendations only when model doesn't fit (deficit > 0)

---
*Phase: 03-core-ui*
*Completed: 2026-02-09*

## Self-Check: PASSED

### Files Created:
✓ FOUND: src/components/outputs/VRAMBreakdownChart.tsx
✓ FOUND: src/components/outputs/MemoryBreakdownTable.tsx
✓ FOUND: src/components/outputs/FitIndicator.tsx
✓ FOUND: src/components/outputs/Recommendations.tsx

### Commits:
✓ FOUND: db9e963 (Task 1 - VRAM breakdown chart and memory table)
✓ FOUND: b4d3bc4 (Task 2 - Fit indicator and recommendations)
✓ FOUND: 66f5608 (Bug fix - Recharts type annotations)

All files created and commits verified successfully.
