---
phase: 03-core-ui
plan: 02
subsystem: ui
tags: [react, headlessui, heroicons, combobox, zustand, tailwind]

# Dependency graph
requires:
  - phase: 03-01
    provides: Zustand store with model/GPU selection actions, dark mode, Tailwind v4 setup
  - phase: 01-03
    provides: Model database JSON with 30+ curated models
  - phase: 01-02
    provides: GPU database JSON with 18+ curated GPUs
provides:
  - Model selector with search, custom input form, and Zustand integration
  - GPU selector with manufacturer grouping, search, custom input form
  - Quantization picker with 13 formats grouped by type
  - KV quantization picker with 4 precision options
  - Sequence length slider with log scale and presets
  - Batch size slider with presets
affects: [03-03, 03-04]

# Tech tracking
tech-stack:
  added: [@heroicons/react]
  patterns:
    - Headless UI Combobox for searchable dropdowns (accessible alternative to native select)
    - Custom form state in local useState (not Zustand) until submission
    - Log-scale sliders for exponential ranges (sequence length 512-131K)
    - Preset buttons alongside sliders for common values

key-files:
  created:
    - src/components/inputs/ModelSelector.tsx
    - src/components/inputs/GPUSelector.tsx
    - src/components/inputs/QuantizationPicker.tsx
    - src/components/inputs/KVQuantizationPicker.tsx
    - src/components/inputs/SequenceLengthInput.tsx
    - src/components/inputs/BatchSizeInput.tsx
  modified:
    - package.json (added @heroicons/react)
    - src/components/outputs/FitIndicator.tsx (fixed JSX namespace error)
    - src/components/outputs/VRAMBreakdownChart.tsx (fixed Recharts type errors)

key-decisions:
  - "Use Headless UI Combobox instead of native select for keyboard navigation and screen reader support"
  - "Keep custom form state local with useState until submission to avoid polluting Zustand store with temporary values"
  - "Implement log-scale slider for sequence length to provide even spacing across exponential range (512 to 131K)"
  - "Group quantization formats by type (float/integer/compressed/GGUF) for easier scanning"
  - "Group GPUs by manufacturer (NVIDIA/AMD/Apple/Intel) with visual dividers"

patterns-established:
  - "Input components: label + control + preset buttons (for sliders) pattern"
  - "Selector components: Combobox + search + curated list + custom form pattern"
  - "Custom forms appear inline below selector when 'Custom...' option selected"

# Metrics
duration: 5min
completed: 2026-02-09
---

# Phase 3 Plan 2: Input Components Summary

**Six input components with searchable model/GPU selectors, custom input forms, and parameter controls feeding Zustand store**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-09T15:57:22Z
- **Completed:** 2026-02-09T16:03:18Z
- **Tasks:** 2
- **Files modified:** 9 (6 created, 3 fixed)

## Accomplishments
- Model selector with search filtering across 30+ curated models and custom model form
- GPU selector with manufacturer grouping, search across 18+ GPUs, and custom GPU form
- Quantization picker covering all 13 formats with human-readable labels grouped by type
- KV cache precision picker with 4 options (fp16/fp8/int8/int4)
- Sequence length slider with logarithmic scale (512-131K) and 6 preset buttons
- Batch size slider with linear scale (1-64) and 6 preset buttons

## Task Commits

Each task was committed atomically:

1. **Task 1: Model and GPU Selector Components** - `24c99d0` (feat)
2. **Task 2: Quantization Pickers and Parameter Sliders** - `36d9136` (feat)

## Files Created/Modified

### Created
- `src/components/inputs/ModelSelector.tsx` - Headless UI Combobox with search, curated models list, custom model form (name, params, optional architecture)
- `src/components/inputs/GPUSelector.tsx` - Headless UI Combobox with manufacturer grouping (NVIDIA/AMD/Apple/Intel), search, custom GPU form (name, VRAM, optional bandwidth/FLOPS)
- `src/components/inputs/QuantizationPicker.tsx` - Listbox with 13 formats grouped (float: fp32/fp16/bf16, integer: int8/int4/nf4, compressed: GPTQ/AWQ, GGUF: 5 variants)
- `src/components/inputs/KVQuantizationPicker.tsx` - Listbox with 4 KV cache precision options
- `src/components/inputs/SequenceLengthInput.tsx` - Range slider with log2 scale (min=9, max=17), value display, and preset buttons (512/2K/4K/8K/32K/128K)
- `src/components/inputs/BatchSizeInput.tsx` - Range slider with linear scale (1-64), value display, and preset buttons (1/4/8/16/32/64)

### Modified (auto-fixes)
- `package.json` / `package-lock.json` - Added @heroicons/react dependency
- `src/components/inputs/ModelSelector.tsx` / `GPUSelector.tsx` - Fixed import path from `@types/model` to `@/types/model` (avoided TypeScript namespace conflict)
- `src/components/outputs/FitIndicator.tsx` - Added ReactElement import to fix JSX namespace error
- `src/components/outputs/VRAMBreakdownChart.tsx` - Fixed Recharts type errors by accepting optional props in renderLabel and Tooltip formatter

## Decisions Made

**Use Headless UI Combobox over native select:**
- Provides keyboard navigation (arrow keys, enter, escape)
- Screen reader announcements for accessibility
- Built-in search filtering support
- Consistent cross-browser styling via Tailwind

**Local useState for custom form state:**
- Custom model/GPU forms use local state (name, params, etc.)
- Only call Zustand actions (setSelectedModel/setSelectedGPU) on form submission
- Avoids polluting global store with temporary form state
- Follows principle: store holds committed selections, not draft input

**Log-scale slider for sequence length:**
- Linear slider would bunch low values (512, 1024, 2048) at left edge
- Log2 scale provides even spacing: slider midpoint = 8K tokens (geometric mean of 512 and 131K)
- Formula: `displayValue = 2^sliderValue`, `sliderValue = log2(sequenceLength)`
- Step 0.1 allows intermediate values between powers of 2

**Visual grouping patterns:**
- Quantization formats grouped by type (float/integer/compressed/GGUF) with non-selectable dividers
- GPUs grouped by manufacturer (NVIDIA/AMD/Apple/Intel) with header labels
- Improves scannability: users can jump to relevant section instead of scanning full list

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing @heroicons/react dependency**
- **Found during:** Task 1 (ModelSelector implementation)
- **Issue:** Plan specified using icons from @heroicons/react/20/solid but package wasn't in package.json. Import failed during build.
- **Fix:** Ran `npm install @heroicons/react` to add missing dependency
- **Files modified:** package.json, package-lock.json
- **Verification:** Build passes, icons render correctly
- **Committed in:** Separate from task commits (applied before verification)

**2. [Rule 1 - Bug] Fixed @types path alias conflict**
- **Found during:** Build verification after Task 2
- **Issue:** TypeScript import `from '@types/model'` conflicted with built-in `@types` namespace (used for DefinitelyTyped packages like @types/node). Error: "Cannot import type declaration files. Consider importing 'model' instead of '@types/model'."
- **Fix:** Changed imports from `@types/model` → `@/types/model` and `@types/gpu` → `@/types/gpu` in ModelSelector, GPUSelector, and Recommendations components
- **Files modified:** src/components/inputs/ModelSelector.tsx, src/components/inputs/GPUSelector.tsx, src/components/outputs/Recommendations.tsx
- **Verification:** TypeScript build passes without errors
- **Committed in:** Applied after task commits, before SUMMARY creation

**3. [Rule 1 - Bug] Fixed JSX namespace error in FitIndicator**
- **Found during:** Build verification after Task 2
- **Issue:** `icon: JSX.Element` in StatusTier interface caused "Cannot find namespace 'JSX'" error. JSX namespace not automatically available in TypeScript strict mode without React import.
- **Fix:** Added `import type { ReactElement } from 'react'` and changed `icon: JSX.Element` → `icon: ReactElement`
- **Files modified:** src/components/outputs/FitIndicator.tsx
- **Verification:** Build passes, no JSX namespace errors
- **Committed in:** Applied after task commits, before SUMMARY creation

**4. [Rule 1 - Bug] Fixed Recharts type errors in VRAMBreakdownChart**
- **Found during:** Build verification after Task 2
- **Issue:** Two type errors with Recharts prop types:
  - `renderLabel` function defined required props (`midAngle: number`) but Recharts PieLabelRenderProps has optional props (`midAngle: number | undefined`)
  - Tooltip formatter defined `(value: number)` but Recharts passes `value: number | undefined`
- **Fix:**
  - Made all renderLabel entry props optional and added guard clause to return null if any are undefined
  - Changed Tooltip formatter signature to accept `value: number | undefined` with conditional return
- **Files modified:** src/components/outputs/VRAMBreakdownChart.tsx
- **Verification:** Build passes, pie chart labels and tooltip render correctly
- **Committed in:** Applied after task commits, before SUMMARY creation

---

**Total deviations:** 4 auto-fixed (1 blocking dependency, 3 type errors)
**Impact on plan:** All auto-fixes were necessary for build correctness. Dependency was missing from plan specs. Type errors were in pre-existing output components discovered during build verification. No scope creep — all fixes enable plan deliverables to function.

## Issues Encountered

**@types path alias collision:**
- TypeScript reserves `@types` namespace for DefinitelyTyped packages (@types/react, @types/node, etc.)
- Using `@types/*` as a custom path alias caused import resolution failures
- Resolution: Use full `@/types/*` path instead, which resolves to `src/types/*` without conflict
- Alternative (not taken): Rename alias from `@types` to `@schemas` or `@domain` in tsconfig and vite config
- Chosen approach requires minimal changes (only import statements), doesn't touch build config

**Recharts type strictness:**
- Recharts label and formatter functions receive optional props (`value?: number`, `midAngle?: number`)
- Stricter than documented examples, which assume required props
- Resolution: Make function signatures match library types, add runtime guards for undefined
- Pattern for future: Always check Recharts prop types, don't trust documentation examples

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 3 Plan 3 (Calculation Parameter Inputs):**
- All six input components functional and connected to Zustand store
- Model/GPU selectors load from JSON databases and accept custom specs
- Quantization and parameter controls map directly to calculation engine types
- Dark mode styling applied consistently across all inputs

**Ready for Phase 3 Plan 4 (Input Panel Assembly):**
- Input components are self-contained and can be assembled into layout panels
- Each component has proper labels and ARIA attributes for accessibility
- Tailwind classes support responsive layout (mobile/desktop)

**No blockers.** All input components verified working via build. UI interactions will be validated when assembled into full page in Plan 4.

---
*Phase: 03-core-ui*
*Completed: 2026-02-09*
