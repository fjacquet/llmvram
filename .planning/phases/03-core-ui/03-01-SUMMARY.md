---
phase: 03-core-ui
plan: 01
subsystem: ui
tags: [tailwind, zustand, react, dark-mode, @headlessui/react]

# Dependency graph
requires:
  - phase: 02-inference-engine
    provides: useInferenceCalculation hook and calculation types for UI consumption
provides:
  - Tailwind CSS v4 with class-based dark mode (@custom-variant dark)
  - Zustand store (useUIStore) managing all UI state with persist middleware
  - useDarkMode hook for theme toggling
  - DarkModeToggle component with accessible sun/moon icons
  - FOUC prevention for dark mode
  - @headlessui/react installed for accessible combobox selectors
affects: [03-02, 03-03, 03-04, 04-advanced-features, 05-multi-gpu]

# Tech tracking
tech-stack:
  added: [zustand, zustand/middleware, @headlessui/react]
  patterns: [Zustand persist with partialize, single DOM manipulation point in useDarkMode, FOUC prevention inline script]

key-files:
  created:
    - src/index.css
    - src/store/uiStore.ts
    - src/hooks/useDarkMode.ts
    - src/components/common/DarkModeToggle.tsx
  modified:
    - src/main.tsx
    - index.html
    - tsconfig.node.json
    - .gitignore

key-decisions:
  - "Persist only preferences (not selections) to avoid stale large objects"
  - "FOUC prevention uses inline script before stylesheets"
  - "useDarkMode hook is single source of truth for dark class manipulation"
  - "SVGs use aria-hidden since button has aria-label"

patterns-established:
  - "Zustand store with persist middleware and partialize for selective persistence"
  - "Dark mode synced from store to DOM via useEffect in hook"
  - "FOUC prevention inline script reads same localStorage key as persist middleware"

# Metrics
duration: 3min
completed: 2026-02-09
---

# Phase 3 Plan 1: UI Foundation & Dark Mode Summary

**Tailwind CSS v4 dark mode with Zustand persist store, FOUC prevention, and accessible theme toggle**

## Performance

- **Duration:** 3 minutes
- **Started:** 2026-02-09T15:51:31Z
- **Completed:** 2026-02-09T15:54:56Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Tailwind CSS v4 foundation with @custom-variant dark for class-based dark mode
- Zustand store managing all UI state (model, GPU, quantization, sequence length, batch size, KV quantization, dark mode)
- Dark mode persists across page refreshes with no flash of unstyled content
- DarkModeToggle component with sun/moon icons and accessible aria-labels
- @headlessui/react installed for Plan 02 combobox selectors

## Task Commits

Each task was committed atomically:

1. **Task 1: CSS Foundation, Headless UI Install, FOUC Prevention** - `29fbaae` (feat)
   - Created index.css with Tailwind v4 dark mode variant
   - Added FOUC prevention inline script to index.html
   - Imported CSS in main.tsx
   - Installed @headlessui/react
   - Fixed tsconfig.node.json to include schemas for script imports

2. **Task 2: Zustand UI Store, Dark Mode Hook, Toggle Component** - `2676bd1` (feat)
   - Created uiStore with persist middleware and partialize
   - Added useDarkMode hook to sync dark mode state to document.documentElement
   - Created DarkModeToggle component with accessible button and SVG icons
   - Updated .gitignore to exclude TypeScript build artifacts
   - Fixed FOUC script lint issues

## Files Created/Modified

**Created:**
- `src/index.css` - Tailwind v4 CSS import with @custom-variant dark for class-based dark mode
- `src/store/uiStore.ts` - Zustand store with persist middleware managing all UI selections and preferences
- `src/hooks/useDarkMode.ts` - Hook to sync isDarkMode from store to document.documentElement.classList
- `src/components/common/DarkModeToggle.tsx` - Accessible toggle button with sun/moon SVG icons

**Modified:**
- `src/main.tsx` - Added CSS import
- `index.html` - Added FOUC prevention inline script reading localStorage before first paint
- `tsconfig.node.json` - Added src/utils/schemas.ts to include array for script imports
- `.gitignore` - Excluded TypeScript build artifacts (scripts/**/*.js, src/**/*.js, *.config.js, etc.)

## Decisions Made

**Zustand persist strategy:**
- Used `partialize` to persist only preferences (quantization, sequenceLength, batchSize, kvQuantization, isDarkMode)
- Excluded selectedModel and selectedGPU from persistence to avoid stale large objects in localStorage
- Storage key `llmvram-ui-preferences` matches FOUC script key for consistency

**Dark mode architecture:**
- useDarkMode hook is the ONLY place that manipulates document.documentElement.classList
- Components call toggleDarkMode action, never touch classList directly
- FOUC prevention script in index.html reads same localStorage key before first paint

**Accessibility:**
- SVG icons use aria-hidden="true" since button already has descriptive aria-label
- Button has explicit type="button" to prevent form submission

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed tsconfig.node.json to include schemas**
- **Found during:** Task 1 (build verification)
- **Issue:** Scripts importing from src/utils/schemas.ts failed TypeScript check because tsconfig.node.json didn't include that file
- **Fix:** Added `src/utils/schemas.ts` to include array in tsconfig.node.json
- **Files modified:** tsconfig.node.json
- **Verification:** `npm run build` succeeded
- **Committed in:** 29fbaae (Task 1 commit)

**2. [Rule 3 - Blocking] Updated .gitignore to exclude build artifacts**
- **Found during:** Task 2 (lint verification)
- **Issue:** Biome linting TypeScript-generated .js and .d.ts files from scripts/ and configs, causing hundreds of lint errors
- **Fix:** Added patterns to .gitignore: scripts/**/*.js, scripts/**/*.d.ts, src/**/*.js, src/**/*.d.ts, *.config.js, *.config.d.ts
- **Files modified:** .gitignore
- **Verification:** `npm run lint` passed
- **Committed in:** 2676bd1 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking issues)
**Impact on plan:** Both auto-fixes necessary to unblock build and lint. No scope creep.

## Issues Encountered

None - plan executed smoothly after blocking issues resolved.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 02 (Model & GPU Selectors):**
- ✅ Zustand store provides typed actions for setSelectedModel, setSelectedGPU
- ✅ @headlessui/react installed for Combobox components
- ✅ Dark mode CSS classes work (Plan 02 components can use dark: utilities)
- ✅ Store persists quantization/sequence/batch preferences for pre-filling inputs

**No blockers.**

## Self-Check

**Files created:**
✓ src/index.css
✓ src/store/uiStore.ts
✓ src/hooks/useDarkMode.ts
✓ src/components/common/DarkModeToggle.tsx

**Files modified:**
✓ src/main.tsx
✓ index.html
✓ tsconfig.node.json
✓ .gitignore

**Commits:**
✓ 29fbaae (Task 1)
✓ 2676bd1 (Task 2)

**Result:** PASSED - All files and commits verified

---
*Phase: 03-core-ui*
*Completed: 2026-02-09*
