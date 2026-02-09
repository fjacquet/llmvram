# Phase 3: Core UI - Research

**Researched:** 2026-02-09
**Domain:** React 19 + Tailwind v4 UI implementation with real-time VRAM calculations
**Confidence:** HIGH

## Summary

Phase 3 builds the complete user interface for single-GPU VRAM calculations. The tech stack is already locked from Phase 1/2: React 19, Tailwind CSS v4 with @tailwindcss/vite, Zustand for state, and Recharts for visualization. All calculation engines and hooks are ready to consume.

The implementation centers on three component groups: **inputs** (model/GPU selectors, quantization/parameter controls), **outputs** (VRAM breakdown chart, fit indicator, recommendations), and **common** (dark mode toggle, responsive layout). Real-time updates flow through Zustand state, triggering the existing useInferenceCalculation hook which offloads work to a Web Worker.

Critical considerations: Tailwind v4 requires explicit @custom-variant configuration for class-based dark mode, Recharts needs careful TypeScript and ResponsiveContainer setup, and accessibility for data visualization requires ARIA roles plus keyboard navigation. Sonner provides toast notifications for errors and confirmations.

**Primary recommendation:** Use Headless UI for accessible select/combobox components (model/GPU pickers), implement dark mode with class strategy + localStorage persistence to avoid FOUC, and structure Recharts with explicit ResponsiveContainer wrapping to prevent TypeScript errors and ensure mobile responsiveness.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 19.2.0 | UI framework | Latest stable, locked in Phase 1 |
| TypeScript | 5.9.3 | Type safety | Strict mode enabled, noUncheckedIndexedAccess |
| Vite | 7.2.4 | Build tool | Fast dev server, ES modules, worker support |
| Tailwind CSS | 4.1.18 | Styling | CSS-first config, 3-8x faster builds than v3 |
| @tailwindcss/vite | 4.1.18 | Tailwind integration | Required for v4 with Vite (no PostCSS needed) |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Zustand | 5.0.9 | State management | UI state (selections, dark mode, form values) |
| Recharts | 3.6.0 | Data visualization | VRAM breakdown charts (donut/stacked bar) |
| sonner | 2.0.7 | Toast notifications | Error messages, success feedback |
| Headless UI | 2.x (to install) | Accessible components | Select/combobox for model/GPU pickers |
| Decimal.js | 10.4.3 | Precise math | Already used by engines, format for display |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Recharts | Victory, Nivo, D3 | Recharts simplest for basic charts, good TypeScript support |
| Zustand | Redux Toolkit, Jotai | Zustand lightest (3kb), minimal boilerplate |
| Headless UI | Radix UI, Ariakit | All viable, Headless UI official Tailwind recommendation |
| sonner | react-hot-toast | sonner has better stacking, keyboard shortcuts |

**Installation:**
```bash
npm install @headlessui/react
```

All other dependencies already installed from Phase 1/2.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── components/
│   ├── inputs/              # User input controls
│   │   ├── ModelSelector.tsx       # Curated + custom model input
│   │   ├── GPUSelector.tsx         # Curated + custom GPU input
│   │   ├── QuantizationPicker.tsx  # 13 quantization formats dropdown
│   │   ├── SequenceLengthInput.tsx # Slider: 512-131072
│   │   ├── BatchSizeInput.tsx      # Slider: 1-64
│   │   └── KVQuantizationPicker.tsx # fp16/fp8/int8/int4 dropdown
│   ├── outputs/             # Calculation results display
│   │   ├── VRAMBreakdownChart.tsx  # Recharts donut/stacked bar
│   │   ├── FitIndicator.tsx        # Green/yellow/red status + percentage
│   │   ├── MemoryBreakdownTable.tsx # Table view of VRAM components
│   │   └── Recommendations.tsx     # Actionable suggestions when doesn't fit
│   ├── common/              # Shared UI elements
│   │   ├── DarkModeToggle.tsx      # Sun/moon icon with localStorage
│   │   ├── Card.tsx                # Reusable container with dark mode
│   │   ├── Slider.tsx              # Range input with labels
│   │   └── Select.tsx              # Headless UI wrapper with Tailwind
│   └── layout/              # Page structure
│       ├── Header.tsx              # Title + dark mode toggle
│       ├── InputPanel.tsx          # Left/top panel with all inputs
│       ├── ResultsPanel.tsx        # Right/bottom panel with outputs
│       └── Layout.tsx              # Responsive grid container
├── store/
│   └── uiStore.ts           # Zustand store for all UI state
└── App.tsx                  # Orchestrator with Toaster
```

### Pattern 1: Zustand UI Store
**What:** Single store for all form inputs, calculated results, and UI preferences.
**When to use:** For app-wide state that multiple components read/write.
**Example:**
```typescript
// src/store/uiStore.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Model, GPU } from '@types'
import type { QuantizationFormat, KVCachePrecision } from '@engines/types'

interface UIState {
  // Selections
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  kvQuantization: KVCachePrecision

  // UI preferences (persisted)
  isDarkMode: boolean
  chartType: 'donut' | 'stacked'

  // Actions
  setSelectedModel: (model: Model | null) => void
  setSelectedGPU: (gpu: GPU | null) => void
  setQuantization: (format: QuantizationFormat) => void
  setSequenceLength: (length: number) => void
  setBatchSize: (size: number) => void
  setKVQuantization: (precision: KVCachePrecision) => void
  toggleDarkMode: () => void
  setChartType: (type: 'donut' | 'stacked') => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Initial state
      selectedModel: null,
      selectedGPU: null,
      quantization: 'gptq',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
      isDarkMode: false,
      chartType: 'donut',

      // Actions
      setSelectedModel: (model) => set({ selectedModel: model }),
      setSelectedGPU: (gpu) => set({ selectedGPU: gpu }),
      setQuantization: (format) => set({ quantization: format }),
      setSequenceLength: (length) => set({ sequenceLength: length }),
      setBatchSize: (size) => set({ batchSize: size }),
      setKVQuantization: (precision) => set({ kvQuantization: precision }),
      toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
      setChartType: (type) => set({ chartType: type }),
    }),
    {
      name: 'llmvram-ui-preferences',
      partialize: (state) => ({
        isDarkMode: state.isDarkMode,
        chartType: state.chartType,
        quantization: state.quantization,
        sequenceLength: state.sequenceLength,
        batchSize: state.batchSize,
        kvQuantization: state.kvQuantization,
      }),
    }
  )
)
```

### Pattern 2: Dark Mode with Tailwind v4
**What:** Class-based dark mode with localStorage persistence and FOUC prevention.
**When to use:** Always, for Phase 3 requirement VIZ-07.
**Example:**
```typescript
// src/hooks/useDarkMode.ts
import { useEffect } from 'react'
import { useUIStore } from '@store/uiStore'

export function useDarkMode() {
  const { isDarkMode, toggleDarkMode } = useUIStore()

  useEffect(() => {
    // Apply dark class to document root
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  return { isDarkMode, toggleDarkMode }
}
```

```css
/* src/index.css */
@import "tailwindcss";

/* Override default media query strategy with class-based */
@custom-variant dark (&:where(.dark, .dark *));

/* Optional: Define theme variables */
@theme {
  --color-background: #ffffff;
  --color-background-dark: #0f172a;
}
```

```html
<!-- index.html - inline script to prevent FOUC -->
<head>
  <script>
    // Run before styles load to prevent flash
    (function() {
      const stored = localStorage.getItem('llmvram-ui-preferences')
      if (stored) {
        try {
          const { state } = JSON.parse(stored)
          if (state.isDarkMode) {
            document.documentElement.classList.add('dark')
          }
        } catch {}
      }
    })()
  </script>
</head>
```

### Pattern 3: Real-time Calculation with Hook
**What:** Connect Zustand store to useInferenceCalculation hook for automatic updates.
**When to use:** In ResultsPanel component to trigger calculations on input change.
**Example:**
```typescript
// src/components/layout/ResultsPanel.tsx
import { useUIStore } from '@store/uiStore'
import { useInferenceCalculation } from '@hooks/useInferenceCalculation'

export function ResultsPanel() {
  const {
    selectedModel,
    selectedGPU,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization,
  } = useUIStore()

  // Hook automatically recalculates when inputs change
  const { result, loading, error } = useInferenceCalculation(
    selectedModel,
    selectedGPU,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization
  )

  if (!selectedModel || !selectedGPU) {
    return <NoSelectionPrompt />
  }

  if (loading) {
    return <LoadingSpinner />
  }

  if (error) {
    return <ErrorDisplay error={error} />
  }

  if (!result) {
    return null
  }

  return (
    <div className="space-y-6">
      <FitIndicator
        totalVRAM={result.vram.total}
        availableVRAM={selectedGPU.vram_gb}
      />
      <VRAMBreakdownChart breakdown={result.vram} />
      <MemoryBreakdownTable breakdown={result.vram} />
      {result.vram.total.greaterThan(selectedGPU.vram_gb) && (
        <Recommendations
          currentConfig={{ model: selectedModel, gpu: selectedGPU }}
          breakdown={result.vram}
        />
      )}
    </div>
  )
}
```

### Pattern 4: Recharts Responsive Charts
**What:** Donut chart wrapped in ResponsiveContainer with explicit aspect ratio.
**When to use:** For VIZ-01 memory breakdown visualization.
**Example:**
```typescript
// src/components/outputs/VRAMBreakdownChart.tsx
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'
import type { InferenceVRAMBreakdown } from '@engines/types'

interface Props {
  breakdown: InferenceVRAMBreakdown
}

const COLORS = {
  modelWeights: '#3b82f6',    // blue
  kvCache: '#10b981',         // green
  activations: '#f59e0b',     // amber
  frameworkOverhead: '#6366f1' // indigo
}

export function VRAMBreakdownChart({ breakdown }: Props) {
  const data = [
    { name: 'Model Weights', value: breakdown.modelWeights.toNumber(), color: COLORS.modelWeights },
    { name: 'KV Cache', value: breakdown.kvCache.toNumber(), color: COLORS.kvCache },
    { name: 'Activations', value: breakdown.activations.toNumber(), color: COLORS.activations },
    { name: 'Framework Overhead', value: breakdown.frameworkOverhead.toNumber(), color: COLORS.frameworkOverhead },
  ]

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={100}
            innerRadius={60}
            label={({ name, value }) => `${name}: ${value.toFixed(2)} GB`}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip formatter={(value: number) => `${value.toFixed(2)} GB`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
```

### Pattern 5: Accessible Combobox with Headless UI
**What:** Model/GPU selector with search, custom input, and keyboard navigation.
**When to use:** For ModelSelector and GPUSelector components.
**Example:**
```typescript
// src/components/inputs/ModelSelector.tsx
import { Combobox, ComboboxInput, ComboboxOptions, ComboboxOption } from '@headlessui/react'
import { useState } from 'react'
import type { Model } from '@types'
import modelsData from '@data/models.json'

interface Props {
  selected: Model | null
  onChange: (model: Model | null) => void
}

export function ModelSelector({ selected, onChange }: Props) {
  const [query, setQuery] = useState('')

  const filteredModels = query === ''
    ? modelsData
    : modelsData.filter((model) =>
        model.name.toLowerCase().includes(query.toLowerCase())
      )

  return (
    <Combobox value={selected} onChange={onChange}>
      <ComboboxInput
        className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-4 py-2"
        displayValue={(model: Model | null) => model?.name ?? ''}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search models..."
      />
      <ComboboxOptions className="mt-1 max-h-60 overflow-auto rounded-lg bg-white dark:bg-gray-800 shadow-lg">
        {filteredModels.length === 0 && query !== '' ? (
          <div className="px-4 py-2 text-gray-500">No models found.</div>
        ) : (
          filteredModels.map((model) => (
            <ComboboxOption
              key={model.id}
              value={model}
              className="cursor-pointer px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              {model.name} ({model.num_parameters_billion}B params)
            </ComboboxOption>
          ))
        )}
        <ComboboxOption
          value={null}
          className="cursor-pointer px-4 py-2 border-t border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
        >
          + Add custom model
        </ComboboxOption>
      </ComboboxOptions>
    </Combobox>
  )
}
```

### Anti-Patterns to Avoid
- **Don't use Context API for UI state** — Zustand is already in the stack, simpler than Context + useReducer
- **Don't put calculation logic in components** — Engines are pure and tested, keep components dumb
- **Don't inline chart data transformations** — Extract to utility functions for testing
- **Don't use CSS-in-JS** — Tailwind v4 already configured, mixing paradigms adds complexity
- **Don't skip ResponsiveContainer** — Charts won't adapt to mobile without it

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Accessible dropdowns | Custom <select> with ARIA | Headless UI Combobox | Keyboard nav, screen readers, search already solved |
| Dark mode toggle | Manual localStorage + class management | Zustand persist middleware + useEffect | Handles storage, hydration, SSR edge cases |
| Toast notifications | Custom fixed-position div system | sonner | Stacking, auto-dismiss, keyboard shortcuts, accessibility |
| Form validation | Manual state + error messages | Zod (already in stack) | Runtime + compile-time validation, single source of truth |
| Chart tooltips | Custom hover state + positioning | Recharts Tooltip component | Responsive positioning, formatting, dark mode support |
| Number formatting | Manual toFixed() and locale logic | Intl.NumberFormat or Decimal.js .toFixed() | Handles locales, edge cases (NaN, Infinity) |
| Responsive breakpoints | Manual window resize listeners | Tailwind responsive prefixes (sm:, md:, lg:) | SSR-safe, no JS needed |

**Key insight:** UI components are the most bug-prone part of web apps due to accessibility, browser differences, and edge cases. The stack is deliberately chosen to avoid custom implementations. Every "I'll just build a simple X" has 20+ edge cases.

## Common Pitfalls

### Pitfall 1: Tailwind v4 Dark Mode Without @custom-variant
**What goes wrong:** Dark mode classes (dark:bg-gray-800) don't apply even when .dark class is on <html>. Styles work in light mode but dark mode silently fails.

**Why it happens:** Tailwind v4 changed from JavaScript config to CSS-first configuration. The default dark mode uses prefers-color-scheme media query. To enable class-based dark mode (required for toggle), you MUST explicitly declare @custom-variant in your CSS.

**How to avoid:**
```css
/* src/index.css - REQUIRED for class-based dark mode */
@import "tailwindcss";

@custom-variant dark (&:where(.dark, .dark *));
```

Without this line, document.documentElement.classList.add('dark') has no effect on Tailwind classes.

**Warning signs:**
- Dark mode toggle changes localStorage but UI doesn't update
- DevTools shows .dark class on <html> but styles don't apply
- prefers-color-scheme works but manual toggle doesn't

**Sources:**
- [How I Fixed Tailwind CSS v4 Dark Mode Not Working](https://medium.com/@balpetekserhat/how-i-fixed-tailwind-css-v4-dark-mode-not-working-in-a-vite-react-project-d7f0b3a31184)
- [Tailwind v4 @custom-variant discussion](https://github.com/tailwindlabs/tailwindcss/discussions/16974)

---

### Pitfall 2: FOUC (Flash of Unstyled Content) on Dark Mode Load
**What goes wrong:** Page loads in light mode for 50-200ms, then flickers to dark mode. Users with dark mode preference see white flash on every page load.

**Why it happens:** React renders, Zustand hydrates from localStorage, useEffect runs, THEN dark class is applied. CSS paint happens before JavaScript runs.

**How to avoid:**
```html
<!-- index.html - inline script in <head> BEFORE stylesheets -->
<head>
  <script>
    (function() {
      const stored = localStorage.getItem('llmvram-ui-preferences')
      if (stored) {
        try {
          const { state } = JSON.parse(stored)
          if (state.isDarkMode) {
            document.documentElement.classList.add('dark')
          }
        } catch {}
      }
    })()
  </script>
  <!-- Stylesheets load AFTER this script -->
</head>
```

This synchronous script runs before first paint, applying dark class before CSS evaluates.

**Warning signs:**
- White flash on page load even with dark mode enabled
- Users complain about "blinding white screen" when opening app
- Dark mode works but initial render is light

**Sources:**
- [Tailwind CSS Dark Mode Docs](https://tailwindcss.com/docs/dark-mode)

---

### Pitfall 3: Recharts ResponsiveContainer TypeScript Errors
**What goes wrong:** TypeScript error "Type 'ReactElement' is not assignable to type 'ReactNode'" when putting charts inside ResponsiveContainer. Code works at runtime but fails compilation.

**Why it happens:** Type mismatch between @types/react 18.2.74+ and Recharts 3.x types. ResponsiveContainer expects children to be React.ReactElement but TypeScript infers React.ReactNode.

**How to avoid:**
```typescript
// Explicit child component extraction
function ChartContent() {
  return (
    <PieChart>
      {/* chart content */}
    </PieChart>
  )
}

// Then use in ResponsiveContainer
<ResponsiveContainer width="100%" height="100%">
  <ChartContent />
</ResponsiveContainer>

// OR: Type assertion (less ideal)
<ResponsiveContainer width="100%" height="100%">
  {<PieChart>{/* content */}</PieChart> as React.ReactElement}
</ResponsiveContainer>
```

Alternatively, set skipLibCheck: true in tsconfig.json (not recommended for strict projects).

**Warning signs:**
- TypeScript error during build but app works in dev
- Error mentions "Type 'ReactElement' is not assignable"
- Chart renders correctly but CI/CD fails

**Sources:**
- [Recharts Issue #2385](https://github.com/recharts/recharts/issues/2385)

---

### Pitfall 4: Recharts Charts Not Responsive on Mobile
**What goes wrong:** Charts overflow container on mobile or have fixed size that doesn't adapt to screen width.

**Why it happens:** Recharts components require explicit width and height. Without ResponsiveContainer, charts default to 0x0 or use hardcoded pixel values.

**How to avoid:**
```typescript
// WRONG - fixed size
<PieChart width={400} height={400}>
  {/* chart */}
</PieChart>

// RIGHT - responsive with container
<div className="w-full h-80">  {/* Tailwind container */}
  <ResponsiveContainer width="100%" height="100%">
    <PieChart>
      {/* chart */}
    </PieChart>
  </ResponsiveContainer>
</div>
```

The outer div must have explicit height (Tailwind h-80, or height: 320px). ResponsiveContainer fills parent dimensions.

**Warning signs:**
- Chart invisible on some screen sizes
- Horizontal scroll on mobile
- Chart doesn't resize when sidebar opens/closes

**Sources:**
- [Recharts ResponsiveContainer Docs](https://recharts.github.io/en-US/examples/StackedBarChart/)

---

### Pitfall 5: Accessibility - Charts Not Keyboard Navigable
**What goes wrong:** Screen reader users can't access chart data. Keyboard users can't focus or interact with chart segments. Fails WCAG 2.1 AA.

**Why it happens:** Recharts renders SVG without ARIA roles or keyboard handlers. Default output is visual-only.

**How to avoid:**
```typescript
// Add ARIA roles and text alternative
<div role="img" aria-label="VRAM breakdown chart showing memory allocation">
  <ResponsiveContainer width="100%" height="100%">
    <PieChart>
      <Pie
        data={data}
        dataKey="value"
        nameKey="name"
        // Add aria-label to each segment
      >
        {data.map((entry, index) => (
          <Cell
            key={`cell-${index}`}
            fill={entry.color}
            aria-label={`${entry.name}: ${entry.value.toFixed(2)} GB`}
          />
        ))}
      </Pie>
      <Tooltip />
    </PieChart>
  </ResponsiveContainer>

  {/* Provide text alternative */}
  <div className="sr-only">
    VRAM breakdown: Model weights {breakdown.modelWeights.toFixed(2)} GB,
    KV cache {breakdown.kvCache.toFixed(2)} GB,
    Activations {breakdown.activations.toFixed(2)} GB,
    Framework overhead {breakdown.frameworkOverhead.toFixed(2)} GB
  </div>
</div>
```

Also provide a table view as alternative:
```typescript
<button onClick={() => setViewMode(mode === 'chart' ? 'table' : 'chart')}>
  Toggle table view
</button>
```

**Warning signs:**
- Lighthouse Accessibility score below 90
- Screen reader only announces "image" with no content
- Can't tab to chart elements

**Sources:**
- [Creating Accessible Data for Charts](https://216digital.com/creating-accessible-data-for-charts-and-graphs/)
- [WCAG Best Practices 2026](https://www.thewcag.com/best-practices)

---

### Pitfall 6: React 19 useFormStatus Called Outside <form>
**What goes wrong:** Runtime error "useFormStatus must be called from a component inside a <form>". Hook returns null or throws.

**Why it happens:** React 19 useFormStatus only works in child components of <form> element. Can't be called in same component that renders <form>.

**How to avoid:**
```typescript
// WRONG - in same component as form
function MyForm() {
  const { pending } = useFormStatus() // ERROR
  return <form>{/* ... */}</form>
}

// RIGHT - in child of form
function SubmitButton() {
  const { pending } = useFormStatus() // Works
  return <button disabled={pending}>Submit</button>
}

function MyForm() {
  return (
    <form>
      <SubmitButton />
    </form>
  )
}
```

For this project: Unlikely to need useFormStatus since calculations are instant (Web Worker). Use useInferenceCalculation's loading state instead.

**Warning signs:**
- Error mentions "useFormStatus must be called from a component inside"
- Hook returns undefined or null unexpectedly

**Sources:**
- [React 19 useFormStatus Docs](https://react.dev/reference/react-dom/hooks/useFormStatus)
- [Avoid using React's useFormStatus](https://allanlasser.com/posts/2024-01-26-avoid-using-reacts-useformstatus)

---

### Pitfall 7: Zustand Persist Middleware Hydration Race
**What goes wrong:** Component reads Zustand state before localStorage hydrates. Initial render shows wrong values (e.g., dark mode off when it should be on), then flickers to correct state.

**Why it happens:** Zustand persist middleware is asynchronous. First render happens with initial state, then useEffect hydrates from localStorage.

**How to avoid:**
```typescript
// Add hasHydrated flag to store
interface UIState {
  _hasHydrated: boolean
  // ... rest of state
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      _hasHydrated: false,
      // ... rest of state
    }),
    {
      name: 'llmvram-ui-preferences',
      onRehydrateStorage: () => (state) => {
        state?._hasHydrated = true
      },
    }
  )
)

// Use in component
function MyComponent() {
  const hasHydrated = useUIStore((s) => s._hasHydrated)

  if (!hasHydrated) {
    return <LoadingSpinner />
  }

  // Safe to render now
}
```

For dark mode specifically, the inline <script> in index.html (Pitfall 2) avoids this issue.

**Warning signs:**
- State flickers on first load
- localStorage values ignored on mount
- Tests fail because state not initialized

**Sources:**
- [Zustand TypeScript Best Practices 2026](https://tessl.io/skills/github/jezweb/claude-skills/zustand-state-management)

## Code Examples

All examples verified from official sources or current best practices.

### Dark Mode Toggle Component
```typescript
// src/components/common/DarkModeToggle.tsx
import { useUIStore } from '@store/uiStore'

export function DarkModeToggle() {
  const { isDarkMode, toggleDarkMode } = useUIStore()

  return (
    <button
      onClick={toggleDarkMode}
      className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDarkMode ? (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
          <path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" />
        </svg>
      ) : (
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
          <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
        </svg>
      )}
    </button>
  )
}
```

### Fit Indicator with Color Coding
```typescript
// src/components/outputs/FitIndicator.tsx
import type Decimal from 'decimal.js'

interface Props {
  totalVRAM: Decimal
  availableVRAM: number
}

export function FitIndicator({ totalVRAM, availableVRAM }: Props) {
  const needed = totalVRAM.toNumber()
  const percentage = (needed / availableVRAM) * 100

  const getStatus = () => {
    if (percentage <= 80) return { color: 'green', text: 'Fits comfortably', emoji: '✓' }
    if (percentage <= 95) return { color: 'yellow', text: 'Tight fit', emoji: '⚠' }
    return { color: 'red', text: 'Does not fit', emoji: '✗' }
  }

  const status = getStatus()

  return (
    <div
      className={`rounded-lg p-6 ${
        status.color === 'green'
          ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
          : status.color === 'yellow'
          ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
          : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
      } border-2`}
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">{status.emoji} {status.text}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Using {needed.toFixed(2)} GB of {availableVRAM} GB ({percentage.toFixed(1)}%)
          </p>
        </div>
        <div className="text-4xl font-bold">
          {percentage.toFixed(0)}%
        </div>
      </div>

      {/* Progress bar */}
      <div className="mt-4 h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${
            status.color === 'green' ? 'bg-green-500' :
            status.color === 'yellow' ? 'bg-yellow-500' :
            'bg-red-500'
          }`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  )
}
```

### Recommendations Generator
```typescript
// src/components/outputs/Recommendations.tsx
import type { Model, GPU } from '@types'
import type { InferenceVRAMBreakdown } from '@engines/types'

interface Props {
  currentConfig: { model: Model; gpu: GPU }
  breakdown: InferenceVRAMBreakdown
}

export function Recommendations({ currentConfig, breakdown }: Props) {
  const { model, gpu } = currentConfig
  const totalNeeded = breakdown.total.toNumber()
  const deficit = totalNeeded - gpu.vram_gb

  return (
    <div className="rounded-lg border-2 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-6">
      <h3 className="text-lg font-semibold mb-4">💡 Recommendations</h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Need {deficit.toFixed(2)} GB more VRAM. Try these options:
      </p>

      <ul className="space-y-3">
        <li className="flex items-start">
          <span className="font-semibold mr-2">1.</span>
          <div>
            <strong>Lower quantization:</strong> Try GPTQ or AWQ 4-bit (saves ~{(breakdown.modelWeights.toNumber() * 0.5).toFixed(2)} GB)
          </div>
        </li>

        <li className="flex items-start">
          <span className="font-semibold mr-2">2.</span>
          <div>
            <strong>Reduce context length:</strong> Lower from current to 2048 tokens (saves ~{(breakdown.kvCache.toNumber() * 0.5).toFixed(2)} GB)
          </div>
        </li>

        <li className="flex items-start">
          <span className="font-semibold mr-2">3.</span>
          <div>
            <strong>Use multiple GPUs:</strong> Need {Math.ceil(totalNeeded / gpu.vram_gb)} × {gpu.name} (coming in Phase 4)
          </div>
        </li>

        <li className="flex items-start">
          <span className="font-semibold mr-2">4.</span>
          <div>
            <strong>Upgrade GPU:</strong> Consider {gpu.vram_gb < 24 ? 'RTX 4090 (24GB)' : 'A100 (80GB)'}
          </div>
        </li>
      </ul>
    </div>
  )
}
```

### Toast Notifications with sonner
```typescript
// src/App.tsx
import { Toaster, toast } from 'sonner'

function App() {
  // Example: show error toast
  const handleError = (error: string) => {
    toast.error('Calculation failed', {
      description: error,
      action: {
        label: 'Retry',
        onClick: () => console.log('Retry clicked'),
      },
    })
  }

  // Example: show success toast
  const handleCustomModelSaved = () => {
    toast.success('Custom model saved', {
      description: 'You can now select it from the dropdown',
    })
  }

  return (
    <>
      <Toaster richColors position="top-right" />
      {/* rest of app */}
    </>
  )
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Redux for simple UI state | Zustand or Jotai | 2022-2023 | 90% less boilerplate for non-complex state |
| CSS Modules or styled-components | Tailwind utility-first | 2021-present | Faster development, smaller bundles |
| react-hot-toast | sonner | 2023-present | Better stacking, keyboard shortcuts, simpler API |
| Manual dark mode with Context | Tailwind dark: + localStorage | 2020-present | No JS for styles, better performance |
| Class component + lifecycle | Hooks (useState, useEffect) | 2019 (React 16.8) | Simpler code, better reuse |
| Victory or D3 for React charts | Recharts | 2020-present | Simpler API, good enough for most use cases |
| Manual form state management | React Hook Form or native React 19 actions | 2024 (React 19) | Less boilerplate, better performance |

**Deprecated/outdated:**
- **React 18 form patterns**: React 19 useActionState/useFormStatus replace manual useState + async handlers (but this project may not need them since calculations are instant)
- **Tailwind v3 JavaScript config**: v4 prefers CSS @theme blocks
- **PostCSS for Tailwind**: v4 has first-party Vite plugin
- **Redux for simple apps**: Overkill unless complex async workflows or time-travel debugging needed

## Open Questions

1. **Custom model/GPU input UX**
   - What we know: Need to support custom inputs (Phase 3 requirement)
   - What's unclear: Should custom inputs be inline in dropdown (like "Add custom model" option) or separate button/modal? How to persist custom entries?
   - Recommendation: Start with modal approach (cleaner separation), store custom entries in Zustand with persist middleware so they survive page refresh

2. **Chart type preference**
   - What we know: Donut chart is standard for breakdown visualization
   - What's unclear: Should we offer stacked bar chart alternative? Users may prefer different views
   - Recommendation: Start with donut only (VIZ-01 explicitly mentions donut/stacked), add toggle in later iteration if users request it

3. **Mobile-first breakpoints**
   - What we know: Must be responsive (VIZ-06 requirement)
   - What's unclear: Should layout stack vertically on mobile (inputs top, results bottom) or use tabs?
   - Recommendation: Vertical stack on mobile (sm: breakpoint), side-by-side panels on desktop (lg: breakpoint). Simpler than tabs, no state management needed

## Sources

### Primary (HIGH confidence)
- [Tailwind CSS v4 Official Docs](https://tailwindcss.com/docs/dark-mode) - Dark mode, responsive design
- [Tailwind CSS v4 Release](https://tailwindcss.com/blog/tailwindcss-v4) - What's new, configuration changes
- [React 19 Official Docs](https://react.dev/blog/2024/12/05/react-19) - Form handling, new hooks
- [Recharts Examples](https://recharts.github.io/en-US/examples/StackedBarChart/) - Chart patterns
- [Zustand GitHub](https://github.com/pmndrs/zustand) - State management patterns
- [sonner GitHub](https://github.com/emilkowalski/sonner) - Toast notifications
- [Headless UI Docs](https://headlessui.com/react/combobox) - Accessible components

### Secondary (MEDIUM confidence)
- [React Hook Form with Zod Guide 2026](https://dev.to/marufrahmanlive/react-hook-form-with-zod-complete-guide-for-2026-1em1) - Form validation patterns
- [Tailwind v4 Dark Mode Fix](https://medium.com/@balpetekserhat/how-i-fixed-tailwind-css-v4-dark-mode-not-working-in-a-vite-react-project-d7f0b3a31184) - @custom-variant requirement
- [Creating Accessible Data for Charts](https://216digital.com/creating-accessible-data-for-charts-and-graphs/) - ARIA patterns
- [WCAG Best Practices 2026](https://www.thewcag.com/best-practices) - Accessibility standards
- [React 19 Form Handling Guide](https://syntackle.com/blog/form-handling-in-react-19/) - useActionState/useFormStatus
- [Zustand State Management Skill](https://tessl.io/skills/github/jezweb/claude-skills/zustand-state-management) - TypeScript patterns

### Tertiary (Issues/Discussions - for pitfall awareness)
- [Recharts TypeScript Issue #2385](https://github.com/recharts/recharts/issues/2385) - ResponsiveContainer type errors
- [Tailwind v4 + React 19 Discussion #16201](https://github.com/tailwindlabs/tailwindcss/discussions/16201) - Integration issues
- [Recharts ResponsiveContainer Warning #6716](https://github.com/recharts/recharts/issues/6716) - Width/height warnings

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already installed and locked from Phase 1/2
- Architecture patterns: HIGH - Zustand, Recharts, Tailwind patterns well-documented and verified
- Dark mode implementation: HIGH - Official Tailwind docs + multiple verified sources for v4 specifics
- Pitfalls: HIGH - All pitfalls verified from official sources or GitHub issues
- Accessibility: MEDIUM - WCAG patterns are standard but Recharts-specific implementation needs testing

**Research date:** 2026-02-09
**Valid until:** ~30 days (Tailwind v4 and React 19 are stable, ecosystem changes slowly)

**Next steps for planner:**
1. Create component tasks following architecture patterns
2. Start with Zustand store (foundation for all UI state)
3. Build input components (model/GPU selectors, sliders, dropdowns)
4. Build output components (chart, fit indicator, recommendations)
5. Implement dark mode with FOUC prevention
6. Add responsive layout and accessibility
7. Test on mobile, tablet, desktop
