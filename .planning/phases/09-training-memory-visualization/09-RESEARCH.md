# Phase 9: Training Memory Visualization - Research

**Researched:** 2026-02-10
**Domain:** React data visualization with Recharts
**Confidence:** HIGH

## Summary

Phase 9 enhances the existing basic training breakdown table (added in Phase 8 Plan 08-03) with proper Recharts pie chart visualization and a dedicated table component. The existing inference visualization components (VRAMBreakdownChart, MemoryBreakdownTable) provide proven patterns to adapt for training data.

Training memory breakdown differs from inference in categories displayed: weights (model/base/adapter), master weights (FP32 for mixed precision), gradients, optimizer states, activations, and framework overhead. The data structure (TrainingVRAMBreakdown, LoRAVRAMBreakdown) is already calculated by engines and returned by useTrainingCalculation hook.

Key technical requirements: convert Decimal.js values to numbers for Recharts, handle conditional fields (masterWeights only for FP16/BF16, baseWeights/adapterWeights only for LoRA/QLoRA), maintain accessibility (ARIA labels, screen reader text), and ensure real-time reactivity (automatic via Zustand/React).

**Primary recommendation:** Create dedicated TrainingBreakdownChart and TrainingBreakdownTable components following the proven patterns from VRAMBreakdownChart and MemoryBreakdownTable, with conditional rendering logic for method-specific fields (LoRA vs full fine-tuning) using TypeScript discriminated unions.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Recharts | 2.x | React data visualization | Already in use for inference charts, declarative API, TypeScript support, widely adopted |
| Decimal.js | 10.x | Arbitrary precision arithmetic | Already in use throughout engines, prevents floating-point errors in VRAM calculations |
| React | 19.x | UI framework | Project standard, latest version with improved concurrent rendering |
| TypeScript | 5.x | Type safety | Project uses strict mode with noUncheckedIndexedAccess enabled |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Tailwind CSS | 4.x | Styling | Project standard, utility-first approach for component styling |
| Zustand | 4.x | State management | Already manages training configuration, auto-triggers re-renders |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Recharts | Chart.js | Chart.js has better performance at 1M+ data points, but training breakdowns have 6-10 categories max—Recharts declarative API better fits React paradigm |
| Recharts | D3.js | D3 offers more control, but requires imperative DOM manipulation—Recharts wraps D3 concepts in React-friendly components |
| Pie chart | Stacked bar chart | Bars better for comparing across multiple configurations, but pie chart better for showing proportion breakdown at single point in time (our use case) |

**Installation:**
```bash
# Already installed in project
# Verify with: npm list recharts decimal.js
```

## Architecture Patterns

### Recommended Component Structure
```
src/components/outputs/
├── TrainingBreakdownChart.tsx     # New: Pie chart for training memory
├── TrainingBreakdownTable.tsx     # New: Table with conditional rows
├── VRAMBreakdownChart.tsx         # Existing: Template to adapt
└── MemoryBreakdownTable.tsx       # Existing: Template to adapt
```

### Pattern 1: Reusable Visualization Components with Conditional Rendering
**What:** Create specialized components for training visualization that handle both TrainingVRAMBreakdown and LoRAVRAMBreakdown using TypeScript discriminated unions.

**When to use:** When data structure varies based on training method (full vs LoRA/QLoRA).

**Example:**
```typescript
// Source: Existing VRAMBreakdownChart.tsx adapted for training
import type { TrainingVRAMBreakdown, LoRAVRAMBreakdown } from '@engines/types'
import { Pie, PieChart, ResponsiveContainer, Cell, Tooltip, Legend } from 'recharts'

interface TrainingBreakdownChartProps {
  breakdown: TrainingVRAMBreakdown | LoRAVRAMBreakdown
}

export function TrainingBreakdownChart({ breakdown }: TrainingBreakdownChartProps) {
  // Type guard for LoRA/QLoRA
  const isLoRA = 'baseWeights' in breakdown

  // Build data array conditionally
  const data = []

  if (isLoRA) {
    data.push(
      { name: 'Base Model Weights', value: breakdown.baseWeights.toNumber(), color: '#3b82f6' },
      { name: 'Adapter Weights', value: breakdown.adapterWeights.toNumber(), color: '#06b6d4' }
    )
  } else {
    data.push({ name: 'Model Weights', value: breakdown.modelWeights.toNumber(), color: '#3b82f6' })
  }

  // Master weights only for mixed precision (fp16/bf16)
  if (breakdown.masterWeights.greaterThan(0)) {
    data.push({ name: 'Master Weights (FP32)', value: breakdown.masterWeights.toNumber(), color: '#8b5cf6' })
  }

  data.push(
    { name: 'Gradients', value: breakdown.gradients.toNumber(), color: '#ef4444' },
    { name: 'Optimizer States', value: breakdown.optimizerStates.toNumber(), color: '#f59e0b' },
    { name: 'Activations', value: breakdown.activations.toNumber(), color: '#10b981' },
    { name: 'Framework Overhead', value: breakdown.frameworkOverhead.toNumber(), color: '#6b7280' }
  )

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          cx="50%"
          cy="50%"
          outerRadius={100}
          innerRadius={60}
          label
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => `${Number(value).toFixed(2)} GB`} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}
```

### Pattern 2: Decimal.js to Number Conversion for Recharts
**What:** Always convert Decimal values to numbers using `.toNumber()` before passing to Recharts.

**When to use:** Every time passing breakdown data to chart components.

**Example:**
```typescript
// Source: Existing VRAMBreakdownChart.tsx lines 21-42
const data = [
  {
    name: 'Model Weights',
    value: breakdown.modelWeights.toNumber(), // Convert Decimal to number
    color: COLORS.modelWeights,
  },
  // ... more entries
]

// WRONG: Don't pass Decimal objects directly
// value: breakdown.modelWeights  // ❌ Recharts won't understand Decimal type
```

### Pattern 3: Accessible Chart with Screen Reader Support
**What:** Provide multiple accessibility layers: ARIA labels, color indicators with text, and sr-only text alternative.

**When to use:** All data visualization components.

**Example:**
```typescript
// Source: Existing VRAMBreakdownChart.tsx lines 109-170
export function TrainingBreakdownChart({ breakdown }: TrainingBreakdownChartProps) {
  return (
    <div
      role="img"
      aria-label="Training memory breakdown chart"
      className="text-gray-900 dark:text-gray-100"
    >
      <div className="w-full h-72 sm:h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>{/* ... */}</PieChart>
        </ResponsiveContainer>
      </div>

      {/* Screen reader accessible text alternative */}
      <div className="sr-only">
        Training memory breakdown: Model Weights {breakdown.modelWeights.toFixed(2)} GB,
        Gradients {breakdown.gradients.toFixed(2)} GB,
        Optimizer States {breakdown.optimizerStates.toFixed(2)} GB,
        Activations {breakdown.activations.toFixed(2)} GB,
        Framework Overhead {breakdown.frameworkOverhead.toFixed(2)} GB,
        Total {breakdown.total.toFixed(2)} GB
      </div>
    </div>
  )
}
```

### Pattern 4: Performance Optimization with Memoization
**What:** Memoize Recharts props (functions, objects) to prevent unnecessary re-renders.

**When to use:** When passing custom formatters, label renderers, or data transformation functions.

**Example:**
```typescript
// Source: Research - Recharts performance best practices
import { useMemo, useCallback } from 'react'

export function TrainingBreakdownChart({ breakdown }: TrainingBreakdownChartProps) {
  // Memoize data array to prevent recalculation on every render
  const data = useMemo(() => {
    // ... build data array from breakdown
    return dataArray
  }, [breakdown]) // Only recalculate when breakdown changes

  // Memoize tooltip formatter
  const tooltipFormatter = useCallback((value: number) => {
    return `${value.toFixed(2)} GB`
  }, [])

  return (
    <PieChart>
      <Pie data={data} dataKey="value" />
      <Tooltip formatter={tooltipFormatter} />
    </PieChart>
  )
}
```

### Anti-Patterns to Avoid
- **Unstable dataKey props:** Don't compute dataKey inline—keep it stable to avoid recalculation of all points
- **Passing Decimal objects directly:** Always convert to numbers; Recharts doesn't understand Decimal.js type
- **Color-only differentiation:** Don't rely solely on color; add text labels and patterns for colorblind accessibility
- **Missing screen reader text:** Always provide sr-only text alternative for charts
- **Inline data transformation:** Don't transform data inside JSX; use useMemo to prevent recalculation

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Chart rendering | Custom SVG drawing logic | Recharts components | Handles responsiveness, legends, tooltips, accessibility roles automatically |
| Number precision | Native JavaScript arithmetic | Decimal.js (already in use) | Prevents floating-point errors (0.1 + 0.2 = 0.30000000000000004) |
| Pie slice layout | Manual angle calculation | Recharts Pie component | Handles label positioning, slice sizing, animations, active states |
| Dark mode styling | Manual CSS variables | Tailwind dark: classes + Zustand isDarkMode | Already integrated, consistent with project patterns |
| Type guards | Runtime type checking | TypeScript discriminated unions | Compile-time type narrowing, exhaustiveness checking |

**Key insight:** Recharts abstracts complex chart layout calculations (angles, positions, responsiveness) and provides React-friendly API. Building custom SVG charts would require recreating ResponsiveContainer logic, legend positioning, tooltip triggers, accessibility features—hundreds of lines of code already solved by Recharts.

## Common Pitfalls

### Pitfall 1: Decimal.js Precision Loss on Conversion
**What goes wrong:** Converting numbers to Decimal via number literals loses precision.

**Why it happens:** JavaScript floats can't represent some decimals exactly (0.7 + 0.1 = 0.7999999999999999).

**How to avoid:** This is already avoided in the project—engines use Decimal throughout and only convert to numbers at the visualization boundary using `.toNumber()`.

**Warning signs:** Financial/VRAM calculations showing unexpected trailing digits.

### Pitfall 2: Breaking Type Guards with Incorrect Conditionals
**What goes wrong:** Using `breakdown.method === 'lora'` instead of `'baseWeights' in breakdown` breaks type narrowing.

**Why it happens:** TypeScript discriminated unions require checking the discriminant property directly (the unique field that exists only in one type).

**How to avoid:**
```typescript
// ✅ CORRECT: Type guard on discriminant property
if ('baseWeights' in breakdown) {
  // TypeScript narrows to LoRAVRAMBreakdown
  const baseGB = breakdown.baseWeights.toNumber() // ✅ Type-safe
}

// ❌ WRONG: Checking method field doesn't narrow type
if (breakdown.method === 'lora') {
  const baseGB = breakdown.baseWeights.toNumber() // ❌ Type error
}
```

**Warning signs:** TypeScript errors like "Property 'baseWeights' does not exist on type 'TrainingVRAMBreakdown'".

### Pitfall 3: Recharts Re-renders from Unstable Props
**What goes wrong:** Chart re-renders on every parent render, even when data hasn't changed.

**Why it happens:** Passing inline objects/functions as props creates new references each render, causing Recharts to recalculate everything.

**How to avoid:** Wrap data transformations in `useMemo` and functions in `useCallback`.

**Warning signs:** Performance issues, chart flickering, slow responsiveness when changing unrelated inputs.

### Pitfall 4: Missing Conditional Rows in Table
**What goes wrong:** Table always shows "Master Weights" row even for FP32 training (where it should be 0).

**Why it happens:** Not checking if values are significant before adding rows.

**How to avoid:**
```typescript
// ✅ CORRECT: Only add row if value is significant
const rows = []
if (breakdown.masterWeights.greaterThan(0)) {
  rows.push({ name: 'Master Weights (FP32)', value: breakdown.masterWeights, color: '#8b5cf6' })
}

// ❌ WRONG: Always adding row
rows.push({ name: 'Master Weights (FP32)', value: breakdown.masterWeights, color: '#8b5cf6' })
// Shows "Master Weights: 0.00 GB (0.0%)" for FP32 training—confusing to users
```

**Warning signs:** Users see 0.00 GB rows in breakdown, cluttering the UI unnecessarily.

### Pitfall 5: Accessibility Only for Mouse Users
**What goes wrong:** Charts work with mouse but not with keyboard or screen readers.

**Why it happens:** Recharts doesn't provide full accessibility out of the box; requires explicit ARIA attributes and sr-only text.

**How to avoid:** Always add `role="img"`, `aria-label`, and sr-only text alternative (already pattern in VRAMBreakdownChart.tsx).

**Warning signs:** Screen reader announces "blank" or "group" instead of meaningful chart description.

## Code Examples

Verified patterns from existing codebase:

### Inference Breakdown Chart (Template to Adapt)
```typescript
// Source: src/components/outputs/VRAMBreakdownChart.tsx (lines 17-82)
export function VRAMBreakdownChart({ breakdown }: VRAMBreakdownChartProps) {
  const isDarkMode = useUIStore((s) => s.isDarkMode)

  // Convert Decimal values to numbers for Recharts
  const data = [
    { name: 'Model Weights', value: breakdown.modelWeights.toNumber(), color: COLORS.modelWeights },
    { name: 'KV Cache', value: breakdown.kvCache.toNumber(), color: COLORS.kvCache },
    { name: 'Activations', value: breakdown.activations.toNumber(), color: COLORS.activations },
    { name: 'Framework Overhead', value: breakdown.frameworkOverhead.toNumber(), color: COLORS.frameworkOverhead },
  ]

  const totalGB = breakdown.total.toNumber()

  const renderLabel = (props: PieLabelRenderProps) => {
    const { value, cx, cy, midAngle, innerRadius, outerRadius } = props
    // Guard against undefined values (Recharts can pass undefined during animation)
    if (typeof value !== 'number' || typeof cx !== 'number' /* ... */) {
      return null
    }

    // Only show label if value is significant (>1% of total)
    if (value < totalGB * 0.01) return null

    // Calculate label position using polar coordinates
    const RADIAN = Math.PI / 180
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)

    return <text x={x} y={y} fill="white">{value.toFixed(1)}</text>
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={renderLabel}
          outerRadius={100}
          innerRadius={60}
          dataKey="value"
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip formatter={(value) => `${Number(value).toFixed(2)} GB`} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}
```

### Inference Breakdown Table (Template to Adapt)
```typescript
// Source: src/components/outputs/MemoryBreakdownTable.tsx (lines 21-84)
export function MemoryBreakdownTable({ breakdown }: MemoryBreakdownTableProps) {
  const rows: RowData[] = [
    { name: 'Model Weights', value: breakdown.modelWeights, color: COLORS.modelWeights },
    { name: 'KV Cache', value: breakdown.kvCache, color: COLORS.kvCache },
    { name: 'Activations', value: breakdown.activations, color: COLORS.activations },
    { name: 'Framework Overhead', value: breakdown.frameworkOverhead, color: COLORS.frameworkOverhead },
  ]

  const calculatePercentage = (value: Decimal): string => {
    if (breakdown.total.isZero()) return '0.0'
    return value.div(breakdown.total).mul(100).toFixed(1)
  }

  return (
    <table className="w-full text-sm border-collapse text-gray-900 dark:text-gray-100">
      <thead>
        <tr className="bg-gray-100 dark:bg-gray-800">
          <th className="text-left font-medium px-4 py-2">Component</th>
          <th className="text-right font-medium px-4 py-2">Size (GB)</th>
          <th className="text-right font-medium px-4 py-2">% of Total</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={row.name} className={index % 2 === 1 ? 'bg-gray-50 dark:bg-gray-800/50' : ''}>
            <td className="px-4 py-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: row.color }} />
                <span>{row.name}</span>
              </div>
            </td>
            <td className="text-right px-4 py-2 tabular-nums">{row.value.toFixed(2)}</td>
            <td className="text-right px-4 py-2 tabular-nums">{calculatePercentage(row.value)}%</td>
          </tr>
        ))}
        <tr className="font-bold border-t-2 border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Total</td>
          <td className="text-right px-4 py-2 tabular-nums">{breakdown.total.toFixed(2)}</td>
          <td className="text-right px-4 py-2 tabular-nums">100.0%</td>
        </tr>
      </tbody>
    </table>
  )
}
```

### Type Guard for LoRA vs Full Fine-Tuning
```typescript
// Source: src/components/layout/ResultsPanel.tsx (lines 285-315)
// Type guard to check if result is LoRA/QLoRA
const isLoRA = 'baseWeights' in trainingResult

if (isLoRA) {
  const loraResult = trainingResult as LoRAVRAMBreakdown
  rows.push({
    name: 'Base Model Weights',
    value: loraResult.baseWeights.toNumber(),
    color: '#3b82f6',
  })
  rows.push({
    name: 'Adapter Weights',
    value: loraResult.adapterWeights.toNumber(),
    color: '#06b6d4',
  })
} else {
  rows.push({
    name: 'Model Weights',
    value: trainingResult.modelWeights.toNumber(),
    color: '#3b82f6',
  })
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Inline table in ResultsPanel | Dedicated TrainingBreakdownChart + TrainingBreakdownTable components | Phase 9 | Better separation of concerns, reusable components, consistent with inference visualization |
| Chart.js for React | Recharts | 2019-2020 | Declarative React-first API, better TypeScript support, easier composition |
| Manual color schemes | Tailwind color palette | Tailwind v3+ (2021) | Consistent colors across light/dark modes, semantic color names |
| Inline type checks | Discriminated unions | TypeScript 2.0+ (2016) | Compile-time type narrowing, exhaustiveness checking, better IDE support |

**Deprecated/outdated:**
- `recharts@1.x`: Version 2.x (2023+) improved TypeScript types, performance, accessibility hooks
- Manual ARIA attributes for charts: Modern Recharts has `accessibilityLayer` prop (though still requires sr-only text supplement)
- Class components: React Hooks (2019+) replaced class-based state management with functional components

## Open Questions

1. **Should we add a toggle to switch between chart and table view?**
   - What we know: Some users prefer visual (pie chart), others prefer precise numbers (table)
   - What's unclear: Whether to show both simultaneously (current pattern) or add toggle (saves space)
   - Recommendation: Show both simultaneously for Phase 9 (matches inference pattern). If user feedback requests it, add toggle in future iteration.

2. **Should we add animations to chart transitions?**
   - What we know: Recharts supports animations (animationBegin, animationDuration, animationEasing props)
   - What's unclear: Whether animations improve UX or feel distracting when rapidly changing training config
   - Recommendation: Enable default Recharts animations (isAnimationActive="auto"). Can disable if users report it feels sluggish.

3. **Should master weights row show "0.00 GB" for FP32 or be hidden entirely?**
   - What we know: Mixed precision (FP16/BF16) requires master weights, FP32 doesn't (masterWeights = 0)
   - What's unclear: Does showing "0.00 GB" help users understand the distinction, or clutter the UI?
   - Recommendation: Hide the row entirely when masterWeights <= 0 (use `breakdown.masterWeights.greaterThan(0)` guard). Cleaner UI, matches user mental model.

## Sources

### Primary (HIGH confidence)
- [Recharts Pie API Documentation](https://recharts.github.io/api/Pie/) - Official API reference for Pie component props
- [Recharts PieChart API Documentation](https://recharts.github.io/en-US/api/PieChart/) - Official API reference for PieChart container
- [Recharts Performance Guide](https://recharts.github.io/en-US/guide/performance/) - Official performance best practices
- [Decimal.js API Documentation](https://mikemcl.github.io/decimal.js/) - Official documentation for arbitrary-precision arithmetic
- Existing codebase: `src/components/outputs/VRAMBreakdownChart.tsx`, `MemoryBreakdownTable.tsx`, `ResultsPanel.tsx` - Proven patterns already in production

### Secondary (MEDIUM confidence)
- [Efficient Deep Learning: Optimization Techniques - Hugging Face](https://huggingface.co/blog/Isayoften/optimization-rush) - Training memory component categories (optimizer states, gradients, weights, activations)
- [Efficient Training on a Single GPU - Hugging Face](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one) - Mixed precision training memory requirements
- [How to Choose Between Bar Chart and Pie Chart - Atlassian](https://www.atlassian.com/data/charts/how-to-choose-pie-chart-vs-bar-chart) - Visualization best practices for proportion breakdowns
- [TypeScript Discriminated Unions for React - OneUpTime (Jan 2026)](https://oneuptime.com/blog/post/2026-01-15-typescript-discriminated-unions-react-props/view) - Current best practices for type-safe conditional rendering
- [WCAG Data Visualization Accessibility - The A11Y Collective](https://www.a11y-collective.com/blog/accessible-charts/) - Comprehensive accessibility checklist for charts

### Tertiary (LOW confidence - marked for validation)
- [Recharts and Accessibility - GitHub Wiki](https://github.com/recharts/recharts/wiki/Recharts-and-accessibility/ad2642f9fa2e43411621e9e29e3c6ea2d6396234) - Older wiki page, may be outdated
- [Color Palette Generators 2026 - Venngage](https://venngage.com/blog/color-palette-generators/) - General color trends, not specific to training visualization

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Recharts and Decimal.js already in use, proven patterns exist in codebase
- Architecture: HIGH - Existing VRAMBreakdownChart and MemoryBreakdownTable provide direct templates to adapt
- Pitfalls: MEDIUM - Common Recharts issues documented, but some training-specific edge cases may emerge during implementation

**Research date:** 2026-02-10
**Valid until:** ~30 days (stable technologies, no fast-moving APIs expected)
