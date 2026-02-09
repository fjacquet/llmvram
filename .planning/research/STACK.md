# Technology Stack

**Project:** LLM VRAM Calculator
**Researched:** 2026-02-09
**Overall Confidence:** MEDIUM (training data from Jan 2025, external verification tools unavailable)

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **React** | 19.x | UI framework | Mature, component-based, excellent TypeScript support. React 19 includes compiler optimizations and improved hooks. Sister project (raidy) already uses this - shared patterns reduce cognitive load. |
| **TypeScript** | 5.8+ (strict mode) | Type safety | Essential for calculation accuracy. Strict mode catches numerical edge cases at compile time. Prevents runtime errors in VRAM estimation logic. |

**Confidence:** HIGH for React/TS as foundation, MEDIUM for version specifics (need verification)

**Rationale for React over alternatives:**
- **vs Vue 3:** React's ecosystem for complex calculations is more mature (better TypeScript inference, more numerical libraries)
- **vs Svelte:** React's larger ecosystem means more calculator/scientific tool examples to learn from
- **vs Solid:** React's stability and raidy alignment outweigh Solid's performance gains for this use case

### Build Tool

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Vite** | 7.x | Build tool & dev server | Fast HMR, native ESM, optimal for calculation-heavy SPAs. Tree-shaking critical for keeping bundle small. Superior DX compared to webpack. Aligns with raidy. |

**Confidence:** MEDIUM (Vite 7 may not be released yet - Vite 5.x was latest in my Jan 2025 training. Verify current version.)

**Why Vite over alternatives:**
- **vs webpack:** Dramatically faster dev experience, simpler config for static sites
- **vs Parcel:** More predictable build output, better plugin ecosystem for optimization
- **vs esbuild directly:** Vite provides better DX wrapper while still using esbuild under the hood

### State Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Zustand** | 5.x | Global state management | Lightweight (1kb), minimal boilerplate, perfect for calculator state (model selections, GPU configs, calculation results). No provider hell. Aligns with raidy patterns. |

**Confidence:** HIGH (excellent fit for calculator domain)

**Why Zustand for calculators:**
- **Minimal re-renders:** Selector-based subscriptions prevent unnecessary recalculations
- **DevTools:** Time-travel debugging for complex calculation flows
- **Middleware:** Persist calculation history to localStorage
- **vs Redux Toolkit:** 80% less boilerplate for same functionality
- **vs Jotai/Recoil:** Simpler mental model for derived state (calculation results from inputs)
- **vs Context API:** Better performance for frequent updates (slider changes, GPU count adjustments)

### Styling

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Tailwind CSS** | 4.x | Utility-first CSS | Rapid UI development, consistent design system. v4 has improved performance and better TypeScript integration. Aligns with raidy. Calculator UIs benefit from utility classes (responsive grids, number formatting). |

**Confidence:** MEDIUM (Tailwind v4 was early release in my training. Verify stability/release status.)

**Why Tailwind:**
- **Rapid prototyping:** Calculator UIs are form-heavy - utility classes speed development
- **Consistency:** Design tokens prevent visual drift across calculation sections
- **Bundle size:** PurgeCSS removes unused styles - critical for static deployment
- **vs CSS Modules:** Better for design system consistency
- **vs styled-components:** No runtime CSS-in-JS cost
- **vs vanilla CSS:** Faster iteration on responsive calculator layouts

**Note:** Consider Tailwind's `@apply` for complex calculator components (formula displays, result cards) to avoid className bloat.

### Visualization

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Recharts** | 2.x | Charts & graphs | React-native charting, good for VRAM breakdown visualizations (pie charts for layer allocation, bar charts for GPU comparison). Aligns with raidy. |
| **D3.js** (optional) | 7.x | Advanced visualizations | **Use sparingly** for custom needs Recharts can't handle (network graphs for multi-GPU topology, custom VRAM allocation diagrams). Heavy library - only import specific modules. |

**Confidence:** MEDIUM-HIGH (Recharts solid for basic viz, may need alternatives for advanced needs)

**Recharts rationale:**
- **Declarative:** Fits React patterns better than imperative D3
- **Responsive:** Built-in responsive containers for mobile calculator views
- **Composition:** Easy to build custom chart combinations (VRAM stacked bar + memory bandwidth line chart)

**Alternative to consider:**
- **Visx (Airbnb):** Lower-level than Recharts, higher-level than D3. Better for custom scientific visualizations. Consider if Recharts feels limiting.
- **Chart.js with react-chartjs-2:** More performant for large datasets, but less React-idiomatic

**Anti-pattern:** Don't use Recharts for static diagrams (GPU architecture, model topology). Use SVG components or static assets instead.

### Calculation Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Decimal.js** | 10.x | Arbitrary-precision arithmetic | **Critical** for VRAM calculations. Prevents floating-point errors in memory math (e.g., 0.1 + 0.2 !== 0.3). Use for all financial-style precision needs (bytes, memory percentages). |
| **mathjs** | 13.x | Advanced math operations | Optional. Use if complex formulas needed (statistical analysis of GPU efficiency, matrix operations for tensor memory). Avoid if calculations are simple - adds 500kb. |
| **zod** | 3.x | Runtime validation & parsing | Validate user inputs (model parameters, GPU configs) and JSON database schemas. TypeScript types generated from schemas ensure type/runtime alignment. |

**Confidence:** HIGH (these are standard for numerical applications)

**Decimal.js is non-negotiable** because:
- VRAM calculations involve large numbers (gigabytes, terabytes)
- Precision errors compound in multi-GPU sharding calculations
- User trust requires exact values matching vendor specs

**Zod rationale:**
- Validates JSON databases (models.json, gpus.json) at load time
- Generates TypeScript types from schemas (single source of truth)
- Error messages guide JSON curation ("expected number, got string")

### Data Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Static JSON** | N/A | Model/GPU databases | Client-side requirement. Store curated data as importable JSON modules. Vite imports as ES modules (tree-shakeable if structured well). |
| **IndexedDB** (via localforage) | 1.x | Local caching | Optional. Cache calculation results for "return to previous session" feature. Simpler API than raw IndexedDB. |

**Confidence:** HIGH

**Why static JSON:**
- **No backend needed:** Aligns with static deployment requirement
- **Version control:** Models/GPUs evolve - Git tracks database changes
- **Type safety:** Generate TypeScript types from JSON schemas (use `json-schema-to-typescript` or Zod)

**Structure recommendation:**
```typescript
// models.json - separate file per vendor for tree-shaking
export interface ModelSpec {
  id: string;
  name: string;
  parameters: number; // in billions
  architecture: string;
  layerConfig: {
    hiddenSize: number;
    numLayers: number;
    numAttentionHeads: number;
    // ... precision fields for VRAM calculation
  };
}
```

### Developer Experience

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Biome** | 1.x | Linting & formatting | All-in-one tool (replaces ESLint + Prettier). 10-100x faster than ESLint. Consistent formatting prevents calculation logic diffs from style changes. Aligns with raidy. |
| **Vitest** | 2.x | Unit testing | Native Vite integration, Jest-compatible API. Essential for testing calculation engines (VRAM estimation, sharding algorithms). Inline snapshots for regression testing. |
| **TypeScript strict mode** | N/A | Compile-time safety | Catches calculation errors at build time. Enable all strict flags: `strictNullChecks`, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`. |

**Confidence:** HIGH (all are current best practices)

**Biome over ESLint/Prettier:**
- **Single tool:** No config conflicts between linter/formatter
- **Speed:** Sub-second linting for entire codebase
- **Opinionated:** Less bikeshedding, more building

**Vitest over Jest:**
- **Native ESM:** No transform overhead for Vite-built code
- **Speed:** Parallel test execution, smart watch mode
- **DX:** Test UI included, better error messages

### Testing Strategy

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Vitest** | 2.x | Unit tests | Calculation engine tests (VRAM formulas, sharding logic, precision handling) |
| **Testing Library** | 16.x | Component tests | Form interactions, state updates, chart rendering |
| **Playwright** | 1.x | E2E tests | Full calculator workflows (select model → configure GPUs → verify results) |

**Confidence:** HIGH

**Test priorities for calculator domain:**
1. **Unit tests (critical):** VRAM calculation accuracy, edge cases (zero GPUs, max sharding)
2. **Component tests (important):** State updates, form validation, derived calculations
3. **E2E tests (nice-to-have):** Full workflows, visual regression for charts

### Package Manager

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **pnpm** | 9.x | Package management | Faster installs, disk-efficient, strict dependency resolution (prevents phantom dependencies). Better for monorepo if raidy integration later. |

**Confidence:** HIGH

**Why pnpm over npm/yarn:**
- **Speed:** Hardlinks instead of copying packages
- **Strict:** Prevents accidental imports of non-declared dependencies
- **Monorepo-ready:** If raidy + llmvram need shared components later

**Alternative:** If raidy uses npm/yarn, match it for consistency. Don't mix package managers across sister projects.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **Framework** | React 19 | Solid.js | Faster, but smaller ecosystem for numerical tools. React's stability + raidy alignment wins. |
| **Framework** | React 19 | Svelte 5 | Excellent DX, but TypeScript integration less mature. Calculator needs strict types. |
| **State** | Zustand | Jotai | Atomic model harder to reason about for large calculation state. Zustand's simplicity better for team onboarding. |
| **Charts** | Recharts | Visx | Consider for advanced viz. Recharts simpler starting point. |
| **Charts** | Recharts | Chart.js | More performant, but less React-idiomatic. Recharts integrates better with React state. |
| **Styling** | Tailwind | UnoCSS | Faster, but less mature. Tailwind's ecosystem + raidy alignment more valuable. |
| **Styling** | Tailwind | Panda CSS | Zero-runtime interesting, but ecosystem smaller. Stick with proven choice. |
| **Testing** | Vitest | Jest | Vitest is Jest-compatible but faster with Vite. No reason to use Jest. |
| **Linting** | Biome | ESLint + Prettier | Biome 10-100x faster, single tool, opinionated. ESLint's flexibility not needed here. |

## Anti-Patterns to Avoid

### Do NOT Use

| Technology | Why Avoid | What to Use Instead |
|------------|-----------|---------------------|
| **Native JavaScript numbers for VRAM** | Floating-point errors compound in calculations | **Decimal.js** for all memory arithmetic |
| **Redux Toolkit** | Overkill boilerplate for calculator state | **Zustand** (5x less code for same result) |
| **Lodash** | Tree-shaking issues, bundle bloat | Native ES2024 methods (`Object.groupBy`, `Array.at`, etc.) or individual imports |
| **Moment.js** | Deprecated, heavy (288kb) | Native `Intl.DateTimeFormat` or **date-fns** if complex needs |
| **axios** | Unnecessary wrapper for static data | Native `fetch` (works in all modern browsers) |
| **Create React App** | Deprecated, slow, webpack-based | **Vite** (already chosen) |
| **Class components** | Legacy React pattern | **Functional components + hooks** |
| **PropTypes** | Weak type checking | **TypeScript** (already chosen) |

## Domain-Specific Considerations

### VRAM Calculation Accuracy

**Critical requirements:**
1. **Precision:** Use Decimal.js for all memory math (no native Number)
2. **Validation:** Zod schemas for model/GPU specs (runtime validation)
3. **Testing:** Property-based testing with `fast-check` (test calculation invariants)
4. **Documentation:** Inline comments explaining VRAM formulas (cite papers/vendor docs)

### Performance

**Calculator-specific optimizations:**
1. **Memoization:** `useMemo` for derived calculations (VRAM per layer, total memory)
2. **Debouncing:** Slider inputs should debounce before triggering recalculation (use `use-debounce` hook)
3. **Web Workers:** For complex scenarios (1000+ layer models, 100+ GPU configurations), offload calculation to worker thread
4. **Lazy loading:** Split database JSON by vendor (import Nvidia GPUs only when needed)

### Accessibility

**Calculator UIs must be accessible:**
1. **Semantic HTML:** Use `<input type="number">` for numerical inputs (mobile keyboard optimization)
2. **ARIA labels:** Screen readers need context for calculation results
3. **Keyboard navigation:** Tab through form fields, Enter to calculate
4. **Focus management:** Focus on result section after calculation completes

## Installation

```bash
# Initialize project with Vite + React + TypeScript
pnpm create vite@latest llmvram --template react-ts
cd llmvram

# Core dependencies
pnpm add react@^19 react-dom@^19
pnpm add zustand@^5
pnpm add tailwindcss@^4 @tailwindcss/vite
pnpm add recharts@^2
pnpm add decimal.js@^10
pnpm add zod@^3

# Optional - advanced needs
pnpm add mathjs@^13              # If complex formulas needed
pnpm add localforage@^1          # If session persistence needed
pnpm add use-debounce@^10        # For input debouncing
pnpm add d3-hierarchy d3-scale   # If custom D3 viz needed (specific modules only)

# Dev dependencies
pnpm add -D typescript@^5.8
pnpm add -D vite@^7              # Verify version exists
pnpm add -D vitest@^2 @vitest/ui
pnpm add -D @testing-library/react@^16 @testing-library/user-event
pnpm add -D @biomejs/biome@^1
pnpm add -D playwright@^1        # If E2E tests needed

# Type definitions
pnpm add -D @types/react@^19 @types/react-dom@^19
pnpm add -D @types/node          # For Vite config
```

## Configuration Files

### Tailwind CSS v4 Setup

```bash
# Install Tailwind CSS v4 (verify syntax for v4)
pnpm add -D tailwindcss@^4 @tailwindcss/vite

# tailwind.config.ts
import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      // Calculator-specific design tokens
      fontFamily: {
        mono: ['JetBrains Mono', 'Monaco', 'Courier New', 'monospace'],
      },
      colors: {
        'vram-low': '#22c55e',    // Green for safe memory usage
        'vram-medium': '#f59e0b', // Orange for moderate usage
        'vram-high': '#ef4444',   // Red for near-limit usage
      },
    },
  },
} satisfies Config
```

### Biome Configuration

```jsonc
// biome.json
{
  "$schema": "https://biomejs.dev/schemas/1.0.0/schema.json",
  "organizeImports": { "enabled": true },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "suspicious": {
        "noExplicitAny": "error" // Critical for calculation types
      }
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "trailingCommas": "es5"
    }
  }
}
```

### Vitest Configuration

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: ['**/*.config.*', '**/test/**'],
      // High coverage for calculation engines
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
      },
    },
  },
})
```

## Version Verification Needed

**CRITICAL - Verify these before implementation:**

| Technology | Assumed Version | Verification Needed |
|------------|-----------------|---------------------|
| Vite | 7.x | **Check if Vite 7 released.** Latest in my training was Vite 5.x. May need to use Vite 5.x or 6.x. |
| Tailwind CSS | 4.x | **Check v4 release status.** Was in beta in my training. May need 3.x if v4 unstable. |
| React | 19.x | **Verify React 19 is stable.** Was released but check for major issues. |
| Zustand | 5.x | **Check current version.** 4.x was latest in my training. |
| Biome | 1.x | **Verify feature parity** with ESLint/Prettier for your needs. |

## Confidence Assessment

| Category | Confidence | Rationale |
|----------|-----------|-----------|
| **React + TypeScript** | HIGH | Standard choice for complex SPAs, excellent ecosystem |
| **Zustand** | HIGH | Proven pattern for calculator UIs, aligns with raidy |
| **Vitest** | HIGH | Best testing solution for Vite projects |
| **Biome** | MEDIUM-HIGH | Rising tool, may lack some ESLint plugins. Check ecosystem. |
| **Decimal.js** | HIGH | Industry standard for precision arithmetic |
| **Zod** | HIGH | Standard for TypeScript validation |
| **Recharts** | MEDIUM | Good starting point, may need Visx for advanced viz |
| **Vite 7** | LOW | **Cannot verify version exists** - check official Vite docs |
| **Tailwind v4** | LOW | **Cannot verify stability** - check official Tailwind docs |

## Next Steps - Verification Protocol

**Before roadmap creation, validate:**

1. **Visit official documentation:**
   - https://vite.dev (verify latest version)
   - https://tailwindcss.com (verify v4 status)
   - https://react.dev (verify React 19 stability)

2. **Check raidy's actual package.json:**
   - Confirm exact versions used in sister project
   - Match package manager (npm/yarn/pnpm)
   - Match tooling versions for consistency

3. **Test critical libraries:**
   - Install Decimal.js and verify precision with VRAM calculations
   - Prototype Recharts with sample VRAM breakdown data
   - Validate Zod schemas with sample model/GPU JSON

## Sources

**Verification status:** MEDIUM confidence overall

Unable to verify with authoritative sources (Context7, official docs) due to tool restrictions. Recommendations based on:
- Training data (current through January 2025)
- Domain expertise for calculator applications
- Alignment with raidy project patterns
- Industry best practices for React SPAs

**Required verification:**
- Official Vite documentation (https://vite.dev)
- Official Tailwind CSS documentation (https://tailwindcss.com)
- Official React documentation (https://react.dev)
- Raidy project's actual package.json
- Zustand documentation (https://zustand.docs.pmnd.rs)
- Recharts documentation (https://recharts.org)
- Biome documentation (https://biomejs.dev)

**Recommendation:** Cross-reference this stack analysis with current official documentation before finalizing roadmap. Version numbers may be outdated.
