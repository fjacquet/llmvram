# Phase 5: Sharing & Comparison - Research

**Researched:** 2026-02-09
**Domain:** URL state persistence and configuration comparison in React SPAs
**Confidence:** MEDIUM-HIGH

## Summary

Phase 5 enables users to share configurations via URL and compare multiple scenarios side-by-side. This requires two distinct technical capabilities:

1. **URL Hash Persistence (VIZ-04)**: Serialize Zustand store state to URL hash for shareable links, using Zustand's persist middleware with custom hash storage. State includes model/GPU selections (by ID reference), quantization settings, and all numeric parameters. Compression via lz-string keeps URLs under browser limits (~2000 chars safe across all browsers).

2. **Side-by-Side Comparison (VIZ-05)**: Store 2-3 configuration snapshots in local state, render them in parallel columns, and highlight differences (quantization format, GPU choice, VRAM totals, performance metrics). Use shallow object comparison for detecting changes and visual indicators (color coding, diff markers) to emphasize differences.

**Primary recommendation:** Use Zustand persist middleware with custom hashStorage implementation for URL state, lz-string for compression, and a simple comparison slice in the store to manage snapshots. Build comparison UI as a separate route/view with side-by-side layout using existing output components.

## Standard Stack

### Core (Already in Project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Zustand | 5.0.9 → 5.0.10 | State management with persist middleware | Built-in persist middleware with custom storage support. Official hashStorage pattern. **Upgrade to 5.0.10 for persist bug fixes.** |
| React | 19.2.0 | UI framework | Current stable version, no changes needed |
| Zod | 3.25.76 | Runtime validation | Already used for data schemas, extend for URL state validation |

### New Dependencies Required

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **lz-string** | ^1.5.0 | URL-safe compression | Compress state before encoding to URL. Use `compressToEncodedURIComponent` / `decompressFromEncodedURIComponent`. Reduces URL size by 60-70%. |

### Optional/Nice-to-Have

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **superjson** | ^2.2.1 | Enhanced JSON serialization | If Decimal.js values need URL serialization. Register custom handlers. **May not be needed** if we convert Decimals to numbers for URL state. |
| **deep-object-diff** | ^1.1.9 | Object comparison | Highlight specific changed fields in comparison view. **Start without it** - simple shallow comparison may suffice. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| lz-string | pako (gzip) | Pako has better compression but larger library size (~45kb vs 3kb). lz-string is "good enough" for config state. |
| URL hash | Query params | Hash doesn't trigger server requests, better for pure client-side SPA. Query params better for SSR/SEO (not needed here). |
| Custom comparison | react-diff-view | react-diff-view is for text/code diffs. Configuration comparison is simpler - just show values side-by-side. |
| superjson | devalue | devalue smaller but less flexible. Only use if superjson too heavy. |

**Installation:**

```bash
npm install lz-string@^1.5.0

# Optional - only if needed
npm install superjson@^2.2.1
npm install deep-object-diff@^1.1.9
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── store/
│   ├── uiStore.ts                    # Existing store - add hashStorage
│   ├── comparisonStore.ts            # NEW: Manage configuration snapshots
│   └── urlSerializer.ts              # NEW: Compress/decompress state for URL
├── components/
│   ├── comparison/                   # NEW: Comparison view components
│   │   ├── ComparisonView.tsx        # Main comparison layout
│   │   ├── ComparisonColumn.tsx      # Single config column
│   │   ├── ComparisonControls.tsx    # Add/remove snapshot buttons
│   │   └── DiffHighlight.tsx         # Visual difference indicators
│   ├── outputs/                      # REUSE: Existing output components
│   │   ├── VRAMBreakdown.tsx         # Already built, use in comparison
│   │   └── PerformanceMetrics.tsx    # Already built, use in comparison
└── hooks/
    └── useURLState.ts                # NEW: Hook to sync store ↔ URL hash
```

### Pattern 1: Hash Storage with Compression

**What:** Implement Zustand StateStorage interface to persist state to URL hash, compressed via lz-string.

**When to use:** URL persistence for sharing configurations.

**Example:**

```typescript
// src/store/urlSerializer.ts
import { compress, decompress } from 'lz-string'
import { z } from 'zod'

// Schema for URL state (subset of full store)
export const URLStateSchema = z.object({
  modelId: z.string().optional(),
  gpuId: z.string().optional(),
  quantization: z.string(),
  sequenceLength: z.number(),
  batchSize: z.number(),
  kvQuantization: z.string(),
  numGPUs: z.number(),
  shardingStrategy: z.string(),
  offloadingEnabled: z.boolean(),
  offloadTarget: z.string().optional(),
  offloadMode: z.string().optional(),
  offloadPercentage: z.number().optional(),
  offloadLayers: z.number().optional(),
  kvCacheOffload: z.boolean().optional(),
})

export type URLState = z.infer<typeof URLStateSchema>

export function serializeToURL(state: URLState): string {
  const json = JSON.stringify(state)
  return compress(json, { compressToEncodedURIComponent })
}

export function deserializeFromURL(encoded: string): URLState | null {
  try {
    const json = decompress(encoded, { decompressFromEncodedURIComponent })
    if (!json) return null
    const parsed = JSON.parse(json)
    return URLStateSchema.parse(parsed) // Validate with Zod
  } catch {
    return null // Invalid URL state, ignore
  }
}
```

```typescript
// src/store/uiStore.ts (additions)
import { StateStorage, createJSONStorage } from 'zustand/middleware'
import { serializeToURL, deserializeFromURL } from './urlSerializer'

const hashStorage: StateStorage = {
  getItem: (key): string | null => {
    const hash = location.hash.slice(1) // Remove '#'
    if (!hash) return null
    const state = deserializeFromURL(hash)
    return state ? JSON.stringify(state) : null
  },
  setItem: (key, value): void => {
    const state = JSON.parse(value)
    const compressed = serializeToURL(state)
    // Don't update if URL would be same (avoid infinite loop)
    if (location.hash.slice(1) !== compressed) {
      location.hash = compressed
    }
  },
  removeItem: (): void => {
    history.replaceState(null, '', ' ') // Clear hash
  },
}

// Modify existing store to add URL persistence
export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // ... existing state and actions
    }),
    {
      name: 'llmvram-ui-preferences',
      storage: createJSONStorage(() => hashStorage), // Use hash instead of localStorage
      partialize: (state) => ({
        // Serialize model/GPU by ID, not full object
        modelId: state.selectedModel?.id,
        gpuId: state.selectedGPU?.id,
        quantization: state.quantization,
        sequenceLength: state.sequenceLength,
        batchSize: state.batchSize,
        kvQuantization: state.kvQuantization,
        numGPUs: state.numGPUs,
        shardingStrategy: state.shardingStrategy,
        offloadingEnabled: state.offloadingEnabled,
        offloadTarget: state.offloadTarget,
        offloadMode: state.offloadMode,
        offloadPercentage: state.offloadPercentage,
        offloadLayers: state.offloadLayers,
        kvCacheOffload: state.kvCacheOffload,
        // DO NOT persist: selectedModel (full object), selectedGPU (full object), isDarkMode
      }),
      // Hydrate: Convert modelId/gpuId back to full objects
      onRehydrateStorage: () => (state) => {
        if (state?.modelId) {
          // Look up model by ID from models.json
          state.selectedModel = findModelById(state.modelId)
        }
        if (state?.gpuId) {
          // Look up GPU by ID from gpus.json
          state.selectedGPU = findGPUById(state.gpuId)
        }
      },
    },
  ),
)
```

**Key decisions:**

- **Store IDs, not objects**: URL contains `modelId: "llama-3-70b"`, not full model object. Hydrate full object on load.
- **Compression is mandatory**: Uncompressed JSON easily exceeds 2000 chars. lz-string reduces by 60-70%.
- **Validation on deserialize**: Use Zod to validate URL state. Ignore invalid/corrupted URLs gracefully.
- **Dark mode NOT in URL**: UI preferences like dark mode stay in localStorage (existing persist), not URL.

### Pattern 2: Configuration Snapshots for Comparison

**What:** Store array of configuration snapshots (2-3 max) in separate comparison store slice.

**When to use:** Side-by-side comparison view.

**Example:**

```typescript
// src/store/comparisonStore.ts
import { create } from 'zustand'
import type { UIState } from './uiStore'

export interface ConfigSnapshot {
  id: string // UUID
  timestamp: number
  label: string // User-editable label
  config: Partial<UIState> // Snapshot of relevant state
  results?: CalculationResults // Cached calculation results
}

interface ComparisonState {
  snapshots: ConfigSnapshot[]
  maxSnapshots: number

  // Actions
  addSnapshot: (snapshot: Omit<ConfigSnapshot, 'id' | 'timestamp'>) => void
  removeSnapshot: (id: string) => void
  updateLabel: (id: string, label: string) => void
  clearAll: () => void
}

export const useComparisonStore = create<ComparisonState>((set) => ({
  snapshots: [],
  maxSnapshots: 3,

  addSnapshot: (snapshot) =>
    set((state) => {
      if (state.snapshots.length >= state.maxSnapshots) {
        // Remove oldest snapshot
        return {
          snapshots: [
            ...state.snapshots.slice(1),
            {
              ...snapshot,
              id: crypto.randomUUID(),
              timestamp: Date.now(),
            },
          ],
        }
      }
      return {
        snapshots: [
          ...state.snapshots,
          {
            ...snapshot,
            id: crypto.randomUUID(),
            timestamp: Date.now(),
          },
        ],
      }
    }),

  removeSnapshot: (id) =>
    set((state) => ({
      snapshots: state.snapshots.filter((s) => s.id !== id),
    })),

  updateLabel: (id, label) =>
    set((state) => ({
      snapshots: state.snapshots.map((s) => (s.id === id ? { ...s, label } : s)),
    })),

  clearAll: () => set({ snapshots: [] }),
}))
```

**Usage in component:**

```typescript
// src/components/comparison/ComparisonView.tsx
export function ComparisonView() {
  const { snapshots, addSnapshot, removeSnapshot } = useComparisonStore()
  const currentConfig = useUIStore() // Get current configuration

  const handleAddCurrent = () => {
    addSnapshot({
      label: `Config ${snapshots.length + 1}`,
      config: {
        selectedModel: currentConfig.selectedModel,
        selectedGPU: currentConfig.selectedGPU,
        quantization: currentConfig.quantization,
        // ... other relevant fields
      },
      results: currentConfig.results, // Cache calculation results
    })
  }

  return (
    <div className="comparison-view">
      <ComparisonControls onAddCurrent={handleAddCurrent} />
      <div className="comparison-grid grid grid-cols-1 lg:grid-cols-3 gap-4">
        {snapshots.map((snapshot) => (
          <ComparisonColumn
            key={snapshot.id}
            snapshot={snapshot}
            onRemove={() => removeSnapshot(snapshot.id)}
          />
        ))}
      </div>
    </div>
  )
}
```

### Pattern 3: Highlight Differences in Comparison View

**What:** Visually emphasize fields that differ between configurations.

**When to use:** Comparison column rendering.

**Example:**

```typescript
// src/components/comparison/ComparisonColumn.tsx
interface DiffStatus {
  quantization: boolean // True if different from other configs
  gpuVRAM: boolean
  numGPUs: boolean
  // ... other fields
}

export function ComparisonColumn({ snapshot, allSnapshots }: Props) {
  // Calculate what's different
  const diff = useMemo(() => {
    const others = allSnapshots.filter((s) => s.id !== snapshot.id)
    return {
      quantization: others.some((s) => s.config.quantization !== snapshot.config.quantization),
      gpuVRAM: others.some((s) => s.config.selectedGPU?.vram !== snapshot.config.selectedGPU?.vram),
      numGPUs: others.some((s) => s.config.numGPUs !== snapshot.config.numGPUs),
      // ... compare other fields
    }
  }, [snapshot, allSnapshots])

  return (
    <div className="comparison-column">
      <h3>{snapshot.label}</h3>

      {/* Highlight differences with visual indicator */}
      <Field
        label="Quantization"
        value={snapshot.config.quantization}
        isDifferent={diff.quantization}
      />
      <Field
        label="GPU VRAM"
        value={`${snapshot.config.selectedGPU?.vram}GB`}
        isDifferent={diff.gpuVRAM}
      />
      <Field label="Num GPUs" value={snapshot.config.numGPUs} isDifferent={diff.numGPUs} />

      {/* Reuse existing output components */}
      <VRAMBreakdown results={snapshot.results} compact />
      <PerformanceMetrics results={snapshot.results} compact />
    </div>
  )
}

function Field({ label, value, isDifferent }: FieldProps) {
  return (
    <div
      className={cn(
        'field',
        isDifferent && 'bg-yellow-100 dark:bg-yellow-900/20 border-l-2 border-yellow-500',
      )}
    >
      <span className="label">{label}</span>
      <span className="value">{value}</span>
      {isDifferent && <DiffIcon className="ml-2 text-yellow-600" />}
    </div>
  )
}
```

### Anti-Patterns to Avoid

- **Don't store full model/GPU objects in URL**: Too large. Store IDs only, hydrate on load.
- **Don't skip compression**: Uncompressed URLs exceed browser limits. Always use lz-string.
- **Don't persist comparison snapshots to URL**: Snapshots are transient UI state. Keep in memory or localStorage, not URL.
- **Don't deep-clone calculation results**: Snapshots store cached results. Don't recalculate on render.
- **Don't update URL on every keystroke**: Debounce URL updates (300ms) to avoid performance issues and browser history pollution.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| URL compression | Custom base64 + compression | **lz-string** | Handles URL-safe encoding, compression, and edge cases (unicode, special chars). Battle-tested in production. |
| State serialization | Manual JSON.stringify with type handling | **Zod + JSON** (or superjson if needed) | Zod validates structure. superjson handles Date/Map/Set/Decimal if needed. Don't write custom serializers. |
| Hash storage | `window.location.hash = ...` directly | **Zustand persist middleware** with custom StateStorage | Handles race conditions, initialization order, SSR compatibility. Custom hash code is bug-prone. |
| Object comparison | Nested loops comparing fields | **Shallow comparison** (for simple cases) or **deep-object-diff** (if complex) | Comparison logic is subtle (NaN, undefined, null edge cases). Use library. |
| URL length validation | Counting characters | **Test with actual browsers** | Limits vary by browser. Test with real data, add buffer (stay under 1800 chars to be safe). |

**Key insight:** URL state management has many edge cases (initialization timing, SSR, browser history, URL length limits, encoding issues). Use proven libraries and patterns from Zustand ecosystem rather than custom solutions.

## Common Pitfalls

### Pitfall 1: URL Too Long - Browser Limits

**What goes wrong:** Browsers have varying URL length limits. Safest limit is ~2000 characters. Uncompressed config state can exceed 5000+ characters, causing URLs to break or get truncated.

**Why it happens:** Configuration state includes model architecture details, GPU specs, and multiple parameters. JSON serialization without compression is verbose.

**How to avoid:**
- **Always compress** with lz-string (`compressToEncodedURIComponent`)
- **Partialize aggressively**: Only serialize essential fields for reconstruction (IDs, not full objects)
- **Test with max data**: Test URL generation with largest model (e.g., Llama 3 405B) and maximum parameters
- **Add size check**: Log warning if compressed URL > 1800 chars

**Warning signs:**
- URL looks truncated when copied
- State doesn't restore correctly after reload
- Browser address bar shows incomplete URL

### Pitfall 2: Stale Data After URL Load

**What goes wrong:** User shares URL from yesterday. Today you updated models.json with new GPUs. URL contains `modelId: "llama-3-70b"` but that model was renamed or removed. State fails to hydrate.

**Why it happens:** URL state references data by ID. If referenced data changes or is removed, hydration fails silently.

**How to avoid:**
- **Graceful fallback**: If modelId not found, set `selectedModel: null` and show info toast ("Model from URL not found")
- **Version URLs** (optional): Include schema version in URL state. Future-proof against breaking changes.
- **Validate on hydrate**: Use Zod schema to validate structure. Catch parse errors and reset to defaults.

**Warning signs:**
- User reports "my shared link doesn't work"
- State partially loads (some fields missing)
- Console errors during hydration

### Pitfall 3: Infinite Update Loop

**What goes wrong:** Setting `location.hash` triggers `hashchange` event, which reads hash and updates store, which triggers `setItem` in hashStorage, which sets `location.hash` again. Infinite loop.

**Why it happens:** Bidirectional sync (store → URL and URL → store) without cycle detection.

**How to avoid:**
- **Check before update**: In `setItem`, compare new hash with current hash. Only update if different.
- **Debounce hash updates**: Debounce `setItem` by 300ms to batch rapid changes.
- **Use `replaceState` not `pushState`**: Avoid polluting browser history with every parameter change.

**Warning signs:**
- Browser freezes
- URL constantly changes
- Console shows rapid hash updates

### Pitfall 4: Dark Mode Persisting to URL

**What goes wrong:** User shares URL with dark mode enabled. Recipient opens link, sees dark mode even though they prefer light mode.

**Why it happens:** Mixing UI preferences (dark mode, sidebar collapsed) with shareable configuration state.

**How to avoid:**
- **Two persistence strategies**: localStorage for UI preferences (dark mode), URL hash for configuration state (model, GPU, quantization)
- **Separate stores** (optional): `uiStore` for preferences, `configStore` for shareable config
- **Document partialize**: Comment which fields go to URL vs localStorage

**Warning signs:**
- User complains "your link forced dark mode on me"
- Theme preference doesn't persist across sessions

### Pitfall 5: Comparison State Bloat

**What goes wrong:** User adds 20 configurations to comparison, each caching full calculation results. Memory usage explodes. UI becomes sluggish.

**Why it happens:** No limit on snapshot count. Each snapshot stores large result objects.

**How to avoid:**
- **Enforce max snapshots**: 3 maximum (based on requirement). Remove oldest when adding 4th.
- **Cache wisely**: Store minimal results needed for comparison (VRAM total, fit status). Don't cache full breakdown.
- **Clear on navigate**: Reset comparison state when user leaves comparison view.
- **Add warning**: Show UI message "Comparison limited to 3 configurations" when limit reached.

**Warning signs:**
- Comparison view slow to render
- High memory usage in DevTools
- Users confused why they can't add more configs

### Pitfall 6: Decimal.js in URL State

**What goes wrong:** Configuration includes Decimal values (precise VRAM calculations). `JSON.stringify(Decimal)` produces object, not number. URL state corrupted.

**Why it happens:** Decimal.js objects aren't JSON-serializable by default.

**How to avoid:**
- **Convert to number for URL**: `decimalValue.toNumber()` before serializing. Precision loss acceptable for URL state.
- **Or use superjson**: Register Decimal as custom type. Adds ~8kb to bundle.
- **Preferred approach**: Store numeric inputs in URL (sequenceLength: 4096), recalculate on hydrate. Don't store calculation results in URL.

**Warning signs:**
- URL contains `{"value":"4.5","exp":0}` instead of `4.5`
- Hydration fails with "expected number, got object"

## Code Examples

Verified patterns from research and official Zustand docs:

### Complete Hash Storage Implementation

```typescript
// src/store/uiStore.ts - Full example with compression and validation
import { create } from 'zustand'
import { persist, StateStorage, createJSONStorage } from 'zustand/middleware'
import { compress, decompress } from 'lz-string'
import { z } from 'zod'

// 1. Define URL state schema
const URLStateSchema = z.object({
  modelId: z.string().optional(),
  gpuId: z.string().optional(),
  quantization: z.enum(['fp32', 'fp16', 'int8', 'int4', 'nf4']),
  sequenceLength: z.number().min(128).max(128000),
  batchSize: z.number().min(1).max(256),
  kvQuantization: z.enum(['fp32', 'fp16', 'int8', 'int4']),
  numGPUs: z.number().min(1).max(8),
  shardingStrategy: z.enum(['tensor-parallel', 'pipeline-parallel']),
  // ... other fields
})

// 2. Create hash storage with compression
const hashStorage: StateStorage = {
  getItem: (key: string): string | null => {
    try {
      const hash = window.location.hash.slice(1) // Remove '#'
      if (!hash) return null

      // Decompress and parse
      const json = decompress(hash)
      if (!json) return null

      // Validate with Zod
      const state = URLStateSchema.parse(JSON.parse(json))
      return JSON.stringify(state)
    } catch (error) {
      console.warn('Failed to parse URL state:', error)
      return null // Graceful failure
    }
  },

  setItem: (key: string, value: string): void => {
    try {
      const state = JSON.parse(value)
      const json = JSON.stringify(state)
      const compressed = compress(json)

      // Only update if different (avoid infinite loop)
      if (window.location.hash.slice(1) !== compressed) {
        // Use replaceState to avoid polluting history
        window.history.replaceState(null, '', `#${compressed}`)
      }

      // Log warning if URL too long
      if (compressed.length > 1800) {
        console.warn(`URL state is ${compressed.length} chars - may exceed browser limits`)
      }
    } catch (error) {
      console.error('Failed to serialize URL state:', error)
    }
  },

  removeItem: (key: string): void => {
    window.history.replaceState(null, '', window.location.pathname)
  },
}

// 3. Use in Zustand store
export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      // ... state and actions
    }),
    {
      name: 'llmvram-config',
      storage: createJSONStorage(() => hashStorage),
      partialize: (state) => ({
        modelId: state.selectedModel?.id,
        gpuId: state.selectedGPU?.id,
        quantization: state.quantization,
        sequenceLength: state.sequenceLength,
        batchSize: state.batchSize,
        kvQuantization: state.kvQuantization,
        numGPUs: state.numGPUs,
        shardingStrategy: state.shardingStrategy,
        // Only shareable config, not UI preferences
      }),
      // Hydrate IDs back to full objects
      onRehydrateStorage: () => (state, error) => {
        if (error) {
          console.error('Hydration error:', error)
          return
        }
        if (state) {
          // Look up full objects by ID
          if (state.modelId) {
            state.selectedModel = findModelById(state.modelId)
          }
          if (state.gpuId) {
            state.selectedGPU = findGPUById(state.gpuId)
          }
        }
      },
    },
  ),
)
```

### Debounced URL Updates

```typescript
// src/hooks/useURLSync.ts - Debounce URL updates to avoid history pollution
import { useEffect, useRef } from 'react'
import { useUIStore } from '@/store/uiStore'

export function useURLSync() {
  const timeoutRef = useRef<number>()
  const store = useUIStore()

  useEffect(() => {
    // Debounce URL updates
    clearTimeout(timeoutRef.current)
    timeoutRef.current = window.setTimeout(() => {
      // Zustand persist middleware handles actual URL update
      // This just batches rapid changes
    }, 300)

    return () => clearTimeout(timeoutRef.current)
  }, [
    store.quantization,
    store.sequenceLength,
    store.batchSize,
    store.numGPUs,
    // ... other fields that trigger URL update
  ])
}
```

### Comparison Snapshot Component

```typescript
// src/components/comparison/ComparisonView.tsx
import { useComparisonStore } from '@/store/comparisonStore'
import { useUIStore } from '@/store/uiStore'
import { VRAMBreakdown } from '@/components/outputs/VRAMBreakdown'

export function ComparisonView() {
  const { snapshots, addSnapshot, removeSnapshot, clearAll } = useComparisonStore()
  const currentConfig = useUIStore()

  const canAddMore = snapshots.length < 3

  const handleSaveCurrent = () => {
    addSnapshot({
      label: `Config ${snapshots.length + 1}`,
      config: {
        selectedModel: currentConfig.selectedModel,
        selectedGPU: currentConfig.selectedGPU,
        quantization: currentConfig.quantization,
        sequenceLength: currentConfig.sequenceLength,
        batchSize: currentConfig.batchSize,
        numGPUs: currentConfig.numGPUs,
      },
      results: currentConfig.calculationResults,
    })
  }

  return (
    <div className="comparison-view">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">Compare Configurations</h2>
        <div className="flex gap-2">
          <button
            onClick={handleSaveCurrent}
            disabled={!canAddMore}
            className="btn-primary"
          >
            {canAddMore ? 'Add Current Config' : 'Max 3 Configs'}
          </button>
          {snapshots.length > 0 && (
            <button onClick={clearAll} className="btn-secondary">
              Clear All
            </button>
          )}
        </div>
      </div>

      {snapshots.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {snapshots.map((snapshot) => (
            <ComparisonColumn
              key={snapshot.id}
              snapshot={snapshot}
              allSnapshots={snapshots}
              onRemove={() => removeSnapshot(snapshot.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Query params (`?model=x&gpu=y`) | URL hash (`#compressed-state`) | 2020+ | Hash doesn't trigger page reload, better for SPAs. Works offline. |
| Manual `JSON.stringify` + base64 | lz-string compression | 2019+ | 60-70% size reduction. Critical for complex state. |
| Redux with manual hash sync | Zustand persist middleware | 2021+ | Built-in storage abstraction. Less boilerplate. |
| Deep cloning for snapshots | Structural sharing (Zustand) | 2020+ | Better memory efficiency. Faster updates. |
| Custom diff algorithms | Shallow comparison + visual hints | Current | Simple configs don't need deep diffing. Visual cues suffice. |

**Deprecated/outdated:**

- **Query string for SPA state**: Query params reload page. Use hash for pure client-side state.
- **Uncompressed JSON in URL**: Quickly exceeds 2000 char limit. Always compress.
- **Redux for URL state**: Too much boilerplate. Zustand persist is 10x simpler.
- **Storing full calculation results in URL**: Bloats URL. Store inputs only, recalculate on load.

## Open Questions

1. **Should comparison snapshots persist across sessions?**
   - What we know: Snapshots are stored in memory. Lost on page refresh.
   - What's unclear: Do users want to save comparison sets for later?
   - Recommendation: Start with transient (memory only). Add localStorage persistence if users request it. Don't add to URL (too large).

2. **Should we support "share comparison" URLs?**
   - What we know: VIZ-04 shares single config. VIZ-05 compares multiple configs.
   - What's unclear: Should URL encode multiple configs for sharing comparison view?
   - Recommendation: Phase 5 supports single config URL. If users request "share my comparison", add in future phase. Requires more complex URL encoding (array of configs).

3. **What if user has non-standard model/GPU?**
   - What we know: User can enter custom model parameters (not in models.json).
   - What's unclear: How to serialize custom model in URL? Can't use ID reference.
   - Recommendation: For custom models, serialize full parameters in URL (increases size). Test URL length with custom model + compression.

4. **Should dark mode be in URL (despite recommendation not to)?**
   - What we know: Best practice is localStorage for UI preferences.
   - What's unclear: Does "raidy pattern" include dark mode in URL?
   - Recommendation: Check raidy implementation. If they include it, match for consistency. Otherwise, keep in localStorage.

## Sources

### Primary (HIGH confidence)

- [Zustand - Connect to state with URL hash](https://zustand.docs.pmnd.rs/guides/connect-to-state-with-url-hash) - Official guide for hashStorage pattern
- [Zustand - Persisting store data](https://zustand.docs.pmnd.rs/integrations/persisting-store-data) - StateStorage interface, custom storage, superjson
- [lz-string npm package](https://pieroxy.net/blog/pages/lz-string/index.html) - Compression algorithm documentation
- [GitHub - pmndrs/zustand](https://github.com/pmndrs/zustand) - v5.0.10 release notes (persist bug fixes)

### Secondary (MEDIUM confidence)

- [Store Zustand State into URL, share state across the web](https://www.saybackend.com/blog/zustand-url-state/) - Implementation examples, verified with official docs
- [SuperJSON - JSON on steroids](https://simonknott.de/articles/SuperJSON.html) - Decimal.js serialization patterns
- [React Diff Viewer - DHiWise](https://www.dhiwise.com/post/exploring-the-power-of-react-diff-viewer-comprehensive-guide) - Comparison UI patterns
- [GitHub - gera2ld/react-deep-diff](https://github.com/gera2ld/react-deep-diff) - Object comparison component
- [State Management in React Applications Through URL Hashes](https://peterkellner.net/2023/09/16/State%20Management-in-React-Applications-Through-URL-Hashes/) - Pitfalls and best practices

### Tertiary (LOW confidence - needs validation)

- [Advanced React state management using URL parameters - LogRocket](https://blog.logrocket.com/advanced-react-state-management-using-url-parameters/) - Pattern examples (not Zustand-specific)
- [Frontend State Management Best Practices](https://lorenzogm.com/blog/frontend-state-management-best-practices) - General patterns (not LLM calculator specific)

## Metadata

**Confidence breakdown:**

- **URL hash persistence**: HIGH - Official Zustand docs provide exact pattern. lz-string is proven library.
- **Compression strategy**: HIGH - lz-string widely used, reduces size by 60-70% (verified by multiple sources).
- **Comparison architecture**: MEDIUM-HIGH - Pattern is straightforward (snapshot array + shallow comparison), but no LLM-calculator-specific examples found.
- **Serialization of complex types**: MEDIUM - superjson pattern documented, but Decimal.js + URL state interaction needs testing.
- **Browser limits**: MEDIUM - 2000 char limit is conservative (some browsers support more), but safe threshold verified by multiple sources.

**Research date:** 2026-02-09
**Valid until:** ~30 days (stable domain, but Zustand may release updates)

**Recommended validation before planning:**

1. Upgrade Zustand from 5.0.9 to 5.0.10 (persist bug fixes released Jan 2026)
2. Test URL length with largest model (Llama 3 405B) + maximum parameters + compression
3. Verify if "raidy" includes dark mode in URL or localStorage (match their pattern)
4. Test custom model serialization (full parameters in URL) + compression size
5. Confirm whether comparison snapshots should persist (localStorage) or be transient (memory only)
