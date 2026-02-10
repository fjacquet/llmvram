# Phase 07: Training State & Basic UI - Research

**Researched:** 2026-02-10
**Domain:** React state management, UI components, mode toggling
**Confidence:** HIGH

## Summary

Phase 7 requires implementing a mode toggle between Inference and Fine-tuning, plus basic training configuration UI (method, optimizer, precision). The existing codebase already uses Zustand with URL hash persistence via lz-string, and Headless UI components styled with Tailwind CSS v4.

The research confirms that the **Zustand slice pattern** (as used by sister project raidy) is the best approach for adding training state alongside existing inference state. This pattern keeps inference and training concerns separated while allowing them to share a single store with URL persistence. Headless UI's **Switch** component provides accessible toggle switches, and **RadioGroup** handles mutually-exclusive selections (training method, optimizer, precision).

Key insight: The existing `urlSerializer.ts` approach (custom serialization) should be replaced with raidy's pattern of using Zustand's built-in `persist` middleware with a custom `StateStorage` implementation. This eliminates the need for manual sync hooks and provides automatic bidirectional sync between store and URL hash.

**Primary recommendation:** Adopt the Zustand slice pattern from raidy (separate slices for inference and training state), replace current URL sync approach with `persist` middleware + custom `StateStorage`, and use Headless UI Switch/RadioGroup components for the training configuration UI.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Zustand | 5.0.9 | State management | Already in use; slice pattern proven in raidy project |
| Headless UI | 2.2.9 | Accessible UI components | Already in use; Switch and RadioGroup perfect for mode toggling |
| Zod | 3.25.76 | Schema validation | Already in use; validates URL state on hydration |
| lz-string | 1.5.0 | URL compression | Already in use; keeps shareable URLs under 2KB |
| Tailwind CSS | 4.1.18 | Styling | Already in use; v4's peer variants ideal for toggle styling |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| zustand/middleware | (built-in) | Persist middleware | URL hash persistence with custom StateStorage |
| react-hook-form | (optional) | Complex forms | NOT needed for Phase 7 - simple state is sufficient |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Zustand slices | Context API | Context causes unnecessary re-renders; Zustand more performant |
| Headless UI Switch | Custom toggle | Headless UI handles accessibility (ARIA, keyboard nav) automatically |
| URL hash | localStorage | URL hash enables shareable links; localStorage is per-device only |
| persist middleware | Manual sync | Manual sync (current approach) requires custom debouncing and hydration logic |

**Installation:**
```bash
# All dependencies already installed
# No new packages required
```

## Architecture Patterns

### Recommended Project Structure (After Phase 7)
```
src/
├── store/
│   ├── slices/
│   │   ├── inferenceSlice.ts    # Move current inference state here
│   │   ├── trainingSlice.ts     # New: training state & actions
│   │   └── index.ts             # Export all slices
│   ├── uiStore.ts               # REFACTOR: Use slice pattern + persist middleware
│   ├── urlStorage.ts            # NEW: Custom StateStorage (from raidy pattern)
│   └── urlSerializer.ts         # DEPRECATED: Remove after refactor
├── components/
│   ├── inputs/
│   │   ├── ModeToggle.tsx       # NEW: Inference/Training switch
│   │   ├── TrainingMethodPicker.tsx  # NEW: Full/LoRA/QLoRA radio group
│   │   ├── OptimizerPicker.tsx      # NEW: AdamW/SGD/etc radio group
│   │   ├── PrecisionPicker.tsx      # NEW: FP32/FP16/BF16 radio group
│   │   └── TrainingPanel.tsx        # NEW: Container for training inputs
│   └── outputs/
│       └── (no changes for Phase 7)
└── hooks/
    └── useURLSync.ts            # DEPRECATED: Remove after refactor
```

### Pattern 1: Zustand Slice Pattern (Inference + Training)

**What:** Separate state concerns into independent slices that are combined into a single store.

**When to use:** When adding new state domains (like training) to an existing store (inference).

**Example:**
```typescript
// Source: raidy/src/store/configStore.ts + Context7 Zustand docs
// store/slices/inferenceSlice.ts
import type { StateCreator } from 'zustand'
import type { InferenceState, InferenceActions } from '@/types'

export interface InferenceSlice extends InferenceState, InferenceActions {}

export const createInferenceSlice: StateCreator<
  InferenceSlice & TrainingSlice,
  [],
  [],
  InferenceSlice
> = (set) => ({
  // State (existing inference state)
  selectedModel: null,
  selectedGPU: null,
  quantization: 'fp16',
  sequenceLength: 4096,
  batchSize: 1,
  kvQuantization: 'fp16',
  numGPUs: 1,
  shardingStrategy: 'tensor-parallel',
  offloadingEnabled: false,
  // ... other inference state

  // Actions (existing inference actions)
  setSelectedModel: (model) => set({ selectedModel: model }),
  setQuantization: (quantization) => set({ quantization }),
  // ... other inference actions
})

// store/slices/trainingSlice.ts
import type { StateCreator } from 'zustand'
import type { TrainingState, TrainingActions } from '@/types'

export interface TrainingSlice extends TrainingState, TrainingActions {}

export const createTrainingSlice: StateCreator<
  InferenceSlice & TrainingSlice,
  [],
  [],
  TrainingSlice
> = (set) => ({
  // State (new training state)
  mode: 'inference',  // 'inference' | 'training'
  trainingMethod: 'lora',
  optimizer: 'adamw',
  trainingPrecision: 'bf16',
  loraRank: 16,
  loraAlpha: 32,
  targetModulesPercent: 30,

  // Actions (new training actions)
  setMode: (mode) => set({ mode }),
  setTrainingMethod: (method) => set({ trainingMethod: method }),
  setOptimizer: (optimizer) => set({ optimizer }),
  setTrainingPrecision: (precision) => set({ trainingPrecision: precision }),
  setLoraRank: (rank) => set({ loraRank: rank }),
  setLoraAlpha: (alpha) => set({ loraAlpha: alpha }),
  setTargetModulesPercent: (percent) => set({ targetModulesPercent: percent }),
})

// store/uiStore.ts (refactored)
import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'
import { createInferenceSlice, createTrainingSlice } from './slices'
import { urlHashStorage } from './urlStorage'
import type { InferenceSlice, TrainingSlice } from './slices'

export type UIStore = InferenceSlice & TrainingSlice

export const useUIStore = create<UIStore>()(
  persist(
    (...args) => ({
      ...createInferenceSlice(...args),
      ...createTrainingSlice(...args),
    }),
    {
      name: 'llmvram',
      storage: createJSONStorage(() => urlHashStorage),
      version: 1,
      partialize: (state) => {
        // Only persist state values, not actions
        // Mode-specific state (only persist relevant fields based on mode)
        return {
          mode: state.mode,
          // Inference state (always persist for mode switching)
          selectedModel: state.selectedModel ? { id: state.selectedModel.id } : null,
          selectedGPU: state.selectedGPU ? { id: state.selectedGPU.id } : null,
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
          // Training state (only persist if mode is 'training')
          ...(state.mode === 'training' ? {
            trainingMethod: state.trainingMethod,
            optimizer: state.optimizer,
            trainingPrecision: state.trainingPrecision,
            loraRank: state.loraRank,
            loraAlpha: state.loraAlpha,
            targetModulesPercent: state.targetModulesPercent,
          } : {}),
        }
      },
    },
  ),
)
```

### Pattern 2: Custom StateStorage for URL Hash Persistence

**What:** Implement Zustand's `StateStorage` interface to sync state to URL hash with lz-string compression and Zod validation.

**When to use:** When you need shareable URLs without a backend, with validation on hydration.

**Example:**
```typescript
// Source: raidy/src/store/urlStorage.ts + Zustand persist middleware docs
// store/urlStorage.ts
import { compressToEncodedURIComponent, decompressFromEncodedURIComponent } from 'lz-string'
import { toast } from 'sonner'
import type { StateStorage } from 'zustand/middleware'
import { validateUrlState } from '@/utils/schemas'

export const urlHashStorage: StateStorage = {
  getItem: (key: string): string | null => {
    if (typeof window === 'undefined') return null

    const hash = window.location.hash.slice(1)
    if (!hash) return null

    try {
      const searchParams = new URLSearchParams(hash)
      const compressed = searchParams.get(key)
      if (!compressed) return null

      const decompressed = decompressFromEncodedURIComponent(compressed)
      if (!decompressed) return null

      // Parse and validate with Zod schema
      const parsed = JSON.parse(decompressed)
      const validated = validateUrlState(parsed)

      if (!validated) {
        console.error('Configuration link is invalid or corrupted')
        toast.error('Invalid configuration link', {
          description: 'The shared link is invalid. Using defaults.',
          duration: 5000,
        })
        return null
      }

      return JSON.stringify(validated)
    } catch (error) {
      console.error('Failed to parse URL hash state:', error)
      toast.error('Invalid configuration link', {
        description: 'Unable to load from URL. Using defaults.',
        duration: 5000,
      })
      return null
    }
  },

  setItem: (key: string, newValue: string): void => {
    if (typeof window === 'undefined') return

    try {
      const compressed = compressToEncodedURIComponent(newValue)
      const searchParams = new URLSearchParams(window.location.hash.slice(1))
      searchParams.set(key, compressed)

      // Update URL without triggering navigation
      window.history.replaceState(
        null,
        '',
        `${window.location.pathname}${window.location.search}#${searchParams.toString()}`,
      )
    } catch (error) {
      console.warn('Failed to persist state to URL:', error)
    }
  },

  removeItem: (key: string): void => {
    if (typeof window === 'undefined') return

    const searchParams = new URLSearchParams(window.location.hash.slice(1))
    searchParams.delete(key)

    const newHash = searchParams.toString()
    window.history.replaceState(
      null,
      '',
      newHash
        ? `${window.location.pathname}${window.location.search}#${newHash}`
        : `${window.location.pathname}${window.location.search}`,
    )
  },
}
```

### Pattern 3: Mode Toggle with Headless UI Switch

**What:** Accessible toggle switch for Inference/Training mode with immediate visual feedback.

**When to use:** For binary mode switches that take effect immediately.

**Example:**
```typescript
// Source: Context7 Headless UI docs + Tailwind toggle best practices
// components/inputs/ModeToggle.tsx
import { Switch } from '@headlessui/react'
import { useUIStore } from '@store/uiStore'

export function ModeToggle() {
  const mode = useUIStore((s) => s.mode)
  const setMode = useUIStore((s) => s.setMode)

  const isTraining = mode === 'training'

  return (
    <Switch.Group>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <Switch.Label className="text-sm font-medium text-gray-900 dark:text-white">
            {isTraining ? 'Fine-tuning Mode' : 'Inference Mode'}
          </Switch.Label>
          <Switch.Description className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {isTraining
              ? 'Calculate VRAM for training/fine-tuning'
              : 'Calculate VRAM for model inference'
            }
          </Switch.Description>
        </div>

        <Switch
          checked={isTraining}
          onChange={(checked) => setMode(checked ? 'training' : 'inference')}
          className={`${
            isTraining ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          } relative inline-flex h-6 w-11 items-center rounded-full transition-colors
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2`}
        >
          <span
            className={`${
              isTraining ? 'translate-x-6' : 'translate-x-1'
            } inline-block h-4 w-4 transform rounded-full bg-white transition-transform
            duration-200 ease-in-out`}
          />
        </Switch>
      </div>
    </Switch.Group>
  )
}
```

### Pattern 4: RadioGroup for Mutually-Exclusive Options

**What:** Accessible radio group for selecting one option from a list (training method, optimizer, precision).

**When to use:** For selections where only one value can be active at a time.

**Example:**
```typescript
// Source: Context7 Headless UI docs
// components/inputs/TrainingMethodPicker.tsx
import { RadioGroup } from '@headlessui/react'
import { CheckIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'
import type { FineTuningMethod } from '@engines/types'

const METHODS: Array<{ value: FineTuningMethod; label: string; description: string }> = [
  { value: 'full', label: 'Full Fine-tuning', description: 'Train all parameters (highest VRAM)' },
  { value: 'lora', label: 'LoRA', description: 'Train adapters only (medium VRAM)' },
  { value: 'qlora', label: 'QLoRA', description: '4-bit base + adapters (lowest VRAM)' },
]

export function TrainingMethodPicker() {
  const method = useUIStore((s) => s.trainingMethod)
  const setMethod = useUIStore((s) => s.setTrainingMethod)

  return (
    <div className="space-y-2">
      <RadioGroup value={method} onChange={setMethod}>
        <RadioGroup.Label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Fine-tuning Method
        </RadioGroup.Label>
        <div className="mt-2 space-y-2">
          {METHODS.map((option) => (
            <RadioGroup.Option
              key={option.value}
              value={option.value}
              className={({ active, checked }) =>
                `${checked ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-600' : 'border-gray-300 dark:border-gray-600'}
                ${active ? 'ring-2 ring-blue-500 ring-offset-2' : ''}
                relative flex cursor-pointer rounded-lg border px-4 py-3 focus:outline-none`
              }
            >
              {({ checked }) => (
                <div className="flex w-full items-center justify-between">
                  <div className="flex-1">
                    <RadioGroup.Label
                      as="p"
                      className={`text-sm font-medium ${
                        checked ? 'text-blue-900 dark:text-blue-100' : 'text-gray-900 dark:text-white'
                      }`}
                    >
                      {option.label}
                    </RadioGroup.Label>
                    <RadioGroup.Description
                      as="span"
                      className={`text-xs ${
                        checked ? 'text-blue-700 dark:text-blue-300' : 'text-gray-500 dark:text-gray-400'
                      }`}
                    >
                      {option.description}
                    </RadioGroup.Description>
                  </div>
                  {checked && (
                    <div className="shrink-0 text-blue-600 dark:text-blue-400">
                      <CheckIcon className="h-5 w-5" />
                    </div>
                  )}
                </div>
              )}
            </RadioGroup.Option>
          ))}
        </div>
      </RadioGroup>
    </div>
  )
}
```

### Pattern 5: Conditional Rendering Based on Mode

**What:** Show/hide UI panels based on the current mode (inference vs training).

**When to use:** When different modes require different input controls.

**Example:**
```typescript
// Source: React docs + React design patterns 2026
// components/layout/InputPanel.tsx
import { ModeToggle } from '@components/inputs/ModeToggle'
import { TrainingPanel } from '@components/inputs/TrainingPanel'
import { useUIStore } from '@store/uiStore'

export function InputPanel() {
  const mode = useUIStore((s) => s.mode)

  return (
    <div className="space-y-6">
      {/* Mode toggle always visible */}
      <ModeToggle />

      {/* Shared inputs (model, GPU) - always visible */}
      <ModelSelector />
      <GPUSelector />

      {/* Conditional panels based on mode */}
      {mode === 'inference' ? (
        <InferencePanel />
      ) : (
        <TrainingPanel />
      )}
    </div>
  )
}

// components/inputs/TrainingPanel.tsx
export function TrainingPanel() {
  return (
    <div className="space-y-4 border-t border-gray-200 dark:border-gray-700 pt-4">
      <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
        Training Configuration
      </h3>
      <TrainingMethodPicker />
      <OptimizerPicker />
      <PrecisionPicker />
      {/* LoRA-specific inputs shown conditionally */}
    </div>
  )
}
```

### Anti-Patterns to Avoid

- **Don't create separate stores for inference and training:** Use slices within a single store to avoid sync issues and enable shared URL state.
- **Don't use nested conditional renders in JSX:** Use early returns or separate components for mode-specific UI (see Pattern 5).
- **Don't skip Zod validation on URL hydration:** Always validate deserialized state to prevent crashes from malformed URLs.
- **Don't use custom debouncing for URL updates:** The `persist` middleware handles updates efficiently; manual debouncing adds complexity.
- **Don't use toggles for actions requiring confirmation:** Toggles are for immediate, reversible changes only (mode switching is reversible, so it's appropriate).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Accessible toggle switches | Custom checkbox + styling | Headless UI Switch | Handles ARIA roles, keyboard nav (Space/Enter), focus management automatically |
| Radio button groups | Custom state + click handlers | Headless UI RadioGroup | Manages keyboard navigation (arrow keys), ARIA roles, focus management |
| URL state persistence | Manual localStorage + URL sync | Zustand persist + custom StateStorage | Handles hydration, serialization, race conditions, SSR compatibility |
| State serialization/compression | Custom JSON.stringify + base64 | lz-string | Achieves 60-80% compression; handles Unicode correctly; battle-tested |
| Form validation | Manual validation logic | Zod schemas | Type inference, composable validators, clear error messages |

**Key insight:** URL state management is deceptively complex. Manual approaches miss edge cases like:
- Race conditions between URL changes and store updates
- SSR hydration mismatches
- Browser back/forward button handling
- URL length limits (IE11: 2083 chars, modern: ~2000 recommended)
- Invalid/corrupted URLs causing crashes

The raidy pattern (persist middleware + custom StateStorage) handles all these cases correctly.

## Common Pitfalls

### Pitfall 1: Forgetting to Update URL Schema When Adding State

**What goes wrong:** New state fields added to slices don't persist to URL, breaking shareable links.

**Why it happens:** Adding state to slices is easy, but forgetting to update the `partialize` function in the persist config.

**How to avoid:**
1. When adding new state fields to a slice, immediately update the `partialize` function
2. Add a test that validates the URL schema matches the persisted state shape
3. Consider using a Zod schema that's shared between slice types and the URL validator

**Warning signs:**
- New fields reset to defaults when sharing URLs
- URL doesn't update when changing certain fields

### Pitfall 2: Mode-Specific State Polluting URL for Wrong Mode

**What goes wrong:** Training state persists in URL even when in inference mode, making URLs longer than necessary.

**Why it happens:** The `partialize` function includes all state regardless of mode.

**How to avoid:** Conditionally include training state in `partialize` based on `state.mode` (see Pattern 1 example).

**Warning signs:**
- URLs are 1500+ characters even for simple inference configs
- URL includes `trainingMethod`, `optimizer`, etc. when `mode=inference`

### Pitfall 3: Toggle Switch Without Visual Feedback During Transition

**What goes wrong:** Users click the toggle but don't see immediate feedback, leading to double-clicks.

**Why it happens:** Missing transition classes on the switch slider, or state updates are slow.

**How to avoid:**
- Always include `transition-transform duration-200 ease-in-out` on the slider element
- Keep state updates synchronous (no async operations in `setMode`)
- Use Tailwind's `group` and `peer` variants for coordinated animations

**Warning signs:**
- Users report "the toggle doesn't work" (they're clicking it twice)
- Slider jumps instantly instead of animating smoothly

### Pitfall 4: RadioGroup Options Without Keyboard Navigation

**What goes wrong:** Users can't navigate options with arrow keys, breaking accessibility.

**Why it happens:** Using custom radio buttons instead of Headless UI RadioGroup, or blocking default keyboard handlers.

**How to avoid:**
- Always use Headless UI RadioGroup for radio button groups
- Don't add `onKeyDown` handlers that prevent default behavior
- Test with keyboard-only navigation (Tab, Arrow keys, Enter/Space)

**Warning signs:**
- Arrow keys don't move between options
- Space/Enter don't select the focused option
- Screen reader users report navigation issues

### Pitfall 5: URL State Validation Failing Silently

**What goes wrong:** Invalid URLs cause silent failures; users see defaults but no error message.

**Why it happens:** Validation errors are logged but not shown to users.

**How to avoid:**
- Always show a toast notification when URL validation fails (see Pattern 2 example)
- Log the validation error details for debugging
- Consider a "Copy URL" button that warns if the URL is too long

**Warning signs:**
- Users report "my shared link doesn't work"
- Console shows validation errors but UI shows no feedback

## Code Examples

Verified patterns from official sources:

### Zod Schema for URL State Validation

```typescript
// Source: Zod docs + raidy validation patterns
// utils/schemas.ts
import { z } from 'zod'

// URL state schema with mode-dependent fields
export const UrlStateSchema = z.object({
  // Mode indicator
  mode: z.enum(['inference', 'training']).default('inference'),

  // Model/GPU (shared)
  modelId: z.string().optional(),
  gpuId: z.string().optional(),

  // Inference state
  quantization: z.enum(['fp32', 'fp16', 'bf16', 'int8', 'int4', 'gptq', 'awq']),
  sequenceLength: z.number().int().min(512).max(131072),
  batchSize: z.number().int().min(1).max(64),
  kvQuantization: z.enum(['fp16', 'fp8', 'int8', 'int4']),
  numGPUs: z.number().int().min(1).max(8),
  shardingStrategy: z.enum(['tensor-parallel', 'pipeline-parallel']),
  offloadingEnabled: z.boolean().optional(),
  // ... other inference fields

  // Training state (optional - only present when mode=training)
  trainingMethod: z.enum(['full', 'lora', 'qlora']).optional(),
  optimizer: z.enum(['adamw', 'sgd-momentum', 'adamw-8bit', 'adafactor']).optional(),
  trainingPrecision: z.enum(['fp32', 'fp16', 'bf16']).optional(),
  loraRank: z.number().int().min(4).max(256).optional(),
  loraAlpha: z.number().int().min(1).max(512).optional(),
  targetModulesPercent: z.number().int().min(10).max(100).optional(),
})

export type UrlState = z.infer<typeof UrlStateSchema>

export function validateUrlState(data: unknown): UrlState | null {
  const result = UrlStateSchema.safeParse(data)
  if (!result.success) {
    console.error('URL state validation failed:', result.error.format())
    return null
  }
  return result.data
}
```

### Tailwind CSS Toggle Switch Styling

```typescript
// Source: Tailwind CSS toggle best practices + Headless UI examples
// Using peer-checked variant for state-dependent styling
export function ModeToggle() {
  const mode = useUIStore((s) => s.mode)
  const setMode = useUIStore((s) => s.setMode)

  return (
    <label className="flex items-center cursor-pointer">
      <input
        type="checkbox"
        checked={mode === 'training'}
        onChange={(e) => setMode(e.target.checked ? 'training' : 'inference')}
        className="sr-only peer"
      />
      <div className="relative w-11 h-6 bg-gray-200 dark:bg-gray-700 rounded-full
        peer-checked:bg-blue-600 peer-focus:ring-2 peer-focus:ring-blue-500
        peer-focus:ring-offset-2 transition-colors duration-200">
        <div className="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full
          peer-checked:translate-x-full transition-transform duration-200 ease-in-out" />
      </div>
      <span className="ml-3 text-sm font-medium text-gray-900 dark:text-white">
        {mode === 'training' ? 'Training Mode' : 'Inference Mode'}
      </span>
    </label>
  )
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual URL sync with useEffect | persist middleware with StateStorage | Zustand v3+ (2021) | Eliminates race conditions, handles SSR |
| localStorage only | URL hash with lz-string | Modern SPAs | Enables shareable links |
| Global state dump | Selective persistence (partialize) | persist middleware | Reduces URL size, faster hydration |
| Context API for forms | Zustand slices | Zustand v4+ (2022) | Better performance, less boilerplate |
| Custom toggle components | Headless UI Switch | Headless UI v2 (2024) | Accessibility built-in |

**Deprecated/outdated:**
- **useURLSync hook pattern** (current llmvram approach): Manual sync is error-prone; replaced by persist middleware
- **Custom toggle styling without ARIA**: Fails accessibility audits; use Headless UI Switch
- **Unvalidated URL hydration**: Security risk; always validate with Zod before hydrating

## Open Questions

1. **Should training state persist when switching back to inference mode?**
   - What we know: raidy persists all state regardless of active panel
   - What's unclear: Whether training config should be "remembered" or reset when switching modes
   - Recommendation: **Persist training state even in inference mode** for better UX (users can experiment without losing config). Only serialize to URL when `mode=training` to keep URLs short.

2. **How to handle LoRA-specific inputs when method is not LoRA/QLoRA?**
   - What we know: Headless UI supports conditional rendering in RadioGroup
   - What's unclear: Should LoRA inputs be hidden or disabled when method=full?
   - Recommendation: **Hide LoRA inputs completely** when `trainingMethod !== 'lora' && trainingMethod !== 'qlora'` for cleaner UI. Preserve their values in state (don't reset) for better UX.

3. **Should mode toggle have a loading state when calculations switch?**
   - What we know: Phase 6 calculations are synchronous (no loading needed)
   - What's unclear: Whether mode switching will trigger heavy re-calculations in Phase 8+
   - Recommendation: **No loading state for Phase 7** since calculations are fast. Revisit in Phase 8 if calculations move to Web Workers.

## Sources

### Primary (HIGH confidence)
- [Context7: Zustand v5.0.8](https://context7.com/pmndrs/zustand/v5.0.8) - Slice pattern, persist middleware, state creators
- [Context7: Headless UI v2.2.9](https://context7.com/tailwindlabs/headlessui/_headlessui_react_v2_2_9) - Switch component, RadioGroup component, accessibility patterns
- [Zod TypeScript Validation](https://zod.dev/) - Schema validation, safe parsing, type inference
- [raidy project codebase](file:///Users/fjacquet/Projects/raidy/src/store/) - urlStorage.ts, configStore.ts, slice pattern implementation

### Secondary (MEDIUM confidence)
- [React Toggle Switch UX Best Practices](https://www.eleken.co/blog-posts/toggle-ux) - Immediate effect, reversible actions, clear labels
- [Tailwind CSS Toggle Accessibility](https://dockyard.com/blog/2024/05/28/creating-an-accessible-toggle-switch-in-tailwindcss) - ARIA roles, peer variants, keyboard nav
- [Zustand Architecture Patterns at Scale](https://brainhub.eu/library/zustand-architecture-patterns-at-scale) - Slice pattern for large stores
- [React Conditional Rendering Patterns 2026](https://www.sayonetech.com/blog/react-design-patterns/) - Strategy pattern, presentational components

### Tertiary (LOW confidence)
- Web search: "form validation patterns React Zod 2026" - General Zod usage patterns (not specific to this project)

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** - All libraries already in use; verified versions from package.json
- Architecture: **HIGH** - Slice pattern proven in raidy; Headless UI patterns from official docs
- Pitfalls: **MEDIUM** - Based on raidy implementation and Zustand best practices; not all tested in llmvram yet

**Research date:** 2026-02-10
**Valid until:** ~March 12, 2026 (30 days for stable libraries)
