# KV Storage Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user park idle sessions' KV cache on an external tier (host memory, local NVMe, network storage: CMX, PowerScale, ObjectScale) and see how many sessions a configuration can hold, how long a parked session takes to resume versus recomputing it, and whether the tier can sustain the traffic.

**Architecture:** One new pure engine module `src/engines/kv-tier.ts` computes a summary from numbers the app already has (max hot sessions from 4b, per-GPU KV per session, prefill time). Settings live in one `kvTier` object in the Zustand store and one `kt` key in the URL hash. A new input panel edits them; the results panel shows the summary. Memory and decode engines are untouched: parked sessions hold no HBM.

**Tech Stack:** React 19, TypeScript strict, Zustand, Zod 4, Vitest + Testing Library, Biome.

**Spec:** `docs/superpowers/specs/2026-09-26-kv-storage-tier-design.md`

## Global Constraints

- Tier "none" (the default) leaves every existing output unchanged.
- Preset bandwidths are **per GPU, read, estimates**; each carries its basis in a code comment:
  host Grace C2C 225, host PCIe 5 50, local NVMe 12, network storage 12.5 GB/s.
- No Dell Lightning FS preset: it targets > 16K GPUs; cluster storage sizing belongs to raidy. The guide says so.
- Defaults: active share 25%, active burst 30 s, capacity unlimited.
- Resume adds a fixed 0.03 s overhead (Dell anchor: offload TTFT 113-129 ms vs 91 ms recompute at 4K).
- PDF/PPTX exports: not in this plan.
- Biome: 2 spaces, single quotes, no semicolons, 100 columns. Run `rtk proxy npx biome check --write .` (plain `npx biome format --write` through rtk silently applies nothing).
- Tests: `npx vitest run <file>`; test files are not typechecked, so build fixtures from real types.
- Commits end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01X6Gm3X317RU4ns4JqR7A4z
  ```

## Review Focus

1. **Active share of 0** (user types 0 or clears the field) — must not divide by zero; clamp to 1%. Test in Task 1.
2. **Custom bandwidth empty or 0** — fall back to the preset, never divide by zero. Test in Task 1.
3. **Capacity smaller than one session** — sessions held is 0, not negative or NaN. Test in Task 1.
4. **KV offloaded or model does not fit** (`maxConcurrentSessions` returns null or 0) — the tier block hides on null and reports 0 held on 0. Test in Task 1.
5. **Old share links** without `kt` / `cu` keys — restore tier "none" and 1 user; links still parse. Test in Task 2.

---

### Task 1: KV tier engine

**Files:**
- Create: `src/engines/kv-tier.ts`
- Test: `src/engines/kv-tier.test.ts`

**Interfaces:**
- Consumes: `calculateInferenceVRAM` (`src/engines/inference.ts`), `calculateMultiGPUVRAM` (`src/engines/multi-gpu.ts`), `estimatePerformance` (`src/engines/performance.ts`) — tests only.
- Produces:
  - `type KVTierType = 'none' | 'host-grace' | 'host-pcie' | 'local-nvme' | 'network'`
  - `interface KVTierSettings { tier: KVTierType; customGBps: number | null; activeShare: number; burstSeconds: number; capacityTB: number | null }`
  - `const DEFAULT_KV_TIER: KVTierSettings`
  - `const KV_TIER_PRESETS: Record<Exclude<KVTierType, 'none'>, { label: string; gbpsPerGPU: number }>`
  - `const KV_TIER_TYPES: readonly KVTierType[]` (for the Zod enum)
  - `function tierBandwidthGBps(settings: KVTierSettings): number | null`
  - `function resumeSeconds(kvPerSessionPerGPUGB: number, gbpsPerGPU: number): number`
  - `interface KVTierSummary { sessionsHeld: number; resumeSeconds: number; recomputeSeconds: number | null; resumeFaster: boolean | null; trafficGBps: number; tierGBps: number }`
  - `function kvTierSummary(p: { settings: KVTierSettings; maxHotSessions: number; kvPerSessionPerGPUGB: number; kvPerSessionGB: number; totalGPUs: number; recomputeSeconds: number | null }): KVTierSummary | null`

- [ ] **Step 1: Write the failing tests**

```ts
// src/engines/kv-tier.test.ts
import type { GPU, Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import { calculateInferenceVRAM } from './inference'
import {
  DEFAULT_KV_TIER,
  type KVTierSettings,
  kvTierSummary,
  resumeSeconds,
  tierBandwidthGBps,
} from './kv-tier'
import { calculateMultiGPUVRAM } from './multi-gpu'
import { estimatePerformance } from './performance'

const network: KVTierSettings = { ...DEFAULT_KV_TIER, tier: 'network' }

const base = {
  settings: network,
  maxHotSessions: 100,
  kvPerSessionPerGPUGB: 2,
  kvPerSessionGB: 8,
  totalGPUs: 4,
  recomputeSeconds: 5,
}

describe('tierBandwidthGBps', () => {
  it('is null with no tier', () => {
    expect(tierBandwidthGBps(DEFAULT_KV_TIER)).toBeNull()
  })

  it('uses the preset, or a positive custom value', () => {
    expect(tierBandwidthGBps(network)).toBe(12.5)
    expect(tierBandwidthGBps({ ...network, customGBps: 40 })).toBe(40)
  })

  it('falls back to the preset when the custom value is 0 or empty', () => {
    expect(tierBandwidthGBps({ ...network, customGBps: 0 })).toBe(12.5)
    expect(tierBandwidthGBps({ ...network, customGBps: null })).toBe(12.5)
  })
})

describe('resumeSeconds', () => {
  it('reproduces the Dell ObjectScale anchor within 10%', () => {
    // Dell: 43 GB KV at 235K tokens, TP4 on XE9680, 837 ms to first token.
    // Per GPU 43 / 4 GB at 12.8 GB/s per GPU (>= 51 GB/s per server).
    const s = resumeSeconds(43 / 4, 12.8)
    expect(s).toBeGreaterThan(0.837 * 0.9)
    expect(s).toBeLessThan(0.837 * 1.1)
  })
})

describe('kvTierSummary', () => {
  it('is null with no tier', () => {
    expect(kvTierSummary({ ...base, settings: DEFAULT_KV_TIER })).toBeNull()
  })

  it('holds hot sessions divided by the active share', () => {
    expect(kvTierSummary(base)?.sessionsHeld).toBe(400)
  })

  it('caps sessions held by the tier capacity', () => {
    // 1 TB / 8 GB per session = 125
    const s = kvTierSummary({ ...base, settings: { ...network, capacityTB: 1 } })
    expect(s?.sessionsHeld).toBe(125)
  })

  it('holds 0 when the capacity is below one session', () => {
    const s = kvTierSummary({ ...base, settings: { ...network, capacityTB: 0.001 } })
    expect(s?.sessionsHeld).toBe(0)
  })

  it('holds 0 when nothing fits in HBM', () => {
    expect(kvTierSummary({ ...base, maxHotSessions: 0 })?.sessionsHeld).toBe(0)
  })

  it('clamps an active share of 0 to 1%', () => {
    const s = kvTierSummary({ ...base, settings: { ...network, activeShare: 0 } })
    expect(s?.sessionsHeld).toBe(10000)
  })

  it('prices tier traffic as resumes per second times one session of KV', () => {
    // 400 held x 25% active / 30 s burst = 3.33 resumes/s x 8 GB = 26.7 GB/s
    const s = kvTierSummary(base)
    expect(s?.trafficGBps).toBeCloseTo((400 * 0.25 * 8) / 30, 6)
    expect(s?.tierGBps).toBe(12.5 * 4)
  })

  it('reports recompute as unknown when prefill time is unknown', () => {
    const s = kvTierSummary({ ...base, recomputeSeconds: null })
    expect(s?.resumeFaster).toBeNull()
  })
})

describe('resume vs recompute, Dell crossover (8-16K tokens)', () => {
  // Qwen3-Coder-30B-A3B-Instruct config.json: 48 layers, 4 KV heads x head_dim 128,
  // 128 experts top-8; safetensors 30.5B. KV = 48 x 2 x 4 x 128 = 49152 per token.
  const coder: Model = {
    id: 'qwen3-coder-30b-a3b',
    name: 'Qwen3 Coder 30B A3B',
    architecture: 'moe',
    num_parameters_billion: 30.5,
    active_parameters_billion: 3.3,
    hidden_size: 2048,
    num_hidden_layers: 48,
    num_attention_heads: 32,
    num_kv_heads: 4,
    intermediate_size: 6144,
    num_experts: 128,
    num_experts_per_token: 8,
    kv_cache_elements_per_token: 49152,
  }
  const h100: GPU = {
    id: 'h100',
    name: 'H100',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 3350,
    memory_type: 'HBM3',
    bus_width: 5120,
    fp16_tflops: 989,
    interconnect: 'nvlink-4',
    tier: 'datacenter',
    max_gpus_per_node: 8,
  }
  const summaryAt = (sequenceLength: number) => {
    const single = calculateInferenceVRAM({
      model: coder,
      quantization: 'bf16',
      sequenceLength,
      batchSize: 1,
    })
    const multi = calculateMultiGPUVRAM(single, coder, 80, 4, 'tensor-parallel', h100)
    const perf = estimatePerformance({
      model: coder,
      gpu: h100,
      quantization: 'bf16',
      batchSize: 1,
      sequenceLength,
      multiGPUResult: multi,
    })
    return kvTierSummary({
      settings: network,
      maxHotSessions: 10,
      kvPerSessionPerGPUGB: multi.perGPU.kvCache.toNumber(),
      kvPerSessionGB: single.kvCache.toNumber(),
      totalGPUs: 4,
      recomputeSeconds: perf.prefillSeconds?.toNumber() ?? null,
    })
  }

  it('resumes faster than recompute at 235K tokens', () => {
    expect(summaryAt(235000)?.resumeFaster).toBe(true)
  })

  it('recomputes faster than resuming at 4K tokens', () => {
    expect(summaryAt(4096)?.resumeFaster).toBe(false)
  })
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/engines/kv-tier.test.ts`
Expected: FAIL, `Failed to resolve import "./kv-tier"`.

- [ ] **Step 3: Implement**

```ts
// src/engines/kv-tier.ts
/**
 * KV storage tier: park idle sessions' KV cache off the GPU and reload it on resume.
 *
 * Parked sessions hold no HBM, so memory and decode engines are unchanged; this
 * module only answers how many sessions a configuration holds, how long a resume
 * takes versus recomputing the prefill, and whether the tier sustains the churn.
 * Spec: docs/superpowers/specs/2026-09-26-kv-storage-tier-design.md
 */

export type KVTierType = 'none' | 'host-grace' | 'host-pcie' | 'local-nvme' | 'network'

export const KV_TIER_TYPES = [
  'none',
  'host-grace',
  'host-pcie',
  'local-nvme',
  'network',
] as const satisfies readonly KVTierType[]

export interface KVTierSettings {
  tier: KVTierType
  /** Overrides the preset when positive (GB/s per GPU, read) */
  customGBps: number | null
  /** Fraction of sessions decoding at any moment (0-1) */
  activeShare: number
  /** Mean length of one active burst before the session parks again */
  burstSeconds: number
  /** Tier capacity in TB; null = unlimited */
  capacityTB: number | null
}

export const DEFAULT_KV_TIER: KVTierSettings = {
  tier: 'none',
  customGBps: null,
  activeShare: 0.25,
  burstSeconds: 30,
  capacityTB: null,
}

/** Per-GPU read bandwidth presets, GB/s. All estimates; basis in each comment. */
export const KV_TIER_PRESETS: Record<
  Exclude<KVTierType, 'none'>,
  { label: string; gbpsPerGPU: number }
> = {
  // NVLink-C2C 900 GB/s per Grace superchip, shared by 2 GPUs, one direction.
  // NVIDIA publishes no per-GPU figure; estimate.
  'host-grace': { label: 'Host memory (Grace NVLink-C2C)', gbpsPerGPU: 225 },
  // PCIe 5 x16 ~64 GB/s theoretical, ~50 practical.
  'host-pcie': { label: 'Host memory (PCIe 5)', gbpsPerGPU: 50 },
  // GB300 tray: 4 E1.S Gen5 drives for 4 GPUs (NVIDIA NVL72 reference architecture).
  'local-nvme': { label: 'Local NVMe', gbpsPerGPU: 12 },
  // One 400 GbE storage NIC share per GPU; Dell ObjectScale anchor: >= 51 GB/s per
  // 4-GPU server = 12.8 per GPU.
  // No Dell Lightning FS preset: it targets > 16K GPUs; size cluster storage in raidy.
  network: { label: 'Network storage (CMX, PowerScale, ObjectScale)', gbpsPerGPU: 12.5 },
}

/**
 * Fixed cost of a resume beyond the transfer: Dell measured offload at 113-129 ms
 * against 91 ms recompute at 4K tokens, where the transfer itself is ~15 ms.
 */
export const KV_TIER_RESUME_OVERHEAD_S = 0.03

export function tierBandwidthGBps(settings: KVTierSettings): number | null {
  if (settings.tier === 'none') return null
  if (settings.customGBps && settings.customGBps > 0) return settings.customGBps
  return KV_TIER_PRESETS[settings.tier].gbpsPerGPU
}

export function resumeSeconds(kvPerSessionPerGPUGB: number, gbpsPerGPU: number): number {
  return KV_TIER_RESUME_OVERHEAD_S + kvPerSessionPerGPUGB / gbpsPerGPU
}

export interface KVTierSummary {
  sessionsHeld: number
  resumeSeconds: number
  recomputeSeconds: number | null
  /** null when the prefill time is unknown (GPU without FLOPS data) */
  resumeFaster: boolean | null
  /** Tier read traffic needed to resume sessions at the configured churn */
  trafficGBps: number
  /** What the tier delivers across all GPUs */
  tierGBps: number
}

export function kvTierSummary(p: {
  settings: KVTierSettings
  maxHotSessions: number
  kvPerSessionPerGPUGB: number
  kvPerSessionGB: number
  totalGPUs: number
  recomputeSeconds: number | null
}): KVTierSummary | null {
  const bandwidth = tierBandwidthGBps(p.settings)
  if (bandwidth === null) return null

  const share = Math.min(1, Math.max(0.01, p.settings.activeShare))
  const burst = Math.max(1, p.settings.burstSeconds)
  const byShare = Math.floor(p.maxHotSessions / share)
  const byCapacity =
    p.settings.capacityTB && p.kvPerSessionGB > 0
      ? Math.floor((p.settings.capacityTB * 1000) / p.kvPerSessionGB)
      : Number.POSITIVE_INFINITY
  const sessionsHeld = Math.max(0, Math.min(byShare, byCapacity))

  const resume = resumeSeconds(p.kvPerSessionPerGPUGB, bandwidth)
  return {
    sessionsHeld,
    resumeSeconds: resume,
    recomputeSeconds: p.recomputeSeconds,
    resumeFaster: p.recomputeSeconds === null ? null : resume < p.recomputeSeconds,
    trafficGBps: ((sessionsHeld * share) / burst) * p.kvPerSessionGB,
    tierGBps: bandwidth * p.totalGPUs,
  }
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/engines/kv-tier.test.ts`
Expected: PASS (14 tests).

- [ ] **Step 5: Export from the engine barrel and commit**

Add to `src/engines/index.ts` next to the other exports:
```ts
export {
  DEFAULT_KV_TIER,
  KV_TIER_PRESETS,
  type KVTierSettings,
  type KVTierType,
  kvTierSummary,
} from './kv-tier'
```
Run: `rtk proxy npx biome check --write src/engines && npx vitest run src/engines`
```bash
git add src/engines/kv-tier.ts src/engines/kv-tier.test.ts src/engines/index.ts
git commit -m "feat: KV storage tier engine (sessions held, resume vs recompute, traffic)"
```

---

### Task 2: Store and shareable URL (tier settings and concurrent users)

`concurrentUsers` is not in the URL today, so a shared link loses the max-sessions answer; this task adds it with the tier.

**Files:**
- Modify: `src/engines/constants.ts` (add `MAX_CONCURRENT_USERS`)
- Modify: `src/components/inputs/ConcurrentUsersInput.tsx` (import the constant instead of defining it)
- Modify: `src/components/inputs/ConcurrentUsersInput.test.tsx` (import path)
- Modify: `src/store/uiStore.ts` (field `kvTier`, action `setKVTier`)
- Modify: `src/store/urlSerializer.ts` (keys `cu`, `kt`)
- Modify: `src/hooks/useURLSync.ts` (restore both)
- Test: `src/store/urlSerializer.test.ts`

**Interfaces:**
- Consumes: `KVTierSettings`, `DEFAULT_KV_TIER`, `KV_TIER_TYPES` from Task 1.
- Produces: store `kvTier: KVTierSettings`, `setKVTier(patch: Partial<KVTierSettings>): void`; `MAX_CONCURRENT_USERS = 65536` in `@engines/constants`.

- [ ] **Step 1: Write the failing tests** (append inside `describe('serializeToURL and deserializeFromURL', ...)` in `src/store/urlSerializer.test.ts`)

```ts
    it('round-trips concurrent users and the KV tier', () => {
      const serialized = serializeToURL({
        ...baseState,
        concurrentUsers: 2500,
        kvTier: {
          tier: 'network',
          customGBps: 20,
          activeShare: 0.1,
          burstSeconds: 45,
          capacityTB: 500,
        },
      })
      const d = deserializeFromURL(serialized)
      expect(d?.cu).toBe(2500)
      expect(d?.kt).toEqual({ t: 'network', g: 20, a: 0.1, b: 45, c: 500 })
    })

    it('omits the tier when it is none and users when there is one', () => {
      const d = deserializeFromURL(
        serializeToURL({
          ...baseState,
          concurrentUsers: 1,
          kvTier: { tier: 'none', customGBps: null, activeShare: 0.25, burstSeconds: 30, capacityTB: null },
        }),
      )
      expect(d?.cu).toBeUndefined()
      expect(d?.kt).toBeUndefined()
    })

    it('still parses links made before these keys existed', () => {
      const d = deserializeFromURL(serializeToURL(baseState))
      expect(d).not.toBeNull()
      expect(d?.cu).toBeUndefined()
      expect(d?.kt).toBeUndefined()
    })
```

- [ ] **Step 2: Run to verify they fail**

Run: `npx vitest run src/store/urlSerializer.test.ts`
Expected: FAIL, `expected undefined to be 2500`.

- [ ] **Step 3: Implement**

`src/engines/constants.ts`, after `GPU_MEMORY_UTILIZATION`:
```ts
/**
 * Upper bound on concurrent users: rack-scale sizing reaches thousands of sessions
 * (e.g. ~2,200 for Kimi K3 on one GB300 NVL72 at 262k context).
 */
export const MAX_CONCURRENT_USERS = 65536
```
`src/components/inputs/ConcurrentUsersInput.tsx`: delete the local `export const MAX_CONCURRENT_USERS = 65536` and its comment, add `import { MAX_CONCURRENT_USERS } from '@engines/constants'`. In `ConcurrentUsersInput.test.tsx` change the import to
`import { ConcurrentUsersInput } from './ConcurrentUsersInput'` plus `import { MAX_CONCURRENT_USERS } from '@engines/constants'`.

`src/store/uiStore.ts`:
- import: `import { DEFAULT_KV_TIER, type KVTierSettings } from '@engines/kv-tier'`
- interface (next to `concurrentUsers: number`): `kvTier: KVTierSettings` and, with the actions, `setKVTier: (patch: Partial<KVTierSettings>) => void`
- defaults (next to `concurrentUsers: 1`): `kvTier: DEFAULT_KV_TIER,`
- actions (next to `setConcurrentUsers`): `setKVTier: (patch) => set((state) => ({ kvTier: { ...state.kvTier, ...patch } })),`

`src/store/urlSerializer.ts`:
- imports: `import { MAX_CONCURRENT_USERS } from '@engines/constants'` and `import { KV_TIER_TYPES, type KVTierSettings } from '@engines/kv-tier'`
- schema, after `ss`:
```ts
  cu: z.number().int().min(1).max(MAX_CONCURRENT_USERS).optional(), // concurrentUsers (absent = 1)
  kt: z
    .object({
      t: z.enum(KV_TIER_TYPES), // tier
      g: z.number().positive().optional(), // customGBps
      a: z.number().min(0.01).max(1), // activeShare
      b: z.number().min(1), // burstSeconds
      c: z.number().positive().optional(), // capacityTB
    })
    .optional(), // KV storage tier (absent = none)
```
- `serializeToURL` parameter type, add: `concurrentUsers?: number` and `kvTier?: KVTierSettings`
- body, after `ss: state.shardingStrategy,`:
```ts
    ...(state.concurrentUsers && state.concurrentUsers > 1 ? { cu: state.concurrentUsers } : {}),
    ...(state.kvTier && state.kvTier.tier !== 'none'
      ? {
          kt: {
            t: state.kvTier.tier,
            ...(state.kvTier.customGBps ? { g: state.kvTier.customGBps } : {}),
            a: state.kvTier.activeShare,
            b: state.kvTier.burstSeconds,
            ...(state.kvTier.capacityTB ? { c: state.kvTier.capacityTB } : {}),
          },
        }
      : {}),
```

`src/hooks/useURLSync.ts`, after `store.setShardingStrategy(...)`:
```ts
    // Absent in links made before these keys existed: 1 user, no tier
    store.setConcurrentUsers(urlState.cu ?? 1)
    store.setKVTier(
      urlState.kt
        ? {
            tier: urlState.kt.t,
            customGBps: urlState.kt.g ?? null,
            activeShare: urlState.kt.a,
            burstSeconds: urlState.kt.b,
            capacityTB: urlState.kt.c ?? null,
          }
        : DEFAULT_KV_TIER,
    )
```
with `import { DEFAULT_KV_TIER } from '@engines/kv-tier'`.

- [ ] **Step 4: Run to verify they pass**

Run: `npx vitest run src/store src/components/inputs && npm run typecheck`
Expected: PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
rtk proxy npx biome check --write .
git add src/engines/constants.ts src/components/inputs/ConcurrentUsersInput.tsx src/components/inputs/ConcurrentUsersInput.test.tsx src/store/uiStore.ts src/store/urlSerializer.ts src/store/urlSerializer.test.ts src/hooks/useURLSync.ts
git commit -m "feat: share concurrent users and KV tier settings in the URL"
```

---

### Task 3: KV tier input panel

**Files:**
- Create: `src/components/inputs/KVTierPanel.tsx`
- Test: `src/components/inputs/KVTierPanel.test.tsx`
- Modify: `src/components/layout/InputPanel.tsx` (render under `<OffloadingPanel />`, inference only)

**Interfaces:**
- Consumes: store `kvTier`, `setKVTier` (Task 2); `KV_TIER_PRESETS`, `KV_TIER_TYPES`, `type KVTierType` (Task 1).
- Produces: `export function KVTierPanel(): JSX.Element`.

- [ ] **Step 1: Write the failing test**

```tsx
// src/components/inputs/KVTierPanel.test.tsx
import { DEFAULT_KV_TIER, type KVTierSettings } from '@engines/kv-tier'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Plain store instead of the persisted uiStore, which throws in jsdom
// (see NodeCountSelector.test.tsx).
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<{
    kvTier: KVTierSettings
    setKVTier: (p: Partial<KVTierSettings>) => void
  }>((set) => ({
    kvTier: {
      tier: 'none',
      customGBps: null,
      activeShare: 0.25,
      burstSeconds: 30,
      capacityTB: null,
    },
    setKVTier: (p) => set((s) => ({ kvTier: { ...s.kvTier, ...p } })),
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { KVTierPanel } from './KVTierPanel'

describe('KVTierPanel', () => {
  beforeEach(() => useUIStore.setState({ kvTier: DEFAULT_KV_TIER }))

  it('hides the tier settings when no tier is selected', () => {
    render(<KVTierPanel />)
    expect(screen.queryByLabelText('Active share (%)')).not.toBeInTheDocument()
  })

  it('selects network storage and shows its settings', () => {
    render(<KVTierPanel />)
    fireEvent.change(screen.getByLabelText('KV storage tier'), { target: { value: 'network' } })
    expect(useUIStore.getState().kvTier.tier).toBe('network')
    expect(screen.getByLabelText('Active share (%)')).toBeInTheDocument()
  })

  it('does not offer a Dell Lightning preset (cluster-scale storage, sized in raidy)', () => {
    render(<KVTierPanel />)
    expect(screen.queryByRole('option', { name: /Lightning/ })).not.toBeInTheDocument()
  })

  it('stores the active share as a fraction, clamped to 1-100%', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network' } })
    render(<KVTierPanel />)
    const share = screen.getByLabelText('Active share (%)')
    fireEvent.change(share, { target: { value: '10' } })
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.1)
    fireEvent.change(share, { target: { value: '0' } })
    expect(useUIStore.getState().kvTier.activeShare).toBeCloseTo(0.01)
  })

  it('treats a cleared capacity as unlimited', () => {
    useUIStore.setState({ kvTier: { ...DEFAULT_KV_TIER, tier: 'network', capacityTB: 100 } })
    render(<KVTierPanel />)
    fireEvent.change(screen.getByLabelText('Tier capacity (TB)'), { target: { value: '' } })
    expect(useUIStore.getState().kvTier.capacityTB).toBeNull()
  })
})
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/components/inputs/KVTierPanel.test.tsx`
Expected: FAIL, `Failed to resolve import "./KVTierPanel"`.

- [ ] **Step 3: Implement**

```tsx
// src/components/inputs/KVTierPanel.tsx
import { InfoTip } from '@components/common/InfoTip'
import { KV_TIER_PRESETS, KV_TIER_TYPES, type KVTierType } from '@engines/kv-tier'
import { useUIStore } from '@store/uiStore'

const inputClass =
  'w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white'

/** Parses a number field; empty or non-positive becomes null. */
const positiveOrNull = (raw: string): number | null => {
  const n = Number(raw)
  return raw.trim() === '' || !Number.isFinite(n) || n <= 0 ? null : n
}

/**
 * KV storage tier: park idle sessions' KV off the GPU (host memory, local NVMe,
 * CMX, PowerScale, ObjectScale) and reload it on resume.
 */
export function KVTierPanel() {
  const kvTier = useUIStore((s) => s.kvTier)
  const setKVTier = useUIStore((s) => s.setKVTier)
  const preset = kvTier.tier === 'none' ? null : KV_TIER_PRESETS[kvTier.tier]

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="kv-tier"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          KV storage tier
        </label>
        <InfoTip text="Idle agentic sessions park their KV cache on this tier and reload it when they resume, instead of recomputing the prompt. Parked sessions use no GPU memory." />
      </div>
      <select
        id="kv-tier"
        aria-label="KV storage tier"
        value={kvTier.tier}
        onChange={(e) => setKVTier({ tier: e.target.value as KVTierType })}
        className={inputClass}
      >
        {KV_TIER_TYPES.map((t) => (
          <option key={t} value={t}>
            {t === 'none'
              ? 'None'
              : `${KV_TIER_PRESETS[t].label} (${KV_TIER_PRESETS[t].gbpsPerGPU} GB/s per GPU)`}
          </option>
        ))}
      </select>

      {preset && (
        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Bandwidth per GPU (GB/s)
            <input
              type="number"
              min={0}
              aria-label="Bandwidth per GPU (GB/s)"
              placeholder={String(preset.gbpsPerGPU)}
              value={kvTier.customGBps ?? ''}
              onChange={(e) => setKVTier({ customGBps: positiveOrNull(e.target.value) })}
              className={inputClass}
            />
          </label>
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Active share (%)
            <input
              type="number"
              min={1}
              max={100}
              aria-label="Active share (%)"
              value={Math.round(kvTier.activeShare * 100)}
              onChange={(e) =>
                setKVTier({ activeShare: Math.min(100, Math.max(1, Number(e.target.value) || 0)) / 100 })
              }
              className={inputClass}
            />
          </label>
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Active burst (s)
            <input
              type="number"
              min={1}
              aria-label="Active burst (s)"
              value={kvTier.burstSeconds}
              onChange={(e) => setKVTier({ burstSeconds: Math.max(1, Number(e.target.value) || 1) })}
              className={inputClass}
            />
          </label>
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Tier capacity (TB)
            <input
              type="number"
              min={0}
              aria-label="Tier capacity (TB)"
              placeholder="unlimited"
              value={kvTier.capacityTB ?? ''}
              onChange={(e) => setKVTier({ capacityTB: positiveOrNull(e.target.value) })}
              className={inputClass}
            />
          </label>
        </div>
      )}
    </div>
  )
}
```

`src/components/layout/InputPanel.tsx`: add `import { KVTierPanel } from '@components/inputs/KVTierPanel'` and, directly after `<OffloadingPanel />`:
```tsx
            {mode === 'inference' && (
              <>
                <hr className="border-gray-200 dark:border-gray-700" />
                <KVTierPanel />
              </>
            )}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/components/inputs/KVTierPanel.test.tsx && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk proxy npx biome check --write .
git add src/components/inputs/KVTierPanel.tsx src/components/inputs/KVTierPanel.test.tsx src/components/layout/InputPanel.tsx
git commit -m "feat: KV storage tier input panel"
```

---

### Task 4: Results block, docs, end-to-end check

**Files:**
- Modify: `src/components/layout/ResultsPanel.tsx`
- Modify: `CHANGELOG.md`, `ARCHITECTURE.md`, `README.md`, `src/components/guide/GuidePage.tsx`, `docs/superpowers/specs/2026-09-26-datacenter-sizing-prd.md`, `docs/superpowers/specs/2026-09-26-kv-storage-tier-design.md`

**Interfaces:**
- Consumes: `kvTierSummary` (Task 1), store `kvTier` (Task 2), `maxSessions` and `perGPU` already computed in `ResultsPanel` (4b).
- Produces: nothing new for later tasks.

- [ ] **Step 1: Add the results block**

In `ResultsPanel.tsx`: add `kvTier` to the `useUIStore()` destructuring, `import { kvTierSummary } from '@engines/kv-tier'`, and after the `maxSessions` computation:
```tsx
  // KV storage tier: sessions held with idle KV parked off-GPU (null = no tier)
  const tierSummary =
    maxSessions === null
      ? null
      : kvTierSummary({
          settings: kvTier,
          maxHotSessions: maxSessions,
          kvPerSessionPerGPUGB: perGPU.kvCache.toNumber() / Math.max(1, concurrentUsers),
          kvPerSessionGB: result.vram.kvCache.toNumber() / Math.max(1, concurrentUsers),
          totalGPUs: result.multiGPU?.numGPUs ?? 1,
          recomputeSeconds: result.performance.prefillSeconds?.toNumber() ?? null,
        })
```
Right after the max-sessions block:
```tsx
            {tierSummary && (
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-600 space-y-1">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  With the KV storage tier ({Math.round(kvTier.activeShare * 100)}% active)
                </p>
                <p className="text-base font-semibold text-gray-900 dark:text-white">
                  {tierSummary.sessionsHeld.toLocaleString('en-US')} sessions held
                </p>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Resume {formatDuration(new Decimal(tierSummary.resumeSeconds))}
                  {tierSummary.recomputeSeconds !== null &&
                    ` vs recompute ${formatDuration(new Decimal(tierSummary.recomputeSeconds))} (${
                      tierSummary.resumeFaster ? 'resume is faster' : 'recompute is faster'
                    })`}
                </p>
                <p
                  className={`text-xs ${
                    tierSummary.trafficGBps > tierSummary.tierGBps
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}
                >
                  Tier reads {tierSummary.trafficGBps.toFixed(1)} GB/s of{' '}
                  {tierSummary.tierGBps.toFixed(0)} GB/s available
                </p>
              </div>
            )}
```
If `Decimal` is not yet imported in `ResultsPanel.tsx`, add `import Decimal from 'decimal.js'` (check `formatDuration`'s parameter type first with `grep -n "function formatDuration" -A3 src -r`; if it takes a number, pass the number and drop `new Decimal`).

- [ ] **Step 2: Docs**

- `CHANGELOG.md`, top of `### Added` under `## [Unreleased]`:
  `- KV storage tier: park idle sessions' KV cache on host memory, local NVMe or network storage (CMX, PowerScale, ObjectScale) and see sessions held (hot sessions / active share, capped by capacity), resume time vs recompute, and tier traffic vs bandwidth. Presets are per-GPU estimates with their basis in code; resume is anchored on Dell's ObjectScale measurement (43 GB, 837 ms). Concurrent users and tier settings are now part of shared links.`
- `ARCHITECTURE.md`, after the `### Concurrency (\`concurrency.ts\`)` section:
  ```markdown
  ### KV Storage Tier (`kv-tier.ts`)

  - Parked sessions hold no HBM; memory and decode engines are unchanged.
  - Sessions held = `floor(maxConcurrentSessions / activeShare)`, capped by `capacityTB × 1000 / kv_per_session`.
  - Resume = 0.03 s + KV per session per GPU / tier GB/s per GPU; compared with `prefillSeconds`.
  - Traffic = held × activeShare / burstSeconds × KV per session, against tier GB/s × GPUs.
  ```
- `README.md`, after the Concurrent Users bullet:
  `- **KV Storage Tier**: Park idle sessions on host memory, NVMe or network storage; shows sessions held, resume vs recompute and tier traffic`
- `GuidePage.tsx`, after the "Max concurrent sessions" list item, a list item with `<strong>KV storage tier</strong>` explaining: idle sessions park their KV off the GPU; sessions held = sessions that fit ÷ active share; resume vs recompute shows which is faster (short prompts recompute faster); tier traffic turns red when it exceeds the tier bandwidth; cluster-scale storage such as Dell Lightning FS (16K+ GPUs) is sized in the storage calculator, not here. Use `&apos;` for apostrophes.
- PRD status line: `**Status:** 4a-4d implemented`
- Spec: in the Model section, after "Resume time", add `Resume includes a fixed 0.03 s overhead (Dell: offload TTFT 113-129 ms vs 91 ms recompute at 4K).`

- [ ] **Step 3: Full verification**

Run:
```bash
rtk proxy npx biome check --write . && rtk proxy npx biome check .
npx vitest run
npm run typecheck
npm run build
```
Expected: Biome clean, all tests pass, no type errors, build succeeds.

- [ ] **Step 4: End-to-end check in the dev server**

Run `npx vite --port 5199 --strictPort` in the background, open `http://localhost:5199/llmvram/`, and in the browser console (or Playwright `browser_evaluate`):
```js
const { useUIStore } = await import('/llmvram/src/store/uiStore.ts')
const models = (await import('/llmvram/src/data/models.json')).default
const gpus = (await import('/llmvram/src/data/gpus.json')).default
const s = useUIStore.getState()
s.setSelectedModel(models.find((m) => m.id === 'moonshotai-kimi-k3'))
s.setSelectedGPU(gpus.find((g) => g.id === 'nvidia-gb300-nvl72'))
s.setQuantization('mxfp4'); s.setSequenceLength(262144); s.setNumGPUs(72)
s.setShardingStrategy('expert-parallel'); s.setConcurrentUsers(1000)
s.setKVTier({ tier: 'network' })
```
Expected: the results show "Max concurrent sessions ... 2,189" and "8,756 sessions held" (2,189 / 0.25), a resume time under 1 s shorter than recompute, and the tier traffic line. Reload the page with the same URL hash: the same numbers appear (URL round-trip). Stop the server; delete `.playwright-mcp/` if created.

- [ ] **Step 5: Commit**

```bash
git add src/components/layout/ResultsPanel.tsx CHANGELOG.md ARCHITECTURE.md README.md src/components/guide/GuidePage.tsx docs/superpowers/specs/2026-09-26-datacenter-sizing-prd.md docs/superpowers/specs/2026-09-26-kv-storage-tier-design.md
git commit -m "feat: show KV storage tier results; docs"
```
