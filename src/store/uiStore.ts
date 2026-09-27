import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import {
  buildNotice,
  type Correction,
  DEFAULT_RULE_CONFIG,
  type Notice,
  type NoticeSource,
  normalizeConfig,
  pickRuleConfig,
  type RuleConfig,
} from '@engines/config-rules'
import { resolveFabricSpec } from '@engines/fabric'
import { FRAMEWORK_PRESETS, type FrameworkPreset } from '@engines/frameworks'
import type { KVTierSettings } from '@engines/kv-tier'
import type {
  FabricType,
  FineTuningMethod,
  KVCachePrecision,
  OffloadMode,
  OffloadTarget,
  OptimizerType,
  QuantizationFormat,
  ShardingStrategy,
  TrainingPrecision,
} from '@engines/types'
import type { CustomFabricInput, GPU, Model } from '@utils/schemas'
import { validateGPUs, validateModels } from '@utils/schemas'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

// Load and validate data once at module level
const models = validateModels(modelsData)
const gpus = validateGPUs(gpusData)

/**
 * Find a model by ID in the curated database
 */
export function findModelById(id: string): Model | null {
  return models.find((m) => m.id === id) ?? null
}

/**
 * Find a GPU by ID in the curated database
 */
export function findGPUById(id: string): GPU | null {
  return gpus.find((g) => g.id === id) ?? null
}

/**
 * Every calculator input. The rule fields (RuleConfig) are normalized by
 * config-rules on every action; the rest have no dependencies.
 *
 * numGPUs is the parallel degree of one replica inside one server (ADR 0006).
 */
export interface UIConfig extends RuleConfig {
  // Model and GPU selections (not persisted - large objects, may become stale)
  selectedModel: Model | null
  selectedGPU: GPU | null
  quantization: QuantizationFormat
  kvQuantization: KVCachePrecision
  interNodeFabric: FabricType
  customFabric: CustomFabricInput | null
  offloadMode: OffloadMode
  trainingMethod: FineTuningMethod
  optimizer: OptimizerType
  trainingPrecision: TrainingPrecision
  loraAlpha: number
  targetModulesPercent: number
  gradientCheckpointing: boolean
  flashAttention: boolean
}

export const DEFAULT_UI_CONFIG: UIConfig = {
  ...DEFAULT_RULE_CONFIG,
  selectedModel: null,
  selectedGPU: null,
  quantization: 'fp16',
  kvQuantization: 'fp16',
  interNodeFabric: 'ethernet-800g',
  customFabric: null,
  offloadMode: 'percentage',
  trainingMethod: 'lora',
  optimizer: 'adamw',
  trainingPrecision: 'bf16',
  loraAlpha: 32,
  targetModulesPercent: 30,
  gradientCheckpointing: false,
  flashAttention: false,
}

/** A notice waiting to be shown; id increases on every notice so a repeat re-fires effects */
export interface PendingNotice extends Notice {
  id: number
}

interface UIState extends UIConfig {
  // UI preferences (persisted)
  isDarkMode: boolean

  /** Corrections from the last action, for useConfigNotices to toast */
  pendingNotice: PendingNotice | null

  // Actions
  setSelectedModel: (model: Model | null) => void
  setSelectedGPU: (gpu: GPU | null) => void
  setInterconnectOverride: (v: string | null) => void
  setQuantization: (quantization: QuantizationFormat) => void
  setSequenceLength: (sequenceLength: number) => void
  setBatchSize: (batchSize: number) => void
  setKVQuantization: (kvQuantization: KVCachePrecision) => void
  setNumGPUs: (numGPUs: number) => void
  setNumNodes: (numNodes: number) => void
  setInterNodeFabric: (fabric: FabricType) => void
  setCustomFabric: (fabric: CustomFabricInput | null) => void
  setShardingStrategy: (strategy: ShardingStrategy) => void
  setOffloadingEnabled: (enabled: boolean) => void
  setOffloadTarget: (target: OffloadTarget) => void
  setOffloadMode: (mode: OffloadMode) => void
  setOffloadPercentage: (percentage: number) => void
  setOffloadLayers: (layers: number) => void
  setKVCacheOffload: (enabled: boolean) => void
  setOffloadHostCapacityGB: (gb: number | null) => void
  setMode: (mode: 'inference' | 'training') => void
  setTrainingMethod: (method: FineTuningMethod) => void
  setOptimizer: (optimizer: OptimizerType) => void
  setTrainingPrecision: (precision: TrainingPrecision) => void
  setLoraRank: (rank: number) => void
  setLoraAlpha: (alpha: number) => void
  setTargetModulesPercent: (percent: number) => void
  setGradientAccumulationSteps: (steps: number) => void
  setGradientCheckpointing: (enabled: boolean) => void
  setFlashAttention: (enabled: boolean) => void
  setFrameworkPreset: (preset: FrameworkPreset) => void
  setCpuOffloadOptimizer: (enabled: boolean) => void
  setConcurrentUsers: (n: number) => void
  setKVTier: (patch: Partial<KVTierSettings>) => void
  /** Apply a whole shared-link configuration, normalized once ("Shared link adjusted") */
  restoreConfig: (patch: Partial<UIConfig>) => void
  /** Advanced section only (spec Section 4): keeps model, GPU, GPU count, servers, format, context, concurrent users */
  resetAdvancedSettings: () => void
  /** Back to the initial empty state and clears the URL hash (spec Section 4) */
  resetAll: () => void
  clearNotice: () => void
  setIsDarkMode: (dark: boolean) => void
  toggleDarkMode: () => void
}

let noticeSeq = 0

/**
 * The one write path for configuration (ADR 0004): merge the patch, normalize the
 * whole config once, and publish every correction as one notice.
 *
 * `extra` carries corrections an action intent makes on its own (not a rule
 * `normalizeConfig` would find, since the action already avoided the rule firing) but
 * that must still warn the user — e.g. enabling offloading on unified memory picks
 * NVMe itself, so R6 never fires, yet the product rule is "always warn the user".
 *
 * `presetNotice` (spec Section 4, resets) supplies the notice directly instead of
 * deriving it from `buildNotice`: a reset's lines describe fields going back to their
 * defaults, which isn't a rule correction and has no `RuleId`/`CorrectionSubject` to
 * hang off `Correction`. Its title always wins; if normalizeConfig's own rules still
 * find something to correct (e.g. R14 renormalizing `numGPUs` after `shardingStrategy`
 * resets to tensor-parallel), those lines are appended after the preset ones, so a
 * reset that cascades still surfaces as one notice, not two. No `pendingNotice` is
 * written when the combined lines end up empty (nothing actually changed).
 */
function commit(
  set: (partial: Partial<UIState>) => void,
  get: () => UIState,
  patch: Partial<UIConfig>,
  source: NoticeSource,
  extra: Correction[] = [],
  presetNotice?: Notice,
): void {
  const next = { ...get(), ...patch }
  const { config, corrections } = normalizeConfig(
    pickRuleConfig(next),
    next.selectedModel,
    next.selectedGPU,
  )
  const built = buildNotice([...extra, ...corrections], source, {
    model: next.selectedModel,
    gpu: next.selectedGPU,
    config,
  })
  const notice = presetNotice
    ? { title: presetNotice.title, lines: [...presetNotice.lines, ...(built?.lines ?? [])] }
    : built
  set({
    ...patch,
    ...config,
    ...(notice && notice.lines.length > 0 ? { pendingNotice: { ...notice, id: ++noticeSeq } } : {}),
  })
}

/**
 * One line per Advanced-section group that changed from `DEFAULT_UI_CONFIG` (spec
 * Section 4). Grouped the same way `countAdvancedChanges` (Task 6b) counts them —
 * one line for "offloading" or "KV tier" even when several of their fields changed —
 * not one line per raw `UIConfig` key. Mirrors `countAdvancedChanges`'s mode gate: in
 * training every other Advanced-section input is hidden as inert (InputPanel never
 * renders them), so only batch size is reset — the rest keep whatever value they held,
 * exactly like `countAdvancedChanges` never counts them as "changed" in training.
 */
function describeAdvancedReset(before: UIConfig): string[] {
  const d = DEFAULT_UI_CONFIG
  const lines: string[] = []
  if (before.batchSize !== d.batchSize) lines.push(`Batch size reset to ${d.batchSize}.`)
  if (before.mode === 'training') return lines
  if (before.kvQuantization !== d.kvQuantization) {
    lines.push(`KV precision reset to ${d.kvQuantization.toUpperCase()}.`)
  }
  if (before.shardingStrategy !== d.shardingStrategy) {
    lines.push(`Sharding strategy reset to ${d.shardingStrategy.replace('-', ' ')}.`)
  }
  if (before.interNodeFabric !== d.interNodeFabric || before.customFabric !== d.customFabric) {
    lines.push(`Fabric reset to ${resolveFabricSpec(d.interNodeFabric, d.customFabric).label}.`)
  }
  if (before.interconnectOverride !== d.interconnectOverride) {
    lines.push('Interconnect variant reset to the default.')
  }
  const offloadChanged =
    before.offloadingEnabled !== d.offloadingEnabled ||
    before.offloadTarget !== d.offloadTarget ||
    before.offloadMode !== d.offloadMode ||
    before.offloadPercentage !== d.offloadPercentage ||
    before.offloadLayers !== d.offloadLayers ||
    before.kvCacheOffload !== d.kvCacheOffload ||
    before.offloadHostCapacityGB !== d.offloadHostCapacityGB
  if (offloadChanged) lines.push('Offloading reset to off.')
  if (JSON.stringify(before.kvTier) !== JSON.stringify(d.kvTier))
    lines.push('KV tier reset to none.')
  return lines
}

/**
 * Every `UIConfig` field (not the store's extra `isDarkMode`/`pendingNotice`/actions)
 * already matches `DEFAULT_UI_CONFIG`. Exported so `useURLSync` (Task 2b) can keep the
 * hash cleared after `resetAll()`, past the debounced sync effect's next write.
 */
export function isAtDefaults(state: UIConfig): boolean {
  return (Object.keys(DEFAULT_UI_CONFIG) as (keyof UIConfig)[]).every(
    (key) => JSON.stringify(state[key]) === JSON.stringify(DEFAULT_UI_CONFIG[key]),
  )
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      ...DEFAULT_UI_CONFIG,
      pendingNotice: null,
      isDarkMode:
        typeof window !== 'undefined'
          ? window.matchMedia('(prefers-color-scheme: dark)').matches
          : false,

      // Actions: every config write goes through commit()
      setSelectedModel: (model) => commit(set, get, { selectedModel: model }, 'model'),
      setSelectedGPU: (gpu) => commit(set, get, { selectedGPU: gpu }, 'gpu'),
      setInterconnectOverride: (v) => commit(set, get, { interconnectOverride: v }, 'setting'),
      setQuantization: (quantization) => commit(set, get, { quantization }, 'setting'),
      setSequenceLength: (sequenceLength) => commit(set, get, { sequenceLength }, 'setting'),
      setBatchSize: (batchSize) => commit(set, get, { batchSize }, 'setting'),
      setKVQuantization: (kvQuantization) => commit(set, get, { kvQuantization }, 'setting'),
      setNumGPUs: (numGPUs) => commit(set, get, { numGPUs }, 'setting'),
      setNumNodes: (numNodes) => commit(set, get, { numNodes }, 'setting'),
      setInterNodeFabric: (interNodeFabric) => commit(set, get, { interNodeFabric }, 'setting'),
      setCustomFabric: (customFabric) => commit(set, get, { customFabric }, 'setting'),
      setShardingStrategy: (shardingStrategy) => commit(set, get, { shardingStrategy }, 'setting'),
      setOffloadingEnabled: (enabled) => {
        const { selectedGPU, offloadTarget } = get()
        // Action intent, not a rule: on unified memory the only offload target is NVMe,
        // so turning offloading on picks it instead of letting R6 switch it off again.
        const toNVMe =
          enabled && selectedGPU?.unified_memory === true && offloadTarget === 'cpu-ram'
        commit(
          set,
          get,
          toNVMe
            ? { offloadingEnabled: true, offloadTarget: 'nvme' }
            : { offloadingEnabled: enabled },
          'setting',
          // The action, not a rule, made this change (R6 never fires: the patch above
          // already leaves offloadTarget off 'cpu-ram'), but the product rule is
          // "always warn the user", so a synthetic Correction still reaches buildNotice.
          toNVMe && selectedGPU
            ? [
                {
                  rule: 'R6',
                  field: 'offloadTarget',
                  from: 'cpu-ram',
                  to: 'nvme',
                  subject: 'gpu',
                  message: `Offload target set to NVMe: ${selectedGPU.name} has unified memory, RAM is the same pool.`,
                },
              ]
            : [],
        )
      },
      setOffloadTarget: (offloadTarget) => commit(set, get, { offloadTarget }, 'setting'),
      setOffloadMode: (offloadMode) => commit(set, get, { offloadMode }, 'setting'),
      setOffloadPercentage: (offloadPercentage) =>
        commit(set, get, { offloadPercentage }, 'setting'),
      setOffloadLayers: (offloadLayers) => commit(set, get, { offloadLayers }, 'setting'),
      setKVCacheOffload: (kvCacheOffload) => commit(set, get, { kvCacheOffload }, 'setting'),
      setOffloadHostCapacityGB: (offloadHostCapacityGB) =>
        commit(set, get, { offloadHostCapacityGB }, 'setting'),
      setMode: (mode) => commit(set, get, { mode }, 'mode'),
      setTrainingMethod: (trainingMethod) => commit(set, get, { trainingMethod }, 'setting'),
      setOptimizer: (optimizer) => commit(set, get, { optimizer }, 'setting'),
      setTrainingPrecision: (trainingPrecision) =>
        commit(set, get, { trainingPrecision }, 'setting'),
      setLoraRank: (loraRank) => commit(set, get, { loraRank }, 'setting'),
      setLoraAlpha: (loraAlpha) => commit(set, get, { loraAlpha }, 'setting'),
      setTargetModulesPercent: (targetModulesPercent) =>
        commit(set, get, { targetModulesPercent }, 'setting'),
      setGradientAccumulationSteps: (gradientAccumulationSteps) =>
        commit(set, get, { gradientAccumulationSteps }, 'setting'),
      setGradientCheckpointing: (gradientCheckpointing) =>
        commit(set, get, { gradientCheckpointing }, 'setting'),
      setFlashAttention: (flashAttention) => commit(set, get, { flashAttention }, 'setting'),
      setFrameworkPreset: (preset) => {
        const { mode, autoOptimizations } = FRAMEWORK_PRESETS[preset]
        commit(
          set,
          get,
          {
            frameworkPreset: preset,
            // Action intent, not a rule: picking vLLM/TGI means inference. Training
            // presets are kept when switching to inference (inert, preserved for the
            // round trip), so this never runs the other way.
            ...(mode === 'inference' ? { mode: 'inference' as const } : {}),
            ...(autoOptimizations.gradientCheckpointing !== undefined
              ? { gradientCheckpointing: autoOptimizations.gradientCheckpointing }
              : {}),
            ...(autoOptimizations.flashAttention !== undefined
              ? { flashAttention: autoOptimizations.flashAttention }
              : {}),
            ...(autoOptimizations.optimizer !== undefined
              ? { optimizer: autoOptimizations.optimizer }
              : {}),
          },
          'setting',
        )
      },
      setCpuOffloadOptimizer: (cpuOffloadOptimizer) =>
        commit(set, get, { cpuOffloadOptimizer }, 'setting'),
      setConcurrentUsers: (concurrentUsers) => commit(set, get, { concurrentUsers }, 'setting'),
      setKVTier: (patch) => commit(set, get, { kvTier: { ...get().kvTier, ...patch } }, 'setting'),
      restoreConfig: (patch) => commit(set, get, patch, 'link'),
      resetAdvancedSettings: () => {
        const before = get()
        const lines = describeAdvancedReset(before)
        // Training hides every Advanced input but batch size (countAdvancedChanges,
        // InputPanel): only patch what's actually visible, so this can't silently
        // clear an offloading/KV-tier setup the user has no way to see or re-check.
        const patch: Partial<UIConfig> =
          before.mode === 'training'
            ? { batchSize: DEFAULT_UI_CONFIG.batchSize }
            : {
                batchSize: DEFAULT_UI_CONFIG.batchSize,
                kvQuantization: DEFAULT_UI_CONFIG.kvQuantization,
                shardingStrategy: DEFAULT_UI_CONFIG.shardingStrategy,
                interNodeFabric: DEFAULT_UI_CONFIG.interNodeFabric,
                customFabric: DEFAULT_UI_CONFIG.customFabric,
                interconnectOverride: DEFAULT_UI_CONFIG.interconnectOverride,
                offloadingEnabled: DEFAULT_UI_CONFIG.offloadingEnabled,
                offloadTarget: DEFAULT_UI_CONFIG.offloadTarget,
                offloadMode: DEFAULT_UI_CONFIG.offloadMode,
                offloadPercentage: DEFAULT_UI_CONFIG.offloadPercentage,
                offloadLayers: DEFAULT_UI_CONFIG.offloadLayers,
                kvCacheOffload: DEFAULT_UI_CONFIG.kvCacheOffload,
                offloadHostCapacityGB: DEFAULT_UI_CONFIG.offloadHostCapacityGB,
                kvTier: DEFAULT_UI_CONFIG.kvTier,
              }
        commit(set, get, patch, 'reset', [], { title: 'Reset to defaults', lines })
      },
      resetAll: () => {
        // A single line is enough here (spec Section 4): unlike resetAdvancedSettings,
        // this clears everything, so there's no useful "which group" breakdown.
        const wasAtDefaults = isAtDefaults(get())
        commit(set, get, DEFAULT_UI_CONFIG, 'reset', [], {
          title: 'Reset to defaults',
          lines: wasAtDefaults ? [] : ['Configuration reset to defaults.'],
        })
        // Drop the shared-link hash immediately: reset means starting over, not
        // re-sharing the config it just cleared. useURLSync's debounced sync effect
        // (Task 2b) also fires from this same state change; its own `isAtDefaults`
        // guard is what keeps the hash from reappearing 300ms later, not this call.
        if (typeof window !== 'undefined') {
          window.history.replaceState(null, '', window.location.pathname + window.location.search)
        }
      },
      clearNotice: () => set({ pendingNotice: null }),
      setIsDarkMode: (dark) => set({ isDarkMode: dark }),
      toggleDarkMode: () => set({ isDarkMode: !get().isDarkMode }),
    }),
    {
      name: 'llmvram-ui-preferences',
      // Only persist dark mode preference - all other state managed via URL hash
      partialize: (state) => ({
        isDarkMode: state.isDarkMode,
      }),
    },
  ),
)
