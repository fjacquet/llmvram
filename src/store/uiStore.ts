import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import type {
  FineTuningMethod,
  KVCachePrecision,
  OffloadMode,
  OffloadTarget,
  OptimizerType,
  QuantizationFormat,
  ShardingStrategy,
  TrainingPrecision,
} from '@engines/types'
import type { GPU, Model } from '@utils/schemas'
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

interface UIState {
  // Model and GPU selections (not persisted - large objects, may become stale)
  selectedModel: Model | null
  selectedGPU: GPU | null

  // Calculation parameters (persisted)
  quantization: QuantizationFormat
  sequenceLength: number
  batchSize: number
  kvQuantization: KVCachePrecision

  // Multi-GPU parameters (persisted)
  numGPUs: number
  shardingStrategy: ShardingStrategy

  // Offloading parameters (persisted)
  offloadingEnabled: boolean
  offloadTarget: OffloadTarget
  offloadMode: OffloadMode
  offloadPercentage: number
  offloadLayers: number
  kvCacheOffload: boolean

  // Training mode
  mode: 'inference' | 'training'
  trainingMethod: FineTuningMethod
  optimizer: OptimizerType
  trainingPrecision: TrainingPrecision
  loraRank: number
  loraAlpha: number
  targetModulesPercent: number

  // UI preferences (persisted)
  isDarkMode: boolean

  // Actions
  setSelectedModel: (model: Model | null) => void
  setSelectedGPU: (gpu: GPU | null) => void
  setQuantization: (quantization: QuantizationFormat) => void
  setSequenceLength: (sequenceLength: number) => void
  setBatchSize: (batchSize: number) => void
  setKVQuantization: (kvQuantization: KVCachePrecision) => void
  setNumGPUs: (numGPUs: number) => void
  setShardingStrategy: (strategy: ShardingStrategy) => void
  setOffloadingEnabled: (enabled: boolean) => void
  setOffloadTarget: (target: OffloadTarget) => void
  setOffloadMode: (mode: OffloadMode) => void
  setOffloadPercentage: (percentage: number) => void
  setOffloadLayers: (layers: number) => void
  setKVCacheOffload: (enabled: boolean) => void
  setMode: (mode: 'inference' | 'training') => void
  setTrainingMethod: (method: FineTuningMethod) => void
  setOptimizer: (optimizer: OptimizerType) => void
  setTrainingPrecision: (precision: TrainingPrecision) => void
  setLoraRank: (rank: number) => void
  setLoraAlpha: (alpha: number) => void
  setTargetModulesPercent: (percent: number) => void
  toggleDarkMode: () => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      // Default state
      selectedModel: null,
      selectedGPU: null,
      quantization: 'fp16',
      sequenceLength: 4096,
      batchSize: 1,
      kvQuantization: 'fp16',
      numGPUs: 1,
      shardingStrategy: 'tensor-parallel' as ShardingStrategy,
      offloadingEnabled: false,
      offloadTarget: 'cpu-ram' as OffloadTarget,
      offloadMode: 'percentage' as OffloadMode,
      offloadPercentage: 0,
      offloadLayers: 0,
      kvCacheOffload: false,
      mode: 'inference',
      trainingMethod: 'lora',
      optimizer: 'adamw',
      trainingPrecision: 'bf16',
      loraRank: 16,
      loraAlpha: 32,
      targetModulesPercent: 30,
      isDarkMode: false,

      // Actions
      setSelectedModel: (model) => set({ selectedModel: model }),
      setSelectedGPU: (gpu) => set({ selectedGPU: gpu }),
      setQuantization: (quantization) => set({ quantization }),
      setSequenceLength: (sequenceLength) => set({ sequenceLength }),
      setBatchSize: (batchSize) => set({ batchSize }),
      setKVQuantization: (kvQuantization) => set({ kvQuantization }),
      setNumGPUs: (numGPUs) => set({ numGPUs }),
      setShardingStrategy: (strategy) => set({ shardingStrategy: strategy }),
      setOffloadingEnabled: (enabled) => set({ offloadingEnabled: enabled }),
      setOffloadTarget: (target) => set({ offloadTarget: target }),
      setOffloadMode: (mode) => set({ offloadMode: mode }),
      setOffloadPercentage: (percentage) => set({ offloadPercentage: percentage }),
      setOffloadLayers: (layers) => set({ offloadLayers: layers }),
      setKVCacheOffload: (enabled) => set({ kvCacheOffload: enabled }),
      setMode: (mode) => set({ mode }),
      setTrainingMethod: (method) => set({ trainingMethod: method }),
      setOptimizer: (optimizer) => set({ optimizer }),
      setTrainingPrecision: (precision) => set({ trainingPrecision: precision }),
      setLoraRank: (rank) => set({ loraRank: rank }),
      setLoraAlpha: (alpha) => set({ loraAlpha: alpha }),
      setTargetModulesPercent: (percent) => set({ targetModulesPercent: percent }),
      toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
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
