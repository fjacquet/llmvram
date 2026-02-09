import type {
  KVCachePrecision,
  OffloadMode,
  OffloadTarget,
  QuantizationFormat,
  ShardingStrategy,
} from '@engines/types'
import type { GPU, Model } from '@utils/schemas'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

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
      toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
    }),
    {
      name: 'llmvram-ui-preferences',
      // Only persist preferences, not selections
      partialize: (state) => ({
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
        isDarkMode: state.isDarkMode,
      }),
    },
  ),
)
