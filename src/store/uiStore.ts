import type { KVCachePrecision, QuantizationFormat, ShardingStrategy } from '@engines/types'
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
        isDarkMode: state.isDarkMode,
      }),
    },
  ),
)
