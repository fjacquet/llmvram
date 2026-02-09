import type { KVCachePrecision, QuantizationFormat } from '@engines/types'
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

  // UI preferences (persisted)
  isDarkMode: boolean

  // Actions
  setSelectedModel: (model: Model | null) => void
  setSelectedGPU: (gpu: GPU | null) => void
  setQuantization: (quantization: QuantizationFormat) => void
  setSequenceLength: (sequenceLength: number) => void
  setBatchSize: (batchSize: number) => void
  setKVQuantization: (kvQuantization: KVCachePrecision) => void
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
      isDarkMode: false,

      // Actions
      setSelectedModel: (model) => set({ selectedModel: model }),
      setSelectedGPU: (gpu) => set({ selectedGPU: gpu }),
      setQuantization: (quantization) => set({ quantization }),
      setSequenceLength: (sequenceLength) => set({ sequenceLength }),
      setBatchSize: (batchSize) => set({ batchSize }),
      setKVQuantization: (kvQuantization) => set({ kvQuantization }),
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
        isDarkMode: state.isDarkMode,
      }),
    },
  ),
)
