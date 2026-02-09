import { create } from 'zustand'

export interface ConfigSnapshot {
  id: string // crypto.randomUUID()
  label: string // User-editable label ("Config 1", etc.)
  timestamp: number // Date.now()

  // Configuration inputs (what the user selected)
  config: {
    modelName: string // Display name
    modelId: string // ID for identification
    gpuName: string
    gpuId: string
    gpuVramGb: number
    quantization: string
    sequenceLength: number
    batchSize: number
    kvQuantization: string
    numGPUs: number
    shardingStrategy: string
    offloadingEnabled: boolean
    offloadTarget: string
    offloadPercentage: number
  }

  // Cached calculation results (plain numbers, NOT Decimal.js)
  results: {
    totalVRAM: number // GB - total VRAM required
    modelWeights: number // GB
    kvCache: number // GB
    activations: number // GB
    frameworkOverhead: number // GB
    tokensPerSecond: number
    timeToFirstToken: number // ms
    bottleneck: string // 'compute' | 'memory' | 'balanced'
    fits: boolean // Whether model fits on GPU(s)
    // Multi-GPU (if applicable)
    perGPUTotal: number | null // Per-GPU VRAM if multi-GPU
    utilizationPercent: number | null
  }
}

interface ComparisonState {
  snapshots: ConfigSnapshot[]
  maxSnapshots: 3 // Constant

  // Actions
  addSnapshot: (snapshot: Omit<ConfigSnapshot, 'id' | 'timestamp'>) => void
  removeSnapshot: (id: string) => void
  updateLabel: (id: string, label: string) => void
  clearAll: () => void
}

export const useComparisonStore = create<ComparisonState>((set) => ({
  snapshots: [],
  maxSnapshots: 3,

  // Actions
  addSnapshot: (snapshot) =>
    set((state) => {
      const newSnapshot: ConfigSnapshot = {
        ...snapshot,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
      }

      // If we're at max capacity, remove the oldest snapshot (index 0)
      const snapshots =
        state.snapshots.length >= state.maxSnapshots
          ? [...state.snapshots.slice(1), newSnapshot]
          : [...state.snapshots, newSnapshot]

      return { snapshots }
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
