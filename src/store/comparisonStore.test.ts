import { beforeEach, describe, expect, it } from 'vitest'
import type { ConfigSnapshot } from './comparisonStore'
import { useComparisonStore } from './comparisonStore'

const makeSnapshot = (label: string): Omit<ConfigSnapshot, 'id' | 'timestamp'> => ({
  label,
  config: {
    modelName: 'LLaMA 3.1 70B',
    modelId: 'llama-3.1-70b',
    gpuName: 'NVIDIA H100 80GB SXM',
    gpuId: 'nvidia-h100-80gb-sxm',
    gpuVramGb: 80,
    quantization: 'fp16',
    sequenceLength: 2048,
    batchSize: 1,
    kvQuantization: 'fp16',
    numGPUs: 1,
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    offloadTarget: 'cpu',
    offloadPercentage: 0,
  },
  results: {
    totalVRAM: 42,
    modelWeights: 39,
    kvCache: 1.25,
    activations: 0.75,
    frameworkOverhead: 1,
    tokensPerSecond: 45,
    timeToFirstToken: 22,
    bottleneck: 'memory',
    fits: true,
    perGPUTotal: null,
    utilizationPercent: null,
  },
})

describe('comparisonStore', () => {
  beforeEach(() => {
    useComparisonStore.setState({ snapshots: [] })
  })

  it('should start with empty snapshots', () => {
    const { snapshots } = useComparisonStore.getState()
    expect(snapshots).toHaveLength(0)
  })

  it('should add a snapshot with generated id and timestamp', () => {
    useComparisonStore.getState().addSnapshot(makeSnapshot('Config 1'))
    const { snapshots } = useComparisonStore.getState()
    expect(snapshots).toHaveLength(1)
    expect(snapshots[0]?.label).toBe('Config 1')
    expect(snapshots[0]?.id).toBeDefined()
    expect(snapshots[0]?.timestamp).toBeGreaterThan(0)
  })

  it('should evict oldest when adding beyond maxSnapshots (3)', () => {
    const store = useComparisonStore.getState()
    store.addSnapshot(makeSnapshot('First'))
    store.addSnapshot(makeSnapshot('Second'))
    store.addSnapshot(makeSnapshot('Third'))

    expect(useComparisonStore.getState().snapshots).toHaveLength(3)

    store.addSnapshot(makeSnapshot('Fourth'))

    const { snapshots } = useComparisonStore.getState()
    expect(snapshots).toHaveLength(3)
    // First snapshot should have been evicted
    expect(snapshots[0]?.label).toBe('Second')
    expect(snapshots[2]?.label).toBe('Fourth')
  })

  it('should remove a snapshot by id', () => {
    const store = useComparisonStore.getState()
    store.addSnapshot(makeSnapshot('Config 1'))
    store.addSnapshot(makeSnapshot('Config 2'))

    const idToRemove = useComparisonStore.getState().snapshots[0]?.id ?? ''
    expect(idToRemove).not.toBe('')

    store.removeSnapshot(idToRemove)
    const { snapshots } = useComparisonStore.getState()
    expect(snapshots).toHaveLength(1)
    expect(snapshots[0]?.label).toBe('Config 2')
  })

  it('should update a snapshot label', () => {
    const store = useComparisonStore.getState()
    store.addSnapshot(makeSnapshot('Config 1'))

    const id = useComparisonStore.getState().snapshots[0]?.id ?? ''
    expect(id).not.toBe('')

    store.updateLabel(id, 'Renamed Config')
    expect(useComparisonStore.getState().snapshots[0]?.label).toBe('Renamed Config')
  })

  it('should clear all snapshots', () => {
    const store = useComparisonStore.getState()
    store.addSnapshot(makeSnapshot('Config 1'))
    store.addSnapshot(makeSnapshot('Config 2'))
    expect(useComparisonStore.getState().snapshots).toHaveLength(2)

    store.clearAll()
    expect(useComparisonStore.getState().snapshots).toHaveLength(0)
  })
})
