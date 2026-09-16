import type { ConfigSnapshot } from '@store/comparisonStore'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ComparisonColumn } from './ComparisonColumn'

// IMPORTANT-2 pin: the store's numGPUs is per-node (see ddcf76b / the
// multi-node fix wave). A comparison column must render cluster topology —
// server count and per-server GPUs — not just the per-node number, and must
// flag numNodes/interNodeFabric differences the same way it flags every
// other config field.
const makeSnapshot = (
  id: string,
  overrides: Partial<ConfigSnapshot['config']> = {},
): ConfigSnapshot => ({
  id,
  label: `Config ${id}`,
  timestamp: Date.now(),
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
    numNodes: 1,
    interNodeFabric: 'ethernet-800g',
    shardingStrategy: 'tensor-parallel',
    offloadingEnabled: false,
    offloadTarget: 'cpu-ram',
    offloadPercentage: 0,
    ...overrides,
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

describe('ComparisonColumn', () => {
  it('shows no GPUs row for a plain single-GPU, single-node config', () => {
    const snapshot = makeSnapshot('a')
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.queryByText('GPUs')).not.toBeInTheDocument()
  })

  it('renders per-node GPU count and strategy for a single-node multi-GPU config', () => {
    const snapshot = makeSnapshot('a', { numGPUs: 8, numNodes: 1 })
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.getByText('8x TP')).toBeInTheDocument()
  })

  it('renders server topology, not just the per-node count, for a multi-node config', () => {
    // 4 servers x 8 GPUs/server = 32 GPUs total. Rendering "8x TP" here would
    // silently drop the server count and misreport the cluster.
    const snapshot = makeSnapshot('a', { numGPUs: 8, numNodes: 4 })
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.getByText('4× 8 GPU servers')).toBeInTheDocument()
    expect(screen.queryByText('8x TP')).not.toBeInTheDocument()
  })

  it('renders a GPUs row even when gpusPerNode is 1 but numNodes > 1 (pure pipeline-parallel)', () => {
    const snapshot = makeSnapshot('a', {
      numGPUs: 1,
      numNodes: 4,
      shardingStrategy: 'pipeline-parallel',
    })
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.getByText('4× 1 GPU servers')).toBeInTheDocument()
  })

  it('flags a numNodes difference as a diff even when numGPUs matches', () => {
    const a = makeSnapshot('a', { numGPUs: 8, numNodes: 1 })
    const b = makeSnapshot('b', { numGPUs: 8, numNodes: 4 })
    render(<ComparisonColumn snapshot={a} allSnapshots={[a, b]} onRemove={() => {}} />)

    const row = screen.getByText('GPUs').closest('div')
    expect(row).not.toBeNull()
    expect(row?.className).toMatch(/bg-amber-50/)
  })

  it('flags an interNodeFabric difference as a diff even when the topology numbers match', () => {
    const a = makeSnapshot('a', { numGPUs: 8, numNodes: 4, interNodeFabric: 'ethernet-800g' })
    const b = makeSnapshot('b', { numGPUs: 8, numNodes: 4, interNodeFabric: 'infiniband-ndr' })
    render(<ComparisonColumn snapshot={a} allSnapshots={[a, b]} onRemove={() => {}} />)

    const row = screen.getByText('GPUs').closest('div')
    expect(row).not.toBeNull()
    expect(row?.className).toMatch(/bg-amber-50/)
  })

  it('does not flag the GPUs row when every field, including topology, matches', () => {
    const a = makeSnapshot('a', { numGPUs: 8, numNodes: 4 })
    const b = makeSnapshot('b', { numGPUs: 8, numNodes: 4 })
    render(<ComparisonColumn snapshot={a} allSnapshots={[a, b]} onRemove={() => {}} />)

    const row = screen.getByText('GPUs').closest('div')
    expect(row).not.toBeNull()
    expect(row?.className).not.toMatch(/bg-amber-50/)
  })
})
