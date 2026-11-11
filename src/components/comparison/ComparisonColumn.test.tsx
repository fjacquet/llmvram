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
  results: Partial<ConfigSnapshot['results']> = {},
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
    tokensPerSecond: 21.2,
    timeToFirstToken: 47,
    ceilingTokensPerSecond: 45,
    ceilingTimeToFirstToken: 22,
    bottleneck: 'memory',
    fits: true,
    perGPUTotal: null,
    utilizationPercent: null,
    ...results,
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

  it('headlines expected speed and TTFT with the theoretical ceiling beside each (spec P3)', () => {
    const snapshot = makeSnapshot('a')
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.getByText('21.2 tok/s')).toBeInTheDocument()
    expect(screen.getByText('Theoretical ceiling: 45.0 tok/s')).toBeInTheDocument()
    expect(screen.getByText('47.0 ms')).toBeInTheDocument()
    expect(screen.getByText('Theoretical ceiling: 22.0 ms')).toBeInTheDocument()
  })

  it('diffs performance on the expected figure, not the ceiling', () => {
    const a = makeSnapshot('a')
    const sameExpected = makeSnapshot('b', {}, { ceilingTokensPerSecond: 99 })
    const { unmount } = render(
      <ComparisonColumn snapshot={a} allSnapshots={[a, sameExpected]} onRemove={() => {}} />,
    )
    expect(screen.getByText('Performance').closest('div')?.className).not.toMatch(/bg-amber-50/)
    unmount()

    const otherExpected = makeSnapshot('c', {}, { tokensPerSecond: 30 })
    render(<ComparisonColumn snapshot={a} allSnapshots={[a, otherExpected]} onRemove={() => {}} />)
    expect(screen.getByText('Performance').closest('div')?.className).toMatch(/bg-amber-50/)
  })

  it('shows Units and Replica for a sizing snapshot, with the KV tier name', () => {
    const snapshot = makeSnapshot(
      'a',
      {},
      {
        sizing: {
          unit: 'rack',
          units: 3,
          replicas: 20,
          tp: 8,
          nodes: 1,
          sessionsPerReplica: 100,
          tier: 'host-pcie',
        },
      },
    )
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.getByText('Units').nextElementSibling).toHaveTextContent('3 racks')
    expect(screen.getByText('Replica').nextElementSibling).toHaveTextContent('TP 8 × 1')
    expect(screen.getByText('KV tier')).toBeInTheDocument()
  })

  it('leaves Units and Replica blank for an inference snapshot', () => {
    const snapshot = makeSnapshot('a')
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    expect(screen.getByText('Units').nextElementSibling).toHaveTextContent('—')
    expect(screen.getByText('Replica').nextElementSibling).toHaveTextContent('—')
    expect(screen.queryByText('KV tier')).not.toBeInTheDocument()
  })

  const sizingResults = {
    sizing: {
      unit: 'rack' as const,
      units: 3,
      replicas: 20,
      tp: 8,
      nodes: 1,
      sessionsPerReplica: 100,
      tier: null,
    },
    modelWeights: 0,
    kvCache: 0,
    activations: 0,
  }

  it('a sizing snapshot shows dashes for the component rows, never 0.00 GB', () => {
    const snapshot = makeSnapshot('a', {}, sizingResults)
    render(<ComparisonColumn snapshot={snapshot} allSnapshots={[snapshot]} onRemove={() => {}} />)
    for (const label of ['Weights:', 'KV Cache:', 'Activations:']) {
      expect(screen.getByText(label).nextElementSibling).toHaveTextContent('—')
    }
    expect(screen.queryByText('0.00 GB')).not.toBeInTheDocument()
    expect(screen.getByText('Total VRAM (per replica)')).toBeInTheDocument()
  })

  it('labels sizing speed per user and never diffs it against an inference snapshot', () => {
    const sizing = makeSnapshot('a', {}, { ...sizingResults, tokensPerSecond: 99 })
    const inference = makeSnapshot('b')
    render(
      <ComparisonColumn snapshot={sizing} allSnapshots={[sizing, inference]} onRemove={() => {}} />,
    )
    expect(screen.getByText('Per-user TTFT:')).toBeInTheDocument()
    expect(screen.getByText('Per-user speed').closest('div')?.className).not.toMatch(/bg-amber-50/)
  })
})
