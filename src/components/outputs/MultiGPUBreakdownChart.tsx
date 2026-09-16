import type { MultiGPUVRAMBreakdown } from '@engines/types'

interface MultiGPUBreakdownChartProps {
  breakdown: MultiGPUVRAMBreakdown
  gpuVRAM: number
  /** Effective tokens/sec after multi-GPU scaling (optional, shown in summary) */
  effectiveTokPerSec?: number
}

const COLORS = {
  'Model Weights': '#6366f1', // indigo-500
  'KV Cache': '#10b981', // emerald-500
  Activations: '#f59e0b', // amber-500
  'Framework & NCCL': '#8b5cf6', // violet-500
  Communication: '#ef4444', // red-500
} as const

const BAR_KEYS = [
  'Model Weights',
  'KV Cache',
  'Activations',
  'Framework & NCCL',
  'Communication',
] as const

// Distinct from the Communication segment color (red-500) so the "everything
// is one color" over-capacity state reads differently from the normal palette.
const OVER_CAPACITY_COLOR = '#dc2626' // red-600

interface CustomLegendProps {
  keys: readonly string[]
  colors: Record<string, string>
}

function BarChartLegend({ keys, colors }: CustomLegendProps) {
  return (
    <div className="mt-1 flex flex-wrap justify-center gap-x-4 gap-y-1">
      {keys.map((key) => (
        <div key={key} className="flex items-center gap-1.5 text-xs">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ backgroundColor: colors[key] }}
          />
          <span className="text-gray-700 dark:text-gray-300">{key}</span>
        </div>
      ))}
    </div>
  )
}

export function MultiGPUBreakdownChart({
  breakdown,
  gpuVRAM,
  effectiveTokPerSec,
}: MultiGPUBreakdownChartProps) {
  const segmentValues: Record<(typeof BAR_KEYS)[number], number> = {
    'Model Weights': breakdown.perGPU.modelWeights.toNumber(),
    'KV Cache': breakdown.perGPU.kvCache.toNumber(),
    Activations: breakdown.perGPU.activations.toNumber(),
    'Framework & NCCL': breakdown.perGPU.frameworkOverhead.toNumber(),
    Communication: breakdown.perGPU.communicationOverhead.toNumber(),
  }

  const totalPerGPU = breakdown.totalPerGPU.toNumber()
  const overCapacity = gpuVRAM > 0 && totalPerGPU > gpuVRAM
  const utilizationPercent = gpuVRAM > 0 ? (totalPerGPU / gpuVRAM) * 100 : 0

  // Under capacity, each segment's width is a percent of gpuVRAM (capacity),
  // so the filled length IS the utilization and the gap IS the headroom.
  // Over capacity, there is no gap to show — segments are renormalized as a
  // percent of totalPerGPU so the meter saturates at exactly 100%.
  const widthBasis = overCapacity ? totalPerGPU : gpuVRAM
  const widthPercent = (value: number): number => (widthBasis > 0 ? (value / widthBasis) * 100 : 0)

  const totalGPUs = breakdown.numGPUs
  const headroom = gpuVRAM - totalPerGPU

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-gray-900 dark:text-white">
        Multi-GPU Memory Distribution
      </h4>

      <div className="space-y-1.5">
        <div className="flex items-baseline justify-between text-sm font-medium text-gray-900 dark:text-white">
          <span>
            Per GPU — {totalPerGPU.toFixed(1)} / {gpuVRAM} GB — {Math.round(utilizationPercent)}%
            used
          </span>
        </div>

        {/* The header line above is already the visible+accessible summary; this
            container stays a plain div (no role/aria-label) so the per-segment
            role="img" elements below remain reachable — role="img" has
            presentational children, so a label here would shadow theirs. */}
        <div className="h-6 w-full overflow-hidden rounded-md bg-gray-200 dark:bg-gray-700">
          <div className="flex h-full">
            {BAR_KEYS.map((key) => (
              <div
                key={key}
                role="img"
                className="h-full transition-all duration-300"
                style={{
                  width: `${widthPercent(segmentValues[key])}%`,
                  backgroundColor: overCapacity ? OVER_CAPACITY_COLOR : COLORS[key],
                }}
                title={`${key}: ${segmentValues[key].toFixed(2)} GB`}
                aria-label={`${key}: ${segmentValues[key].toFixed(2)} GB`}
              />
            ))}
          </div>
        </div>

        <p
          className={
            overCapacity
              ? 'text-xs font-medium text-red-600 dark:text-red-400'
              : 'text-xs text-gray-500 dark:text-gray-400'
          }
        >
          {overCapacity
            ? `${Math.abs(headroom).toFixed(1)} GB over`
            : `${headroom.toFixed(1)} GB headroom`}
        </p>
      </div>

      <BarChartLegend keys={BAR_KEYS} colors={COLORS} />

      <p className="text-xs text-gray-600 dark:text-gray-400">
        identical across all {totalGPUs} GPU{totalGPUs === 1 ? '' : 's'}
        {totalGPUs > 1 &&
          ` (${breakdown.gpusPerNode} per node × ${breakdown.numNodes} node${
            breakdown.numNodes === 1 ? '' : 's'
          })`}
      </p>

      {/* Summary text */}
      <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
        <p>
          <span className="font-medium">Strategy:</span>{' '}
          {breakdown.strategy === 'tensor-parallel' ? 'Tensor Parallel' : 'Pipeline Parallel'}
        </p>
        <p>
          <span className="font-medium">Per-GPU Total:</span> {breakdown.totalPerGPU.toFixed(2)} GB
        </p>
        <p>
          <span className="font-medium">Utilization:</span>{' '}
          {breakdown.utilizationPercent.toFixed(1)}%
        </p>
        {breakdown.strategy === 'tensor-parallel' && breakdown.interconnectBandwidthGBps > 0 && (
          <p>
            <span className="font-medium">Interconnect:</span> {breakdown.interconnectBandwidthGBps}{' '}
            GB/s · {Math.round(breakdown.scalingEfficiency * 100)}% TP scaling efficiency
          </p>
        )}
        {effectiveTokPerSec !== undefined && breakdown.numGPUs > 1 && (
          <p>
            <span className="font-medium">Effective throughput:</span> ~
            {effectiveTokPerSec.toFixed(0)} tok/s ({breakdown.numGPUs}× GPUs ×{' '}
            {Math.round(breakdown.scalingEfficiency * 100)}% efficiency)
          </p>
        )}
      </div>

      {breakdown.numNodes > 1 && (
        <div className="mt-3 text-xs text-gray-600 dark:text-gray-400 space-y-1">
          {/* The counts live in the meter footer above; this line carries only
              what that footer does not say, namely how the work is split. */}
          <p>
            {breakdown.strategy === 'tensor-parallel'
              ? 'Tensor parallel within each server, pipeline parallel across them.'
              : 'Pipeline parallel across all GPUs.'}
          </p>
          <p>
            Efficiency (modelled from bandwidth, not measured):{' '}
            {(breakdown.intraNodeEfficiency * 100).toFixed(0)}% intra-server ·{' '}
            {(breakdown.interNodePrefillEfficiency * 100).toFixed(0)}% inter-server on prefill ·{' '}
            {(breakdown.interNodeDecodeEfficiency * 100).toFixed(0)}% on decode ·{' '}
            {(breakdown.bubbleEfficiency * 100).toFixed(0)}% pipeline fill
          </p>
        </div>
      )}
    </div>
  )
}
