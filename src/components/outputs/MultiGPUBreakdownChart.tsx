import type { MultiGPUVRAMBreakdown } from '@engines/types'
import { useUIStore } from '@store/uiStore'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface MultiGPUBreakdownChartProps {
  breakdown: MultiGPUVRAMBreakdown
  gpuVRAM: number
}

/**
 * Stacked bar chart showing per-GPU VRAM breakdown
 *
 * Displays memory distribution across multiple GPUs with stacked components:
 * - Model Weights (blue)
 * - KV Cache (emerald)
 * - Activations (amber)
 * - Framework & NCCL (violet)
 * - Communication (red)
 *
 * Includes a reference line at GPU VRAM capacity and summary statistics.
 */
export function MultiGPUBreakdownChart({ breakdown, gpuVRAM }: MultiGPUBreakdownChartProps) {
  const isDarkMode = useUIStore((s) => s.isDarkMode)

  // Create data array with one entry per GPU
  // All GPUs have identical memory in both TP and PP strategies
  const data = Array.from({ length: breakdown.numGPUs }, (_, i) => ({
    name: `GPU ${i + 1}`,
    'Model Weights': breakdown.perGPU.modelWeights.toNumber(),
    'KV Cache': breakdown.perGPU.kvCache.toNumber(),
    Activations: breakdown.perGPU.activations.toNumber(),
    'Framework & NCCL': breakdown.perGPU.frameworkOverhead.toNumber(),
    Communication: breakdown.perGPU.communicationOverhead.toNumber(),
  }))

  // Colors for each component
  const colors = {
    'Model Weights': '#3b82f6', // blue
    'KV Cache': '#10b981', // emerald
    Activations: '#f59e0b', // amber
    'Framework & NCCL': '#8b5cf6', // violet
    Communication: '#ef4444', // red
  }

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-gray-900 dark:text-white">
        Multi-GPU Memory Distribution
      </h4>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={isDarkMode ? '#374151' : '#e5e7eb'}
            vertical={false}
          />
          <XAxis
            dataKey="name"
            stroke={isDarkMode ? '#9ca3af' : '#6b7280'}
            style={{ fontSize: '12px' }}
          />
          <YAxis
            label={{
              value: 'VRAM (GB)',
              angle: -90,
              position: 'insideLeft',
              style: { fill: isDarkMode ? '#9ca3af' : '#6b7280', fontSize: '12px' },
            }}
            stroke={isDarkMode ? '#9ca3af' : '#6b7280'}
            style={{ fontSize: '12px' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: isDarkMode ? '#1f2937' : '#ffffff',
              border: `1px solid ${isDarkMode ? '#374151' : '#e5e7eb'}`,
              borderRadius: '0.5rem',
              color: isDarkMode ? '#f3f4f6' : '#1f2937',
            }}
            labelStyle={{
              color: isDarkMode ? '#f3f4f6' : '#1f2937',
              fontWeight: 600,
            }}
            formatter={(value?: number) => (value ? `${value.toFixed(2)} GB` : '0.00 GB')}
          />
          <Legend
            wrapperStyle={{
              fontSize: '12px',
              color: isDarkMode ? '#d1d5db' : '#4b5563',
            }}
          />
          <ReferenceLine
            y={gpuVRAM}
            stroke="#dc2626"
            strokeDasharray="3 3"
            label={{
              value: `GPU Capacity: ${gpuVRAM} GB`,
              position: 'top',
              fill: isDarkMode ? '#fca5a5' : '#dc2626',
              fontSize: 11,
            }}
          />
          <Bar dataKey="Model Weights" stackId="a" fill={colors['Model Weights']} />
          <Bar dataKey="KV Cache" stackId="a" fill={colors['KV Cache']} />
          <Bar dataKey="Activations" stackId="a" fill={colors.Activations} />
          <Bar dataKey="Framework & NCCL" stackId="a" fill={colors['Framework & NCCL']} />
          <Bar dataKey="Communication" stackId="a" fill={colors.Communication} />
        </BarChart>
      </ResponsiveContainer>

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
      </div>
    </div>
  )
}
