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

interface BarTooltipProps {
  active?: boolean
  payload?: Array<{
    name: string
    value: number
    color: string
  }>
  label?: string
  totalPerGPU: number
}

function BarChartTooltip({ active, payload, label, totalPerGPU }: BarTooltipProps) {
  if (!active || !payload || payload.length === 0) return null

  return (
    <div className="rounded-lg border border-gray-200 bg-white/95 px-3 py-2 shadow-lg backdrop-blur-sm dark:border-gray-700 dark:bg-gray-800/95">
      <div className="mb-1.5 text-sm font-semibold text-gray-900 dark:text-gray-100">{label}</div>
      {payload.map((entry) => {
        const pct = totalPerGPU > 0 ? ((entry.value / totalPerGPU) * 100).toFixed(0) : '0'
        return (
          <div key={entry.name} className="flex items-center gap-2 py-0.5 text-xs">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-gray-700 dark:text-gray-300">{entry.name}</span>
            <span className="ml-auto font-medium text-gray-900 dark:text-gray-100">
              {entry.value.toFixed(2)} GB
            </span>
            <span className="text-gray-500 dark:text-gray-500">({pct}%)</span>
          </div>
        )
      })}
      <div className="mt-1.5 border-t border-gray-200 pt-1.5 text-xs font-medium text-gray-900 dark:border-gray-700 dark:text-gray-100">
        Total: {totalPerGPU.toFixed(2)} GB
      </div>
    </div>
  )
}

interface CustomLegendProps {
  payload?: Array<{ value: string; color: string }>
}

function BarChartLegend({ payload }: CustomLegendProps) {
  if (!payload) return null
  return (
    <div className="mt-1 flex flex-wrap justify-center gap-x-4 gap-y-1">
      {payload.map((entry) => (
        <div key={entry.value} className="flex items-center gap-1.5 text-xs">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-gray-700 dark:text-gray-300">{entry.value}</span>
        </div>
      ))}
    </div>
  )
}

export function MultiGPUBreakdownChart({ breakdown, gpuVRAM }: MultiGPUBreakdownChartProps) {
  const isDarkMode = useUIStore((s) => s.isDarkMode)

  // Create data array with one entry per GPU
  const data = Array.from({ length: breakdown.numGPUs }, (_, i) => ({
    name: `GPU ${i + 1}`,
    'Model Weights': breakdown.perGPU.modelWeights.toNumber(),
    'KV Cache': breakdown.perGPU.kvCache.toNumber(),
    Activations: breakdown.perGPU.activations.toNumber(),
    'Framework & NCCL': breakdown.perGPU.frameworkOverhead.toNumber(),
    Communication: breakdown.perGPU.communicationOverhead.toNumber(),
  }))

  const totalPerGPU = breakdown.totalPerGPU.toNumber()

  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-gray-900 dark:text-white">
        Multi-GPU Memory Distribution
      </h4>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <defs>
            {BAR_KEYS.map((key) => (
              <linearGradient
                key={key}
                id={`grad-${key.replace(/\s+/g, '-')}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={COLORS[key]} stopOpacity={0.9} />
                <stop offset="100%" stopColor={COLORS[key]} stopOpacity={0.65} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={isDarkMode ? '#374151' : '#e5e7eb'}
            strokeOpacity={0.5}
            vertical={false}
          />
          <XAxis
            dataKey="name"
            stroke={isDarkMode ? '#9ca3af' : '#6b7280'}
            style={{ fontSize: '12px' }}
            tickLine={false}
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
            tickLine={false}
            axisLine={false}
          />
          <Tooltip
            content={<BarChartTooltip totalPerGPU={totalPerGPU} />}
            cursor={{ fill: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)' }}
          />
          <Legend content={<BarChartLegend />} />
          <ReferenceLine
            y={gpuVRAM}
            stroke="#dc2626"
            strokeDasharray="6 3"
            strokeWidth={1.5}
            label={{
              value: `Capacity: ${gpuVRAM} GB`,
              position: 'top',
              fill: isDarkMode ? '#fca5a5' : '#dc2626',
              fontSize: 11,
              fontWeight: 500,
            }}
          />
          {BAR_KEYS.map((key, idx) => (
            <Bar
              key={key}
              dataKey={key}
              stackId="a"
              fill={`url(#grad-${key.replace(/\s+/g, '-')})`}
              radius={idx === BAR_KEYS.length - 1 ? [4, 4, 0, 0] : undefined}
            />
          ))}
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
