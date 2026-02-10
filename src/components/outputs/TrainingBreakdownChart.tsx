import type { LoRAVRAMBreakdown, TrainingVRAMBreakdown } from '@engines/types'
import { useUIStore } from '@store/uiStore'
import { useCallback, useMemo } from 'react'
import type { PieLabelRenderProps } from 'recharts'
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

interface TrainingBreakdownChartProps {
  breakdown: TrainingVRAMBreakdown | LoRAVRAMBreakdown
}

const TRAINING_COLORS = {
  modelWeights: '#6366f1', // indigo-500
  baseWeights: '#6366f1', // indigo-500 (LoRA base)
  adapterWeights: '#06b6d4', // cyan-500 (LoRA adapters)
  masterWeights: '#a78bfa', // violet-400
  gradients: '#ef4444', // red-500
  optimizerStates: '#f59e0b', // amber-500
  activations: '#10b981', // emerald-500
  frameworkOverhead: '#6b7280', // gray-500
}

interface ChartDataEntry {
  name: string
  value: number
  color: string
}

interface CustomTooltipProps {
  active?: boolean
  payload?: Array<{ payload: ChartDataEntry; value: number }>
  totalGB: number
}

function ChartTooltip({ active, payload, totalGB }: CustomTooltipProps) {
  if (!active || !payload || payload.length === 0) return null
  const entry = payload[0]
  if (!entry) return null
  const pct = totalGB > 0 ? ((entry.value / totalGB) * 100).toFixed(1) : '0.0'

  return (
    <div className="rounded-lg border border-gray-200 bg-white/95 px-3 py-2 shadow-lg backdrop-blur-sm dark:border-gray-700 dark:bg-gray-800/95">
      <div className="flex items-center gap-2">
        <span
          className="inline-block h-3 w-3 rounded-full"
          style={{ backgroundColor: entry.payload.color }}
        />
        <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
          {entry.payload.name}
        </span>
      </div>
      <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">
        {entry.value.toFixed(2)} GB ({pct}%)
      </div>
    </div>
  )
}

interface CustomLegendProps {
  payload?: Array<{ value: string; color: string }>
  data: ChartDataEntry[]
  totalGB: number
}

function ChartLegend({ data, totalGB }: CustomLegendProps) {
  return (
    <div className="mt-2 flex flex-wrap justify-center gap-x-4 gap-y-1.5">
      {data
        .filter((d) => d.value > 0)
        .map((entry) => {
          const pct = totalGB > 0 ? ((entry.value / totalGB) * 100).toFixed(0) : '0'
          return (
            <div key={entry.name} className="flex items-center gap-1.5 text-xs">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-gray-700 dark:text-gray-300">{entry.name}</span>
              <span className="font-medium text-gray-900 dark:text-gray-100">
                {entry.value.toFixed(1)} GB
              </span>
              <span className="text-gray-500 dark:text-gray-500">({pct}%)</span>
            </div>
          )
        })}
    </div>
  )
}

export function TrainingBreakdownChart({ breakdown }: TrainingBreakdownChartProps) {
  const isDarkMode = useUIStore((s) => s.isDarkMode)

  // Type guard to check if result is LoRA/QLoRA
  const isLoRA = 'baseWeights' in breakdown

  // Convert Decimal values to numbers for Recharts (memoized for performance)
  const data: ChartDataEntry[] = useMemo(() => {
    const result: ChartDataEntry[] = []

    if (isLoRA) {
      const loraBreakdown = breakdown as LoRAVRAMBreakdown
      result.push(
        {
          name: 'Base Model Weights',
          value: loraBreakdown.baseWeights.toNumber(),
          color: TRAINING_COLORS.baseWeights,
        },
        {
          name: 'Adapter Weights',
          value: loraBreakdown.adapterWeights.toNumber(),
          color: TRAINING_COLORS.adapterWeights,
        },
      )
    } else {
      result.push({
        name: 'Model Weights',
        value: breakdown.modelWeights.toNumber(),
        color: TRAINING_COLORS.modelWeights,
      })
    }

    // Conditionally add master weights only when > 0
    if (breakdown.masterWeights.greaterThan(0)) {
      result.push({
        name: 'Master Weights (FP32)',
        value: breakdown.masterWeights.toNumber(),
        color: TRAINING_COLORS.masterWeights,
      })
    }

    // Always add remaining components
    result.push(
      {
        name: 'Gradients',
        value: breakdown.gradients.toNumber(),
        color: TRAINING_COLORS.gradients,
      },
      {
        name: 'Optimizer States',
        value: breakdown.optimizerStates.toNumber(),
        color: TRAINING_COLORS.optimizerStates,
      },
      {
        name: 'Activations',
        value: breakdown.activations.toNumber(),
        color: TRAINING_COLORS.activations,
      },
      {
        name: 'Framework Overhead',
        value: breakdown.frameworkOverhead.toNumber(),
        color: TRAINING_COLORS.frameworkOverhead,
      },
    )

    return result
  }, [breakdown, isLoRA])

  const totalGB = breakdown.total.toNumber()
  const strokeColor = isDarkMode ? '#1f2937' : '#ffffff'

  // External label with value and percentage (memoized to prevent Recharts re-renders)
  const renderLabel = useCallback(
    (props: PieLabelRenderProps) => {
      const { value, cx, cy, midAngle, outerRadius, percent } = props
      if (
        typeof value !== 'number' ||
        typeof cx !== 'number' ||
        typeof cy !== 'number' ||
        typeof midAngle !== 'number' ||
        typeof outerRadius !== 'number' ||
        typeof percent !== 'number'
      ) {
        return null
      }

      // Only show label if segment is >2% of total
      if (percent < 0.02) return null

      const RADIAN = Math.PI / 180
      const radius = outerRadius + 20
      const x = cx + radius * Math.cos(-midAngle * RADIAN)
      const y = cy + radius * Math.sin(-midAngle * RADIAN)

      return (
        <text
          x={x}
          y={y}
          fill={isDarkMode ? '#d1d5db' : '#374151'}
          textAnchor={x > cx ? 'start' : 'end'}
          dominantBaseline="central"
          className="text-xs font-medium"
        >
          {value.toFixed(1)} GB ({(percent * 100).toFixed(0)}%)
        </text>
      )
    },
    [isDarkMode],
  )

  // Custom center label showing total
  const renderCenterLabel = () => {
    return (
      <g>
        <text
          x="50%"
          y="44%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-[10px] font-medium uppercase tracking-wider fill-gray-400 dark:fill-gray-500"
        >
          Total
        </text>
        <text
          x="50%"
          y="54%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-xl font-bold fill-gray-900 dark:fill-gray-100"
        >
          {totalGB.toFixed(1)} GB
        </text>
      </g>
    )
  }

  return (
    <div
      role="img"
      aria-label="Training memory breakdown chart"
      className="text-gray-900 dark:text-gray-100"
    >
      <div className="w-full h-80 sm:h-88">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="45%"
              labelLine
              label={renderLabel}
              outerRadius={120}
              innerRadius={72}
              paddingAngle={3}
              dataKey="value"
              animationDuration={600}
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} stroke={strokeColor} strokeWidth={2} />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip totalGB={totalGB} />} cursor={false} />
            <Legend
              content={<ChartLegend data={data} totalGB={totalGB} />}
              verticalAlign="bottom"
            />
            {renderCenterLabel()}
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Screen reader accessible text alternative */}
      <div className="sr-only">
        Training memory breakdown:{' '}
        {isLoRA
          ? `Base Model Weights ${(breakdown as LoRAVRAMBreakdown).baseWeights.toFixed(2)} GB, Adapter Weights ${(breakdown as LoRAVRAMBreakdown).adapterWeights.toFixed(2)} GB, `
          : `Model Weights ${breakdown.modelWeights.toFixed(2)} GB, `}
        {breakdown.masterWeights.greaterThan(0) &&
          `Master Weights ${breakdown.masterWeights.toFixed(2)} GB, `}
        Gradients {breakdown.gradients.toFixed(2)} GB, Optimizer States{' '}
        {breakdown.optimizerStates.toFixed(2)} GB, Activations {breakdown.activations.toFixed(2)}{' '}
        GB, Framework Overhead {breakdown.frameworkOverhead.toFixed(2)} GB, Total{' '}
        {totalGB.toFixed(2)} GB
      </div>
    </div>
  )
}
