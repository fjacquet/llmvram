import type { LoRAVRAMBreakdown, TrainingVRAMBreakdown } from '@engines/types'
import { useUIStore } from '@store/uiStore'
import { useCallback, useMemo } from 'react'
import type { PieLabelRenderProps } from 'recharts'
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

interface TrainingBreakdownChartProps {
  breakdown: TrainingVRAMBreakdown | LoRAVRAMBreakdown
}

const TRAINING_COLORS = {
  modelWeights: '#3b82f6', // blue-500
  baseWeights: '#3b82f6', // blue-500 (LoRA base)
  adapterWeights: '#06b6d4', // cyan-500 (LoRA adapters)
  masterWeights: '#8b5cf6', // violet-500
  gradients: '#ef4444', // red-500
  optimizerStates: '#f59e0b', // amber-500
  activations: '#10b981', // emerald-500
  frameworkOverhead: '#6b7280', // gray-500
}

export function TrainingBreakdownChart({ breakdown }: TrainingBreakdownChartProps) {
  const isDarkMode = useUIStore((s) => s.isDarkMode)

  // Type guard to check if result is LoRA/QLoRA
  const isLoRA = 'baseWeights' in breakdown

  // Convert Decimal values to numbers for Recharts (memoized for performance)
  const data = useMemo(() => {
    const result = []

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

  // Custom label renderer showing GB value on segments (memoized to prevent Recharts re-renders)
  const renderLabel = useCallback(
    (props: PieLabelRenderProps) => {
      // Guard against undefined values
      const { value, cx, cy, midAngle, innerRadius, outerRadius } = props
      if (
        typeof value !== 'number' ||
        typeof cx !== 'number' ||
        typeof cy !== 'number' ||
        typeof midAngle !== 'number' ||
        typeof innerRadius !== 'number' ||
        typeof outerRadius !== 'number'
      ) {
        return null
      }

      const RADIAN = Math.PI / 180
      const radius = innerRadius + (outerRadius - innerRadius) * 0.5
      const x = cx + radius * Math.cos(-midAngle * RADIAN)
      const y = cy + radius * Math.sin(-midAngle * RADIAN)

      // Only show label if value is significant (>1% of total)
      if (value < totalGB * 0.01) return null

      return (
        <text
          x={x}
          y={y}
          fill="white"
          textAnchor={x > cx ? 'start' : 'end'}
          dominantBaseline="central"
          className="text-xs font-medium"
        >
          {value.toFixed(1)}
        </text>
      )
    },
    [totalGB],
  )

  // Custom center label showing total
  const renderCenterLabel = () => {
    return (
      <g>
        <text
          x="50%"
          y="48%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-xs font-medium fill-gray-500 dark:fill-gray-400"
        >
          Total
        </text>
        <text
          x="50%"
          y="52%"
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-lg font-bold fill-gray-900 dark:fill-gray-100"
        >
          {totalGB.toFixed(2)} GB
        </text>
      </g>
    )
  }

  // Memoized tooltip formatter
  const tooltipFormatter = useCallback((value: unknown) => {
    if (typeof value === 'number') {
      return `${value.toFixed(2)} GB`
    }
    return ''
  }, [])

  return (
    <div
      role="img"
      aria-label="Training memory breakdown chart"
      className="text-gray-900 dark:text-gray-100"
    >
      <div className="w-full h-72 sm:h-80">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={renderLabel}
              outerRadius={100}
              innerRadius={60}
              fill="#8884d8"
              dataKey="value"
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={tooltipFormatter}
              contentStyle={{
                backgroundColor: isDarkMode
                  ? 'rgba(31, 41, 55, 0.95)'
                  : 'rgba(255, 255, 255, 0.95)',
                border: isDarkMode ? '1px solid #374151' : '1px solid #e5e7eb',
                borderRadius: '0.375rem',
                color: isDarkMode ? '#f3f4f6' : '#111827',
              }}
            />
            <Legend
              verticalAlign="bottom"
              height={36}
              iconType="circle"
              wrapperStyle={{
                paddingTop: '1rem',
                color: isDarkMode ? '#d1d5db' : '#374151',
              }}
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
