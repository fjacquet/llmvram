import type { InferenceVRAMBreakdown } from '@engines/types'
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

interface VRAMBreakdownChartProps {
  breakdown: InferenceVRAMBreakdown
}

const COLORS = {
  modelWeights: '#3b82f6', // blue-500
  kvCache: '#10b981', // emerald-500
  activations: '#f59e0b', // amber-500
  frameworkOverhead: '#8b5cf6', // violet-500
}

export function VRAMBreakdownChart({ breakdown }: VRAMBreakdownChartProps) {
  // Convert Decimal values to numbers for Recharts
  const data = [
    {
      name: 'Model Weights',
      value: breakdown.modelWeights.toNumber(),
      color: COLORS.modelWeights,
    },
    {
      name: 'KV Cache',
      value: breakdown.kvCache.toNumber(),
      color: COLORS.kvCache,
    },
    {
      name: 'Activations',
      value: breakdown.activations.toNumber(),
      color: COLORS.activations,
    },
    {
      name: 'Framework Overhead',
      value: breakdown.frameworkOverhead.toNumber(),
      color: COLORS.frameworkOverhead,
    },
  ]

  const totalGB = breakdown.total.toNumber()

  // Custom label renderer showing GB value on segments
  const renderLabel = (entry: {
    value: number
    cx: number
    cy: number
    midAngle: number
    innerRadius: number
    outerRadius: number
  }) => {
    const RADIAN = Math.PI / 180
    const radius = entry.innerRadius + (entry.outerRadius - entry.innerRadius) * 0.5
    const x = entry.cx + radius * Math.cos(-entry.midAngle * RADIAN)
    const y = entry.cy + radius * Math.sin(-entry.midAngle * RADIAN)

    // Only show label if value is significant (>1% of total)
    if (entry.value < totalGB * 0.01) return null

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > entry.cx ? 'start' : 'end'}
        dominantBaseline="central"
        className="text-xs font-medium"
      >
        {`${entry.value.toFixed(1)}`}
      </text>
    )
  }

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

  return (
    <div
      role="img"
      aria-label="VRAM memory breakdown chart"
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
              formatter={(value: number) => `${value.toFixed(2)} GB`}
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                border: '1px solid #e5e7eb',
                borderRadius: '0.375rem',
              }}
            />
            <Legend
              verticalAlign="bottom"
              height={36}
              iconType="circle"
              wrapperStyle={{ paddingTop: '1rem' }}
            />
            {renderCenterLabel()}
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Screen reader accessible text alternative */}
      <div className="sr-only">
        VRAM breakdown: Model Weights {breakdown.modelWeights.toFixed(2)} GB, KV Cache{' '}
        {breakdown.kvCache.toFixed(2)} GB, Activations {breakdown.activations.toFixed(2)} GB,
        Framework Overhead {breakdown.frameworkOverhead.toFixed(2)} GB, Total {totalGB.toFixed(2)}{' '}
        GB
      </div>
    </div>
  )
}
