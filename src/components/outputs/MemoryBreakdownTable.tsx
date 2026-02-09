import type { InferenceVRAMBreakdown } from '@engines/types'
import type Decimal from 'decimal.js'

interface MemoryBreakdownTableProps {
  breakdown: InferenceVRAMBreakdown
}

const COLORS = {
  modelWeights: '#3b82f6', // blue-500
  kvCache: '#10b981', // emerald-500
  activations: '#f59e0b', // amber-500
  frameworkOverhead: '#8b5cf6', // violet-500
}

interface RowData {
  name: string
  value: Decimal
  color: string
}

export function MemoryBreakdownTable({ breakdown }: MemoryBreakdownTableProps) {
  const rows: RowData[] = [
    { name: 'Model Weights', value: breakdown.modelWeights, color: COLORS.modelWeights },
    { name: 'KV Cache', value: breakdown.kvCache, color: COLORS.kvCache },
    { name: 'Activations', value: breakdown.activations, color: COLORS.activations },
    {
      name: 'Framework Overhead',
      value: breakdown.frameworkOverhead,
      color: COLORS.frameworkOverhead,
    },
  ]

  // Calculate percentage with Decimal precision
  const calculatePercentage = (value: Decimal): string => {
    if (breakdown.total.isZero()) {
      return '0.0'
    }
    return value.div(breakdown.total).mul(100).toFixed(1)
  }

  return (
    <table aria-label="Memory breakdown details" className="w-full text-sm border-collapse text-gray-900 dark:text-gray-100">
      <thead>
        <tr className="bg-gray-100 dark:bg-gray-800">
          <th className="text-left font-medium px-4 py-2 text-gray-700 dark:text-gray-300">Component</th>
          <th className="text-right font-medium px-4 py-2 text-gray-700 dark:text-gray-300">Size (GB)</th>
          <th className="text-right font-medium px-4 py-2 text-gray-700 dark:text-gray-300">% of Total</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={row.name} className={index % 2 === 1 ? 'bg-gray-50 dark:bg-gray-800/50' : ''}>
            <td className="px-4 py-2">
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: row.color }}
                  aria-hidden="true"
                />
                <span>{row.name}</span>
              </div>
            </td>
            <td className="text-right px-4 py-2 tabular-nums">{row.value.toFixed(2)}</td>
            <td className="text-right px-4 py-2 tabular-nums">{calculatePercentage(row.value)}%</td>
          </tr>
        ))}
        <tr className="font-bold border-t-2 border-gray-300 dark:border-gray-600">
          <td className="px-4 py-2">Total</td>
          <td className="text-right px-4 py-2 tabular-nums">{breakdown.total.toFixed(2)}</td>
          <td className="text-right px-4 py-2 tabular-nums">100.0%</td>
        </tr>
      </tbody>
    </table>
  )
}
