import type { LoRAVRAMBreakdown, TrainingVRAMBreakdown } from '@engines/types'
import type Decimal from 'decimal.js'

interface TrainingBreakdownTableProps {
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

interface RowData {
  name: string
  value: Decimal
  color: string
}

export function TrainingBreakdownTable({ breakdown }: TrainingBreakdownTableProps) {
  // Type guard to check if result is LoRA/QLoRA
  const isLoRA = 'baseWeights' in breakdown

  // Build breakdown rows dynamically
  const rows: RowData[] = []

  if (isLoRA) {
    const loraBreakdown = breakdown as LoRAVRAMBreakdown
    rows.push(
      {
        name: 'Base Model Weights',
        value: loraBreakdown.baseWeights,
        color: TRAINING_COLORS.baseWeights,
      },
      {
        name: 'Adapter Weights',
        value: loraBreakdown.adapterWeights,
        color: TRAINING_COLORS.adapterWeights,
      },
    )
  } else {
    rows.push({
      name: 'Model Weights',
      value: breakdown.modelWeights,
      color: TRAINING_COLORS.modelWeights,
    })
  }

  // Conditionally add master weights only when > 0
  if (breakdown.masterWeights.greaterThan(0)) {
    rows.push({
      name: 'Master Weights (FP32)',
      value: breakdown.masterWeights,
      color: TRAINING_COLORS.masterWeights,
    })
  }

  // Always add remaining components
  rows.push(
    { name: 'Gradients', value: breakdown.gradients, color: TRAINING_COLORS.gradients },
    {
      name: 'Optimizer States',
      value: breakdown.optimizerStates,
      color: TRAINING_COLORS.optimizerStates,
    },
    { name: 'Activations', value: breakdown.activations, color: TRAINING_COLORS.activations },
    {
      name: 'Framework Overhead',
      value: breakdown.frameworkOverhead,
      color: TRAINING_COLORS.frameworkOverhead,
    },
  )

  // Calculate percentage with Decimal precision
  const calculatePercentage = (value: Decimal): string => {
    if (breakdown.total.isZero()) {
      return '0.0'
    }
    return value.div(breakdown.total).mul(100).toFixed(1)
  }

  // Calculate trainable parameter percentage
  const trainablePercent = breakdown.trainableParameters
    .div(breakdown.totalParameters)
    .mul(100)
    .toNumber()

  return (
    <>
      <table
        aria-label="Training memory breakdown details"
        className="w-full text-sm border-collapse text-gray-900 dark:text-gray-100"
      >
        <thead>
          <tr className="bg-gray-100 dark:bg-gray-800">
            <th className="text-left font-medium px-4 py-2 text-gray-700 dark:text-gray-300">
              Component
            </th>
            <th className="text-right font-medium px-4 py-2 text-gray-700 dark:text-gray-300">
              Size (GB)
            </th>
            <th className="text-right font-medium px-4 py-2 text-gray-700 dark:text-gray-300">
              % of Total
            </th>
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
              <td className="text-right px-4 py-2 tabular-nums">
                {calculatePercentage(row.value)}%
              </td>
            </tr>
          ))}
          <tr className="font-bold border-t-2 border-gray-300 dark:border-gray-600">
            <td className="px-4 py-2">Total</td>
            <td className="text-right px-4 py-2 tabular-nums">{breakdown.total.toFixed(2)}</td>
            <td className="text-right px-4 py-2 tabular-nums">100.0%</td>
          </tr>
        </tbody>
      </table>

      {/* Trainable parameters info section */}
      <div className="mt-3 text-sm text-gray-600 dark:text-gray-400">
        <span className="font-medium">Trainable:</span> {breakdown.trainableParameters.toFixed(3)}B
        of {breakdown.totalParameters.toFixed(1)}B parameters ({trainablePercent.toFixed(1)}%)
        {isLoRA && (
          <div className="text-xs text-gray-500 dark:text-gray-500 mt-1">
            Adapter parameters: {(breakdown as LoRAVRAMBreakdown).adapterParameters.toFixed(4)}B
          </div>
        )}
      </div>
    </>
  )
}
