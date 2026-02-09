import type { ReactElement } from 'react'
import type Decimal from 'decimal.js'

interface FitIndicatorProps {
  totalVRAM: Decimal
  availableVRAM: number
}

interface StatusTier {
  label: string
  bgLight: string
  bgDark: string
  border: string
  bar: string
  text: string
  icon: ReactElement
}

export function FitIndicator({ totalVRAM, availableVRAM }: FitIndicatorProps) {
  const percentage = totalVRAM.div(availableVRAM).mul(100).toNumber()

  // Status icons as inline SVG
  const CheckIcon = (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <title>Success</title>
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  )

  const WarningIcon = (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <title>Warning</title>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
      />
    </svg>
  )

  const ErrorIcon = (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <title>Error</title>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
  )

  // Determine status tier
  const getStatusTier = (): StatusTier => {
    if (percentage <= 80) {
      return {
        label: 'Fits Comfortably',
        bgLight: 'bg-green-50',
        bgDark: 'dark:bg-green-900/20',
        border: 'border-green-200 dark:border-green-800',
        bar: 'bg-green-500',
        text: 'text-green-700 dark:text-green-400',
        icon: CheckIcon,
      }
    }
    if (percentage <= 95) {
      return {
        label: 'Tight Fit',
        bgLight: 'bg-yellow-50',
        bgDark: 'dark:bg-yellow-900/20',
        border: 'border-yellow-200 dark:border-yellow-800',
        bar: 'bg-yellow-500',
        text: 'text-yellow-700 dark:text-yellow-400',
        icon: WarningIcon,
      }
    }
    return {
      label: 'Does Not Fit',
      bgLight: 'bg-red-50',
      bgDark: 'dark:bg-red-900/20',
      border: 'border-red-200 dark:border-red-800',
      bar: 'bg-red-500',
      text: 'text-red-700 dark:text-red-400',
      icon: ErrorIcon,
    }
  }

  const status = getStatusTier()
  const totalGB = totalVRAM.toNumber()

  return (
    <output
      aria-live="polite"
      className={`rounded-lg border-2 p-4 ${status.bgLight} ${status.bgDark} ${status.border} block`}
    >
      <div className="space-y-3">
        {/* Status header */}
        <div className="flex items-center justify-between">
          <div className={`flex items-center gap-2 font-medium ${status.text}`}>
            {status.icon}
            <span>{status.label}</span>
          </div>
          <span className={`text-xl font-bold ${status.text}`}>{percentage.toFixed(0)}%</span>
        </div>

        {/* Usage text */}
        <div className="text-sm text-gray-600 dark:text-gray-300">
          Using {totalGB.toFixed(2)} GB of {availableVRAM} GB
        </div>

        {/* Progress bar */}
        <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${status.bar} transition-all duration-300`}
            style={{ width: `${Math.min(percentage, 100)}%` }}
            aria-hidden="true"
          />
        </div>
      </div>
    </output>
  )
}
