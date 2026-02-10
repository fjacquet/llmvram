import { calculateEffectiveBatchSize } from '@engines/optimizations'
import { useUIStore } from '@store/uiStore'

/**
 * Display effective batch size calculation
 *
 * Shows the formula breakdown: micro-batch x accumulation x GPUs
 * Only displayed when gradient accumulation > 1 or numGPUs > 1
 */
export function EffectiveBatchDisplay() {
  const { batchSize, gradientAccumulationSteps, numGPUs } = useUIStore()

  // Only show when effective batch differs from micro-batch
  if (gradientAccumulationSteps === 1 && numGPUs === 1) {
    return null
  }

  const effectiveBatch = calculateEffectiveBatchSize(batchSize, gradientAccumulationSteps, numGPUs)

  return (
    <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
      <div className="space-y-1">
        <div className="text-sm font-semibold text-gray-900 dark:text-white">
          Effective Batch Size: {effectiveBatch.toString()}
        </div>
        <div className="text-xs text-gray-600 dark:text-gray-400">
          = {batchSize} (micro-batch) × {gradientAccumulationSteps} (accumulation) × {numGPUs}{' '}
          (GPUs)
        </div>
      </div>
    </div>
  )
}
