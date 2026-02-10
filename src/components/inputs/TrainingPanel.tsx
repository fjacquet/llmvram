import { EffectiveBatchDisplay } from '@components/inputs/EffectiveBatchDisplay'
import { GradientAccumulationInput } from '@components/inputs/GradientAccumulationInput'
import { OptimizationToggles } from '@components/inputs/OptimizationToggles'
import { OptimizerPicker } from '@components/inputs/OptimizerPicker'
import { PrecisionPicker } from '@components/inputs/PrecisionPicker'
import { TrainingMethodPicker } from '@components/inputs/TrainingMethodPicker'

/**
 * Training configuration panel assembling all training-specific inputs
 *
 * Contains training method, optimizer, precision pickers, and memory optimization controls
 */
export function TrainingPanel() {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        Training Configuration
      </h3>
      <TrainingMethodPicker />
      <OptimizerPicker />
      <PrecisionPicker />

      <hr className="border-gray-200 dark:border-gray-700" />

      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Memory Optimizations</h3>
      <GradientAccumulationInput />
      <EffectiveBatchDisplay />
      <OptimizationToggles />
    </div>
  )
}
