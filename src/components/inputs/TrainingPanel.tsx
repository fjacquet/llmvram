import { OptimizerPicker } from '@components/inputs/OptimizerPicker'
import { PrecisionPicker } from '@components/inputs/PrecisionPicker'
import { TrainingMethodPicker } from '@components/inputs/TrainingMethodPicker'

/**
 * Training configuration panel assembling all training-specific inputs
 *
 * Contains training method, optimizer, and precision pickers
 * LoRA-specific options (rank, alpha, target modules) will be added in a later phase
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
    </div>
  )
}
