import { BatchSizeInput } from '@components/inputs/BatchSizeInput'
import { GPUSelector } from '@components/inputs/GPUSelector'
import { KVQuantizationPicker } from '@components/inputs/KVQuantizationPicker'
import { ModelSelector } from '@components/inputs/ModelSelector'
import { QuantizationPicker } from '@components/inputs/QuantizationPicker'
import { SequenceLengthInput } from '@components/inputs/SequenceLengthInput'

/**
 * Input panel assembling all 6 input components in a structured form layout
 *
 * All components read/write Zustand store independently - no local state here
 */
export function InputPanel() {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="space-y-6">
        {/* Model Configuration Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Model Configuration
          </h3>
          <ModelSelector />
          <QuantizationPicker />
        </div>

        <hr className="border-gray-200 dark:border-gray-700" />

        {/* GPU Selection Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">GPU Selection</h3>
          <GPUSelector />
        </div>

        <hr className="border-gray-200 dark:border-gray-700" />

        {/* Parameters Section */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Parameters</h3>
          <SequenceLengthInput />
          <BatchSizeInput />
          <KVQuantizationPicker />
        </div>
      </div>
    </div>
  )
}
