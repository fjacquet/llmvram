import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

/**
 * GPU count selector with range slider (1-8 GPUs)
 *
 * Allows users to select the number of GPUs for multi-GPU calculations.
 * Shows the current count as a numeric display alongside the slider.
 */
export function GPUCountSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const setNumGPUs = useUIStore((s) => s.setNumGPUs)

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="gpu-count"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Number of GPUs
        </label>
        <InfoTip text="Use multiple GPUs to distribute model weights. More GPUs enable larger models but add communication overhead between devices." />
      </div>
      <div className="flex items-center gap-4">
        <input
          id="gpu-count"
          type="range"
          min={1}
          max={8}
          step={1}
          value={numGPUs}
          onChange={(e) => setNumGPUs(Number(e.target.value))}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <span className="text-lg font-semibold text-gray-900 dark:text-white w-8 text-center tabular-nums">
          {numGPUs}
        </span>
      </div>
      {numGPUs > 1 && (
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {numGPUs} GPUs with distributed memory
        </p>
      )}
    </div>
  )
}
