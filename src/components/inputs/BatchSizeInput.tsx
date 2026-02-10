import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

// Preset values for quick selection
const PRESETS = [1, 4, 8, 16, 32, 64]

export function BatchSizeInput() {
  const { batchSize, setBatchSize } = useUIStore()

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBatchSize(Number.parseInt(e.target.value, 10))
  }

  const formatValue = (value: number): string => {
    return value === 1 ? '1 sequence' : `${value} sequences`
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="batch-size"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Batch Size
        </label>
        <InfoTip text="Number of sequences processed simultaneously. Higher batch sizes increase throughput but require more VRAM for activations and KV cache." />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">1</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatValue(batchSize)}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">64</span>
        </div>

        <input
          type="range"
          id="batch-size"
          min={1}
          max={64}
          step={1}
          value={batchSize}
          onChange={handleSliderChange}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          aria-label="Batch size"
          aria-valuetext={formatValue(batchSize)}
        />
      </div>

      {/* Preset buttons */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            onClick={() => setBatchSize(preset)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              batchSize === preset
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {preset}
          </button>
        ))}
      </div>
    </div>
  )
}
