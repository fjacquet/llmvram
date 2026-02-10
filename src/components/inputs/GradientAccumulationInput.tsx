import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

// Preset values for quick selection
const PRESETS = [1, 2, 4, 8, 16, 32, 64]

export function GradientAccumulationInput() {
  const { gradientAccumulationSteps, setGradientAccumulationSteps } = useUIStore()

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setGradientAccumulationSteps(Number.parseInt(e.target.value, 10))
  }

  const formatValue = (value: number): string => {
    return value === 1 ? '1 step' : `${value} steps`
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="gradient-accumulation"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Gradient Accumulation Steps
        </label>
        <InfoTip text="Accumulate gradients over multiple micro-batches before updating weights. Simulates larger batch sizes without the VRAM cost." />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">1</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatValue(gradientAccumulationSteps)}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">128</span>
        </div>

        <input
          type="range"
          id="gradient-accumulation"
          min={1}
          max={128}
          step={1}
          value={gradientAccumulationSteps}
          onChange={handleSliderChange}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          aria-label="Gradient accumulation steps"
          aria-valuetext={formatValue(gradientAccumulationSteps)}
        />
      </div>

      {/* Preset buttons */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            onClick={() => setGradientAccumulationSteps(preset)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              gradientAccumulationSteps === preset
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
