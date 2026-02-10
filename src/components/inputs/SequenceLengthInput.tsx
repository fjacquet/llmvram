import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

// Preset values for quick selection
const PRESETS = [
  { value: 512, label: '512' },
  { value: 2048, label: '2K' },
  { value: 4096, label: '4K' },
  { value: 8192, label: '8K' },
  { value: 32768, label: '32K' },
  { value: 131072, label: '128K' },
]

// Min/max for the slider (log2 scale)
const MIN_LOG = 9 // log2(512) = 9
const MAX_LOG = 17 // log2(131072) = 17

export function SequenceLengthInput() {
  const { sequenceLength, setSequenceLength } = useUIStore()

  // Convert sequence length to log2 scale for slider
  const sliderValue = Math.log2(sequenceLength)

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const logValue = Number.parseFloat(e.target.value)
    const actualValue = Math.round(2 ** logValue)
    setSequenceLength(actualValue)
  }

  const formatValue = (value: number): string => {
    if (value >= 1024) {
      const kValue = value / 1024
      return `${kValue.toFixed(kValue % 1 === 0 ? 0 : 1)}K tokens`
    }
    return `${value.toLocaleString()} tokens`
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="sequence-length"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Sequence Length
        </label>
        <InfoTip text="Maximum number of tokens in the context window. Longer sequences increase KV cache memory. Common values: 2K (chat), 8K (documents), 32K+ (long context)." />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">512</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatValue(sequenceLength)}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">128K</span>
        </div>

        <input
          type="range"
          id="sequence-length"
          min={MIN_LOG}
          max={MAX_LOG}
          step={0.1}
          value={sliderValue}
          onChange={handleSliderChange}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          aria-label="Sequence length"
          aria-valuetext={formatValue(sequenceLength)}
        />
      </div>

      {/* Preset buttons */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset.value}
            type="button"
            onClick={() => setSequenceLength(preset.value)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              sequenceLength === preset.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>
    </div>
  )
}
