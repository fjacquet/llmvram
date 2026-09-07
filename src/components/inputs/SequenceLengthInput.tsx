import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'
import { MAX_SEQUENCE_LENGTH } from '@utils/schemas'

// Preset values for quick selection
const PRESETS = [
  { value: 512, label: '512' },
  { value: 2048, label: '2K' },
  { value: 4096, label: '4K' },
  { value: 8192, label: '8K' },
  { value: 32768, label: '32K' },
  { value: 131072, label: '128K' },
  { value: 262144, label: '256K' },
  { value: 524288, label: '512K' },
  { value: 1048576, label: '1M' },
]

// Slider is log2-scaled. The floor is fixed; the ceiling is 1M for almost every model,
// extended only for a model whose native context is larger (Llama 4 Scout at 10M).
const MIN_LOG = 9 // log2(512)
const LOG_STEP = 0.1
const DEFAULT_MAX_TOKENS = 1_048_576

function formatTokens(value: number): string {
  if (value >= 1_048_576) {
    const mValue = value / 1_048_576
    return `${mValue.toFixed(mValue % 1 === 0 ? 0 : 1)}M`
  }
  if (value >= 1024) {
    const kValue = value / 1024
    return `${kValue.toFixed(kValue % 1 === 0 ? 0 : 1)}K`
  }
  return value.toLocaleString()
}

export function SequenceLengthInput() {
  const { sequenceLength, setSequenceLength, selectedModel } = useUIStore()

  const nativeContext = selectedModel?.context_length
  const maxTokens = Math.min(Math.max(DEFAULT_MAX_TOKENS, nativeContext ?? 0), MAX_SEQUENCE_LENGTH)
  // A range input only exposes values of min + n*step, so a fractional ceiling is
  // unreachable: log2(10,485,760) = 23.3219 would top out at 23.3, i.e. 10,301,796 tokens,
  // and the advertised maximum could never be selected. Round the ceiling UP to the step
  // grid and clamp the resulting token count back to maxTokens.
  const maxLog = MIN_LOG + Math.ceil((Math.log2(maxTokens) - MIN_LOG) / LOG_STEP) * LOG_STEP

  // The user's value is never rewritten — not on model change, not when it exceeds the
  // model's native context. RoPE/YaRN extension is a real workload, custom models carry
  // no context_length at all, and silently clamping would destroy a shared URL.
  const sliderValue = Math.min(Math.log2(sequenceLength), maxLog)
  const exceedsNative = nativeContext !== undefined && sequenceLength > nativeContext
  // The value can outrun the track when the model changes under it (set 4M on a 10M-context
  // model, then select a 128K one). The thumb pins to the right while the readout keeps the
  // real value, so say that out loud rather than letting the two silently disagree.
  const exceedsSliderMax = sequenceLength > maxTokens

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const logValue = Number.parseFloat(e.target.value)
    setSequenceLength(Math.min(Math.round(2 ** logValue), maxTokens))
  }

  const formatValue = (value: number): string => `${formatTokens(value)} tokens`

  // Position of the native-context marker along the log2 track, as a percentage
  const nativeMarkerPercent =
    nativeContext !== undefined && nativeContext >= 512 && nativeContext <= maxTokens
      ? ((Math.log2(nativeContext) - MIN_LOG) / (maxLog - MIN_LOG)) * 100
      : null

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="sequence-length"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Sequence Length
        </label>
        <InfoTip text="Maximum number of tokens in the context window. Longer sequences increase KV cache memory. Common values: 2K (chat), 8K (documents), 128K-1M (long context). Values above the model's native context require RoPE scaling." />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">512</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatValue(sequenceLength)}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {formatTokens(maxTokens)}
          </span>
        </div>

        <div className="relative">
          {nativeMarkerPercent !== null && (
            <div
              className="absolute top-0 h-2 w-0.5 bg-gray-500 dark:bg-gray-300 pointer-events-none"
              style={{ left: `${nativeMarkerPercent}%` }}
              aria-hidden="true"
              title={`Native context: ${formatTokens(nativeContext ?? 0)}`}
            />
          )}
          <input
            type="range"
            id="sequence-length"
            min={MIN_LOG}
            max={maxLog}
            step={0.1}
            value={sliderValue}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
            aria-label="Sequence length"
            aria-valuetext={formatValue(sequenceLength)}
          />
        </div>
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

      {exceedsSliderMax && (
        <output className="block text-xs text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-md px-2 py-1.5">
          {formatTokens(sequenceLength)} tokens is above this slider's range (max{' '}
          {formatTokens(maxTokens)}). The value is kept and still used in every calculation, but the
          slider cannot show it — moving the slider will replace it.
        </output>
      )}

      {exceedsNative && (
        <output className="block text-xs text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-md px-2 py-1.5">
          Beyond native context ({formatTokens(nativeContext ?? 0)}) — requires RoPE scaling / YaRN.
          The estimate still computes.
        </output>
      )}
    </div>
  )
}
