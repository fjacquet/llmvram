import { InfoTip } from '@components/common/InfoTip'
import { MAX_TTFT_TARGET, MIN_TTFT_TARGET } from '@engines/config-rules'
import { useUIStore } from '@store/uiStore'

const PRESETS = [1, 2, 5, 10]

/** Optional TTFT target in seconds; it applies to the last prompt served in a burst */
export function TTFTTargetInput() {
  const target = useUIStore((s) => s.ttftTargetSeconds)
  const setTTFTTarget = useUIStore((s) => s.setTTFTTarget)

  const commit = (raw: string) => {
    const v = Number(raw)
    if (raw.trim() === '' || !Number.isFinite(v)) return setTTFTTarget(null)
    setTTFTTarget(Math.min(MAX_TTFT_TARGET, Math.max(MIN_TTFT_TARGET, v)))
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <label
          htmlFor="ttft-target"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          TTFT target (seconds)
        </label>
        <InfoTip text="Optional. The wait you accept before the first token. It is checked against the last of the concurrent prompts (all arriving at once, first come first served, no prefix cache). Leave empty for no target." />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <input
          key={target ?? 'none'}
          id="ttft-target"
          type="number"
          min={MIN_TTFT_TARGET}
          max={MAX_TTFT_TARGET}
          step={0.1}
          defaultValue={target ?? ''}
          placeholder="none"
          onBlur={(e) => commit(e.target.value)}
          className="w-28 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
        />
        {PRESETS.map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setTTFTTarget(p)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              target === p
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {p} s
          </button>
        ))}
        {target !== null && (
          <button
            type="button"
            onClick={() => setTTFTTarget(null)}
            className="text-xs text-gray-500 dark:text-gray-400 hover:underline"
          >
            Clear
          </button>
        )}
      </div>
    </div>
  )
}
