import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

const PRESETS = [1, 4, 8, 16, 32, 64, 128]

export function ConcurrentUsersInput() {
  const { concurrentUsers, setConcurrentUsers } = useUIStore()

  const formatValue = (value: number): string => {
    return value === 1 ? '1 user' : `${value} users`
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="concurrent-users"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Concurrent Users
        </label>
        <InfoTip text="Active sessions whose KV cache must fit in VRAM simultaneously. KV cache grows linearly: each user adds ~1.25 GB for Llama 3 70B at 4 096 tokens (FP16). Typical production: 8–64 for large models, 64–256 for small models." />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">1</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatValue(concurrentUsers)}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">256</span>
        </div>

        <input
          type="range"
          id="concurrent-users"
          min={1}
          max={256}
          step={1}
          value={concurrentUsers}
          onChange={(e) => setConcurrentUsers(Number.parseInt(e.target.value, 10))}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          aria-label="Concurrent users"
          aria-valuetext={formatValue(concurrentUsers)}
        />
      </div>

      {/* Preset buttons */}
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button
            key={preset}
            type="button"
            onClick={() => setConcurrentUsers(preset)}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
              concurrentUsers === preset
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
