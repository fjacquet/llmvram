import { InfoTip } from '@components/common/InfoTip'
import { MAX_CONCURRENT_USERS } from '@engines/constants'
import { useUIStore } from '@store/uiStore'

const PRESETS = [1, 8, 32, 128, 256, 1024, 4096]

const MAX_EXPONENT = Math.log2(MAX_CONCURRENT_USERS)

const clampUsers = (n: number): number =>
  Number.isFinite(n) ? Math.min(MAX_CONCURRENT_USERS, Math.max(1, Math.round(n))) : 1

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
        <InfoTip text="Active sessions whose KV cache must fit in VRAM simultaneously. KV cache grows linearly: each user adds ~1.25 GB for Llama 3 70B at 4 096 tokens (FP16). The results show the maximum that fits at the chosen context." />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600 dark:text-gray-400">1</span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {formatValue(concurrentUsers)}
          </span>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {MAX_CONCURRENT_USERS.toLocaleString('en-US')}
          </span>
        </div>

        <input
          type="range"
          id="concurrent-users"
          min={0}
          max={MAX_EXPONENT}
          step={0.25}
          value={Math.log2(concurrentUsers)}
          onChange={(e) => setConcurrentUsers(clampUsers(2 ** Number(e.target.value)))}
          className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          aria-label="Concurrent users"
          aria-valuetext={formatValue(concurrentUsers)}
        />

        <input
          type="number"
          min={1}
          max={MAX_CONCURRENT_USERS}
          value={concurrentUsers}
          onChange={(e) => setConcurrentUsers(clampUsers(Number(e.target.value)))}
          className="w-28 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          aria-label="Concurrent users (exact)"
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
