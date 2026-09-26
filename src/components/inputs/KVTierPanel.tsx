import { InfoTip } from '@components/common/InfoTip'
import { KV_TIER_PRESETS, KV_TIER_TYPES, type KVTierType } from '@engines/kv-tier'
import { useUIStore } from '@store/uiStore'

const inputClass =
  'w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white'

/** Parses a number field; empty or non-positive becomes null. */
const positiveOrNull = (raw: string): number | null => {
  const n = Number(raw)
  return raw.trim() === '' || !Number.isFinite(n) || n <= 0 ? null : n
}

/**
 * KV storage tier: park idle sessions' KV off the GPU (host memory, local NVMe,
 * CMX, PowerScale, ObjectScale) and reload it on resume.
 */
export function KVTierPanel() {
  const kvTier = useUIStore((s) => s.kvTier)
  const setKVTier = useUIStore((s) => s.setKVTier)
  const preset = kvTier.tier === 'none' ? null : KV_TIER_PRESETS[kvTier.tier]

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <label
          htmlFor="kv-tier"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          KV storage tier
        </label>
        <InfoTip text="Idle agentic sessions park their KV cache on this tier and reload it when they resume, instead of recomputing the prompt. Parked sessions use no GPU memory." />
      </div>
      <select
        id="kv-tier"
        aria-label="KV storage tier"
        value={kvTier.tier}
        onChange={(e) => setKVTier({ tier: e.target.value as KVTierType })}
        className={inputClass}
      >
        {KV_TIER_TYPES.map((t) => (
          <option key={t} value={t}>
            {t === 'none'
              ? 'None'
              : `${KV_TIER_PRESETS[t].label} (${KV_TIER_PRESETS[t].gbpsPerGPU} GB/s per GPU)`}
          </option>
        ))}
      </select>

      {preset && (
        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Bandwidth per GPU (GB/s)
            <input
              type="number"
              min={0}
              aria-label="Bandwidth per GPU (GB/s)"
              placeholder={String(preset.gbpsPerGPU)}
              value={kvTier.customGBps ?? ''}
              onChange={(e) => setKVTier({ customGBps: positiveOrNull(e.target.value) })}
              className={inputClass}
            />
          </label>
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Active share (%)
            <input
              type="number"
              min={1}
              max={100}
              aria-label="Active share (%)"
              value={Math.round(kvTier.activeShare * 100)}
              onChange={(e) =>
                setKVTier({
                  activeShare: Math.min(100, Math.max(1, Number(e.target.value) || 0)) / 100,
                })
              }
              className={inputClass}
            />
          </label>
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Active burst (s)
            <input
              type="number"
              min={1}
              aria-label="Active burst (s)"
              value={kvTier.burstSeconds}
              onChange={(e) =>
                setKVTier({ burstSeconds: Math.max(1, Number(e.target.value) || 1) })
              }
              className={inputClass}
            />
          </label>
          <label className="text-xs text-gray-600 dark:text-gray-400">
            Tier capacity (TB)
            <input
              type="number"
              min={0}
              aria-label="Tier capacity (TB)"
              placeholder="unlimited"
              value={kvTier.capacityTB ?? ''}
              onChange={(e) => setKVTier({ capacityTB: positiveOrNull(e.target.value) })}
              className={inputClass}
            />
          </label>
        </div>
      )}
    </div>
  )
}
