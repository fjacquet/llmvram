import { InfoTip } from '@components/common/InfoTip'
import { graceLinkGBps, KV_TIER_PRESETS, type KVTierType } from '@engines/kv-tier'
import { useAllowedOptions } from '@hooks/useAllowedOptions'
import { useUIStore } from '@store/uiStore'
import { useEffect, useState } from 'react'

const inputClass =
  'w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white'

/** Parses a number field; empty or non-finite becomes null. The store clamps the rest. */
const parseNumber = (raw: string): number | null => {
  const n = Number(raw)
  return raw.trim() === '' || !Number.isFinite(n) ? null : n
}

/**
 * Number input that keeps what the user types and commits on blur, so clearing a
 * field or typing "0.5" digit by digit never snaps to a clamped value mid-edit.
 */
function NumberField(props: {
  label: string
  value: number | null
  commit: (raw: string) => void
  placeholder?: string
  min?: number
  max?: number
}) {
  const shown = props.value === null ? '' : String(props.value)
  const [text, setText] = useState(shown)
  // Follow store changes made elsewhere (URL restore, another commit)
  useEffect(() => setText(shown), [shown])
  return (
    <label className="text-xs text-gray-600 dark:text-gray-400">
      {props.label}
      <input
        type="number"
        min={props.min}
        max={props.max}
        aria-label={props.label}
        placeholder={props.placeholder}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          props.commit(text)
          setText(shown)
        }}
        className={inputClass}
      />
    </label>
  )
}

/**
 * KV storage tier: park idle sessions' KV off the GPU (host memory, local NVMe,
 * CMX, PowerScale, ObjectScale) and reload it on resume.
 */
export function KVTierPanel() {
  const kvTier = useUIStore((s) => s.kvTier)
  const setKVTier = useUIStore((s) => s.setKVTier)
  const selectedGPU = useUIStore((s) => s.selectedGPU)
  const preset = kvTier.tier === 'none' ? null : KV_TIER_PRESETS[kvTier.tier]
  // Grace-host bandwidth depends on which Grace GPU is selected (225 NVL72,
  // 396 Desktop Superchip) — resolve it once and use it everywhere the
  // static KV_TIER_PRESETS figure would otherwise stand in for host-grace.
  const graceGBps = graceLinkGBps(selectedGPU?.id ?? '')
  const tierOptions = useAllowedOptions().kvTiers
  const presetGBps =
    kvTier.tier === 'host-grace' ? (graceGBps ?? preset?.gbpsPerGPU) : preset?.gbpsPerGPU

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
        {tierOptions.map((t) => (
          <option key={t} value={t}>
            {t === 'none'
              ? 'None'
              : t === 'host-grace'
                ? // Visible only when graceGBps !== null, so it's a real number here.
                  `${KV_TIER_PRESETS[t].label} (${graceGBps} GB/s per GPU)`
                : `${KV_TIER_PRESETS[t].label} (${KV_TIER_PRESETS[t].gbpsPerGPU} GB/s per GPU)`}
          </option>
        ))}
      </select>

      {preset && (
        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Bandwidth per GPU (GB/s)"
            min={0}
            placeholder={String(presetGBps)}
            value={kvTier.customGBps}
            commit={(raw) => setKVTier({ customGBps: parseNumber(raw) })}
          />
          <NumberField
            label="Active share (%)"
            min={1}
            max={100}
            value={Math.round(kvTier.activeShare * 100)}
            commit={(raw) => {
              const n = parseNumber(raw)
              if (n !== null) setKVTier({ activeShare: n / 100 })
            }}
          />
          <NumberField
            label="Active burst (s)"
            min={1}
            value={kvTier.burstSeconds}
            commit={(raw) => {
              const n = parseNumber(raw)
              if (n !== null) setKVTier({ burstSeconds: n })
            }}
          />
          <NumberField
            label="Tier capacity (TB)"
            min={0}
            placeholder="unlimited"
            value={kvTier.capacityTB}
            commit={(raw) => setKVTier({ capacityTB: parseNumber(raw) })}
          />
        </div>
      )}
    </div>
  )
}
