import { INTERCONNECT_LABELS } from '@engines/constants'
import { useAllowedOptions } from '@hooks/useAllowedOptions'
import { useUIStore } from '@store/uiStore'

export function InterconnectSelector() {
  const selectedGPU = useUIStore((s) => s.selectedGPU)
  const interconnectOverride = useUIStore((s) => s.interconnectOverride)
  const setInterconnectOverride = useUIStore((s) => s.setInterconnectOverride)
  const { interconnectOptions: options } = useAllowedOptions()

  // R5 + R13: variants only for multi-GPU parts with two or more options
  if (!selectedGPU || options.length < 2) {
    return null
  }

  const current = interconnectOverride ?? selectedGPU.interconnect ?? options[0]

  return (
    <div className="mt-3">
      <p className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        Interconnect Type
      </p>
      <div className="flex flex-col gap-1.5">
        {options.map((opt) => {
          const label = INTERCONNECT_LABELS[opt] ?? opt
          return (
            <label key={opt} className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="interconnect-override"
                value={opt}
                checked={current === opt}
                onChange={() => setInterconnectOverride(opt)}
                className="text-blue-600"
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">{label}</span>
            </label>
          )
        })}
      </div>
    </div>
  )
}
