import { InfoTip } from '@components/common/InfoTip'
import type { ConfigMode } from '@engines/config-rules'
import { Radio, RadioGroup } from '@headlessui/react'
import { useUIStore } from '@store/uiStore'

const MODES: { value: ConfigMode; label: string; title: string; description: string }[] = [
  {
    value: 'inference',
    label: 'Inference',
    title: 'Inference Mode',
    description: 'Calculate VRAM for model inference',
  },
  {
    value: 'sizing',
    label: 'Sizing',
    title: 'Sizing Mode',
    description: 'Size NVL72 racks for N simultaneous users',
  },
  {
    value: 'training',
    label: 'Training',
    title: 'Fine-tuning Mode',
    description: 'Calculate VRAM for training/fine-tuning',
  },
]

/**
 * Mode switch between Inference, Sizing and Fine-tuning
 *
 * Uses Headless UI RadioGroup for accessibility (arrow-key navigation)
 * Updates mode state in Zustand store
 */
export function ModeToggle() {
  const mode = useUIStore((s) => s.mode)
  const setMode = useUIStore((s) => s.setMode)

  const current = MODES.find((m) => m.value === mode) ?? MODES[0]

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{current?.title}</h2>
        <InfoTip text="Inference mode estimates VRAM for running a model. Sizing mode counts the racks that serve N simultaneous users. Training mode adds optimizer states, gradients, and activation memory for fine-tuning." />
      </div>
      <RadioGroup
        value={mode}
        onChange={setMode}
        aria-label="Mode"
        className="inline-flex rounded-lg bg-gray-200 p-0.5 dark:bg-gray-700"
      >
        {MODES.map((m) => (
          <Radio
            key={m.value}
            value={m.value}
            className={({ checked }) =>
              `cursor-pointer rounded-md px-3 py-1 text-sm font-medium transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                checked
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white'
              }`
            }
          >
            {m.label}
          </Radio>
        ))}
      </RadioGroup>
      <p className="text-sm text-gray-600 dark:text-gray-400">{current?.description}</p>
    </div>
  )
}
