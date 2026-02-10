import { Description, Field, Label, Switch } from '@headlessui/react'
import { useUIStore } from '@store/uiStore'

/**
 * Mode toggle for switching between Inference and Fine-tuning modes
 *
 * Uses Headless UI Switch component for accessibility
 * Updates mode state in Zustand store
 */
export function ModeToggle() {
  const mode = useUIStore((s) => s.mode)
  const setMode = useUIStore((s) => s.setMode)

  const isTraining = mode === 'training'

  return (
    <Field className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-lg font-semibold text-gray-900 dark:text-white">
          {isTraining ? 'Fine-tuning Mode' : 'Inference Mode'}
        </Label>
        <Switch
          checked={isTraining}
          onChange={(checked) => setMode(checked ? 'training' : 'inference')}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
            isTraining ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
              isTraining ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </Switch>
      </div>
      <Description className="text-sm text-gray-600 dark:text-gray-400">
        {isTraining
          ? 'Calculate VRAM for training/fine-tuning'
          : 'Calculate VRAM for model inference'}
      </Description>
    </Field>
  )
}
