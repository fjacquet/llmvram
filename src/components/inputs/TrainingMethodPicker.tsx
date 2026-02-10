import { InfoTip } from '@components/common/InfoTip'
import type { FineTuningMethod } from '@engines/types'
import {
  Description,
  Field,
  RadioGroup as HeadlessRadioGroup,
  Label,
  Radio,
} from '@headlessui/react'
import { CheckIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'

const TRAINING_METHODS: Array<{
  value: FineTuningMethod
  label: string
  description: string
}> = [
  {
    value: 'full',
    label: 'Full Fine-tuning',
    description: 'Train all parameters (highest VRAM)',
  },
  { value: 'lora', label: 'LoRA', description: 'Low-rank adapters (medium VRAM)' },
  {
    value: 'qlora',
    label: 'QLoRA',
    description: '4-bit base + adapters (lowest VRAM)',
  },
]

/**
 * Training method picker (Full / LoRA / QLoRA)
 *
 * Uses Headless UI RadioGroup for accessibility
 * Updates trainingMethod state in Zustand store
 */
export function TrainingMethodPicker() {
  const trainingMethod = useUIStore((s) => s.trainingMethod)
  const setTrainingMethod = useUIStore((s) => s.setTrainingMethod)

  return (
    <Field className="space-y-2">
      <div className="flex items-center gap-1">
        <Label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Fine-tuning Method
        </Label>
        <InfoTip text="Full fine-tuning updates all parameters. LoRA trains small adapter layers (~1% of params). QLoRA combines 4-bit quantization with LoRA for minimum VRAM." />
      </div>
      <HeadlessRadioGroup value={trainingMethod} onChange={setTrainingMethod}>
        <div className="space-y-2">
          {TRAINING_METHODS.map((method) => (
            <Radio
              key={method.value}
              value={method.value}
              className={({ checked }) =>
                `relative flex cursor-pointer rounded-lg border-2 p-3 transition-colors ${
                  checked
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`
              }
            >
              {({ checked }) => (
                <div className="flex w-full items-start gap-3">
                  <div className="flex h-5 items-center">
                    <div
                      className={`flex h-5 w-5 items-center justify-center rounded-full border-2 ${
                        checked
                          ? 'border-blue-600 bg-blue-600'
                          : 'border-gray-300 dark:border-gray-600'
                      }`}
                    >
                      {checked && <CheckIcon className="h-3 w-3 text-white" aria-hidden="true" />}
                    </div>
                  </div>
                  <div className="flex-1">
                    <Label className="block font-medium text-gray-900 dark:text-white">
                      {method.label}
                    </Label>
                    <Description className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                      {method.description}
                    </Description>
                  </div>
                </div>
              )}
            </Radio>
          ))}
        </div>
      </HeadlessRadioGroup>
    </Field>
  )
}
