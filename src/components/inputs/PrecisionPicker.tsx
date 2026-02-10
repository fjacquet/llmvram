import type { TrainingPrecision } from '@engines/types'
import {
  Description,
  Field,
  RadioGroup as HeadlessRadioGroup,
  Label,
  Radio,
} from '@headlessui/react'
import { CheckIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'

const PRECISIONS: Array<{
  value: TrainingPrecision
  label: string
  description: string
}> = [
  { value: 'fp32', label: 'FP32', description: 'Full precision (no master weights needed)' },
  { value: 'fp16', label: 'FP16', description: 'Half precision (+ FP32 master weights)' },
  {
    value: 'bf16',
    label: 'BF16',
    description: 'BFloat16 (+ FP32 master weights, recommended)',
  },
]

/**
 * Training precision picker (FP32 / FP16 / BF16)
 *
 * Uses Headless UI RadioGroup for accessibility
 * Updates trainingPrecision state in Zustand store
 */
export function PrecisionPicker() {
  const trainingPrecision = useUIStore((s) => s.trainingPrecision)
  const setTrainingPrecision = useUIStore((s) => s.setTrainingPrecision)

  return (
    <Field className="space-y-2">
      <Label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Training Precision
      </Label>
      <HeadlessRadioGroup value={trainingPrecision} onChange={setTrainingPrecision}>
        <div className="space-y-2">
          {PRECISIONS.map((precision) => (
            <Radio
              key={precision.value}
              value={precision.value}
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
                      {precision.label}
                    </Label>
                    <Description className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                      {precision.description}
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
