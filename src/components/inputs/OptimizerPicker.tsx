import { InfoTip } from '@components/common/InfoTip'
import type { OptimizerType } from '@engines/types'
import {
  Description,
  Field,
  RadioGroup as HeadlessRadioGroup,
  Label,
  Radio,
} from '@headlessui/react'
import { CheckIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'

const OPTIMIZERS: Array<{
  value: OptimizerType
  label: string
  description: string
}> = [
  { value: 'adamw', label: 'AdamW', description: '8 bytes/param (standard)' },
  {
    value: 'sgd-momentum',
    label: 'SGD + Momentum',
    description: '4 bytes/param (memory efficient)',
  },
  {
    value: 'adamw-8bit',
    label: '8-bit AdamW',
    description: '2 bytes/param (quantized states)',
  },
  {
    value: 'adafactor',
    label: 'Adafactor',
    description: '4 bytes/param (factored approx.)',
  },
]

/**
 * Optimizer type picker (AdamW / SGD / 8-bit Adam / Adafactor)
 *
 * Uses Headless UI RadioGroup for accessibility
 * Updates optimizer state in Zustand store
 */
export function OptimizerPicker() {
  const optimizer = useUIStore((s) => s.optimizer)
  const setOptimizer = useUIStore((s) => s.setOptimizer)

  return (
    <Field className="space-y-2">
      <div className="flex items-center gap-1">
        <Label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Optimizer Type
        </Label>
        <InfoTip text="The optimizer determines memory overhead. AdamW needs 8 bytes/param for states. SGD uses 4 bytes/param. 8-bit AdamW halves optimizer memory." />
      </div>
      <HeadlessRadioGroup value={optimizer} onChange={setOptimizer}>
        <div className="space-y-2">
          {OPTIMIZERS.map((opt) => (
            <Radio
              key={opt.value}
              value={opt.value}
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
                      {opt.label}
                    </Label>
                    <Description className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                      {opt.description}
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
