import { InfoTip } from '@components/common/InfoTip'
import type { KVCachePrecision } from '@engines/types'
import { Listbox, ListboxButton, ListboxOption, ListboxOptions } from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'

const KV_QUANTIZATION_LABELS: Record<KVCachePrecision, string> = {
  fp16: 'FP16 (default)',
  fp8: 'FP8',
  int8: 'INT8',
  int4: 'INT4 (most aggressive)',
}

const KV_QUANTIZATION_OPTIONS: KVCachePrecision[] = ['fp16', 'fp8', 'int8', 'int4']

export function KVQuantizationPicker() {
  const { kvQuantization, setKVQuantization } = useUIStore()

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <label
          htmlFor="kv-quantization-picker"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          KV Cache Precision
        </label>
        <InfoTip text="Quantize the KV cache independently from model weights. INT8 or INT4 KV cache can significantly reduce memory for long sequences with minimal quality impact." />
      </div>

      <Listbox value={kvQuantization} onChange={setKVQuantization}>
        <div className="relative">
          <ListboxButton
            id="kv-quantization-picker"
            className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 py-2 pl-3 pr-10 text-left shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-600 text-sm"
          >
            <span className="block truncate text-gray-900 dark:text-white">
              {KV_QUANTIZATION_LABELS[kvQuantization]}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
            </span>
          </ListboxButton>

          <ListboxOptions className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            {KV_QUANTIZATION_OPTIONS.map((format) => (
              <ListboxOption
                key={format}
                value={format}
                className={({ active }) =>
                  `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                    active ? 'bg-blue-600 text-white' : 'text-gray-900 dark:text-gray-100'
                  }`
                }
              >
                {({ selected, active }) => (
                  <>
                    <span className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}>
                      {KV_QUANTIZATION_LABELS[format]}
                    </span>
                    {selected && (
                      <span
                        className={`absolute inset-y-0 left-0 flex items-center pl-3 ${
                          active ? 'text-white' : 'text-blue-600'
                        }`}
                      >
                        <CheckIcon className="h-5 w-5" aria-hidden="true" />
                      </span>
                    )}
                  </>
                )}
              </ListboxOption>
            ))}
          </ListboxOptions>
        </div>
      </Listbox>
    </div>
  )
}
