import type { QuantizationFormat } from '@engines/types'
import { Listbox, ListboxButton, ListboxOption, ListboxOptions } from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'

const QUANTIZATION_LABELS: Record<QuantizationFormat, string> = {
  fp32: 'FP32 (32-bit float)',
  fp16: 'FP16 (16-bit float)',
  bf16: 'BF16 (Brain Float 16)',
  nvfp6: 'NVFP6 (NVIDIA 6-bit float)',
  nvfp4: 'NVFP4 (NVIDIA 4-bit float)',
  int8: 'INT8 (8-bit integer)',
  int4: 'INT4 (4-bit integer)',
  nf4: 'NF4 (4-bit NormalFloat)',
  gptq: 'GPTQ (4-bit + overhead)',
  awq: 'AWQ (4-bit + overhead)',
  'gguf-q8_0': 'Q8_0 (8-bit)',
  'gguf-q6_k': 'Q6_K (6-bit)',
  'gguf-q5_k_s': 'Q5_K_S (5-bit small)',
  'gguf-q5_k_m': 'Q5_K_M (5-bit mixed)',
  'gguf-q5_0': 'Q5_0 (5-bit)',
  'gguf-q4_k_s': 'Q4_K_S (4-bit small)',
  'gguf-q4_k_m': 'Q4_K_M (4-bit mixed)',
  'gguf-q4_0': 'Q4_0 (4-bit)',
  'gguf-q3_k_l': 'Q3_K_L (3-bit large)',
  'gguf-q3_k_m': 'Q3_K_M (3-bit mixed)',
  'gguf-q3_k_s': 'Q3_K_S (3-bit small)',
  'gguf-q2_k': 'Q2_K (2-bit)',
}

const QUANTIZATION_GROUPS = {
  float: ['fp32', 'fp16', 'bf16'] as QuantizationFormat[],
  nvidia: ['nvfp6', 'nvfp4'] as QuantizationFormat[],
  integer: ['int8', 'int4', 'nf4'] as QuantizationFormat[],
  compressed: ['gptq', 'awq'] as QuantizationFormat[],
  gguf: [
    'gguf-q8_0',
    'gguf-q6_k',
    'gguf-q5_k_s',
    'gguf-q5_k_m',
    'gguf-q5_0',
    'gguf-q4_k_s',
    'gguf-q4_k_m',
    'gguf-q4_0',
    'gguf-q3_k_l',
    'gguf-q3_k_m',
    'gguf-q3_k_s',
    'gguf-q2_k',
  ] as QuantizationFormat[],
}

function QuantizationGroup({ label, formats }: { label: string; formats: QuantizationFormat[] }) {
  return (
    <>
      <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider bg-gray-100 dark:bg-gray-900 mt-1 first:mt-0">
        {label}
      </div>
      {formats.map((format) => (
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
                {QUANTIZATION_LABELS[format]}
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
    </>
  )
}

export function QuantizationPicker() {
  const { quantization, setQuantization } = useUIStore()

  return (
    <div className="space-y-2">
      <label
        htmlFor="quantization-picker"
        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        Inference Quantization
      </label>
      <p className="text-xs text-gray-500 dark:text-gray-400">
        Precision for model weights during inference. Lower uses less VRAM but may affect quality.
      </p>

      <Listbox value={quantization} onChange={setQuantization}>
        <div className="relative">
          <ListboxButton
            id="quantization-picker"
            className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 py-2 pl-3 pr-10 text-left shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-600 text-sm"
          >
            <span className="block truncate text-gray-900 dark:text-white">
              {QUANTIZATION_LABELS[quantization]}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
            </span>
          </ListboxButton>

          <ListboxOptions className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            <QuantizationGroup label="Float Formats" formats={QUANTIZATION_GROUPS.float} />
            <QuantizationGroup label="NVIDIA FP Formats" formats={QUANTIZATION_GROUPS.nvidia} />
            <QuantizationGroup label="Integer Formats" formats={QUANTIZATION_GROUPS.integer} />
            <QuantizationGroup label="GPTQ / AWQ" formats={QUANTIZATION_GROUPS.compressed} />
            <QuantizationGroup label="GGUF Formats" formats={QUANTIZATION_GROUPS.gguf} />
          </ListboxOptions>
        </div>
      </Listbox>
    </div>
  )
}
