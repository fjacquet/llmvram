import { InfoTip } from '@components/common/InfoTip'
import gpusData from '@data/gpus.json'
import {
  Combobox,
  ComboboxButton,
  ComboboxInput,
  ComboboxOption,
  ComboboxOptions,
} from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'
import { validateGPUs } from '@utils/schemas'
import { useMemo, useState } from 'react'
import { type CustomGPUInput, createCustomGPU, type GPU } from '@/types/gpu'

// Validate GPUs data at module load
const gpus = validateGPUs(gpusData)

export function GPUSelector() {
  const { selectedGPU, setSelectedGPU } = useUIStore()
  const [query, setQuery] = useState('')
  const [showCustomForm, setShowCustomForm] = useState(false)

  // Custom GPU form state
  const [customName, setCustomName] = useState('')
  const [customVRAM, setCustomVRAM] = useState('')
  const [customBandwidth, setCustomBandwidth] = useState('')
  const [customFLOPS, setCustomFLOPS] = useState('')

  // Filter and group GPUs
  const filteredGPUs = useMemo(() => {
    const filtered =
      query === ''
        ? gpus
        : gpus.filter((gpu) => gpu.name.toLowerCase().includes(query.toLowerCase()))

    // Group by manufacturer
    const grouped = {
      nvidia: filtered.filter((gpu) => gpu.manufacturer === 'nvidia'),
      amd: filtered.filter((gpu) => gpu.manufacturer === 'amd'),
      apple: filtered.filter((gpu) => gpu.manufacturer === 'apple'),
      intel: filtered.filter((gpu) => gpu.manufacturer === 'intel'),
    }

    return grouped
  }, [query])

  const handleCustomGPUSubmit = () => {
    const vramNum = Number.parseFloat(customVRAM)
    if (!customName || !vramNum || vramNum < 1) {
      return
    }

    const input: CustomGPUInput = {
      name: customName,
      vram_gb: vramNum,
      memory_bandwidth_gbps: customBandwidth ? Number.parseFloat(customBandwidth) : undefined,
      fp16_tflops: customFLOPS ? Number.parseFloat(customFLOPS) : undefined,
    }

    const customGPU = createCustomGPU(input)
    setSelectedGPU(customGPU)

    // Reset form
    setCustomName('')
    setCustomVRAM('')
    setCustomBandwidth('')
    setCustomFLOPS('')
    setShowCustomForm(false)
  }

  const getTierBadge = (tier: string) => {
    switch (tier) {
      case 'datacenter':
        return 'DC'
      case 'consumer':
        return 'Consumer'
      case 'apple-silicon':
        return 'Apple'
      default:
        return ''
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <label
          htmlFor="gpu-selector"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          GPU
        </label>
        <InfoTip text="Choose your target GPU or define custom hardware specs. VRAM capacity determines whether the model fits. Memory bandwidth affects inference speed." />
      </div>

      <Combobox
        value={selectedGPU}
        onChange={(gpu) => {
          if (gpu === null) {
            // "Custom GPU..." option selected
            setShowCustomForm(true)
            setQuery('')
          } else {
            setSelectedGPU(gpu)
            setShowCustomForm(false)
          }
        }}
      >
        <div className="relative">
          <div className="relative w-full cursor-default overflow-hidden rounded-lg bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-left shadow-sm focus-within:ring-2 focus-within:ring-blue-600">
            <ComboboxInput
              id="gpu-selector"
              className="w-full border-none py-2 pl-3 pr-10 text-sm leading-5 text-gray-900 dark:text-white bg-transparent focus:ring-0 focus:outline-none"
              displayValue={(gpu: GPU | null) => gpu?.name || ''}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search GPUs..."
            />
            <ComboboxButton className="absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
            </ComboboxButton>
          </div>

          <ComboboxOptions className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            {filteredGPUs.nvidia.length === 0 &&
            filteredGPUs.amd.length === 0 &&
            filteredGPUs.apple.length === 0 &&
            filteredGPUs.intel.length === 0 &&
            query !== '' ? (
              <div className="relative cursor-default select-none py-2 px-4 text-gray-700 dark:text-gray-300">
                No GPUs found.
              </div>
            ) : (
              <>
                {/* NVIDIA GPUs */}
                {filteredGPUs.nvidia.length > 0 && (
                  <>
                    <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider bg-gray-100 dark:bg-gray-900">
                      NVIDIA
                    </div>
                    {filteredGPUs.nvidia.map((gpu) => (
                      <ComboboxOption
                        key={gpu.id}
                        value={gpu}
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                            active ? 'bg-blue-600 text-white' : 'text-gray-900 dark:text-gray-100'
                          }`
                        }
                      >
                        {({ selected, active }) => (
                          <>
                            <div className="flex items-center justify-between">
                              <span
                                className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                              >
                                {gpu.name}
                              </span>
                              <div className="flex items-center gap-2 shrink-0 ml-2">
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.vram_gb} GB
                                </span>
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.memory_bandwidth_gbps} GB/s
                                </span>
                                <span
                                  className={`text-xs px-1.5 py-0.5 rounded ${active ? 'bg-blue-700' : 'bg-gray-200 dark:bg-gray-700'}`}
                                >
                                  {getTierBadge(gpu.tier)}
                                </span>
                              </div>
                            </div>
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
                      </ComboboxOption>
                    ))}
                  </>
                )}

                {/* AMD GPUs */}
                {filteredGPUs.amd.length > 0 && (
                  <>
                    <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider bg-gray-100 dark:bg-gray-900 mt-1">
                      AMD
                    </div>
                    {filteredGPUs.amd.map((gpu) => (
                      <ComboboxOption
                        key={gpu.id}
                        value={gpu}
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                            active ? 'bg-blue-600 text-white' : 'text-gray-900 dark:text-gray-100'
                          }`
                        }
                      >
                        {({ selected, active }) => (
                          <>
                            <div className="flex items-center justify-between">
                              <span
                                className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                              >
                                {gpu.name}
                              </span>
                              <div className="flex items-center gap-2 shrink-0 ml-2">
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.vram_gb} GB
                                </span>
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.memory_bandwidth_gbps} GB/s
                                </span>
                                <span
                                  className={`text-xs px-1.5 py-0.5 rounded ${active ? 'bg-blue-700' : 'bg-gray-200 dark:bg-gray-700'}`}
                                >
                                  {getTierBadge(gpu.tier)}
                                </span>
                              </div>
                            </div>
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
                      </ComboboxOption>
                    ))}
                  </>
                )}

                {/* Apple GPUs */}
                {filteredGPUs.apple.length > 0 && (
                  <>
                    <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider bg-gray-100 dark:bg-gray-900 mt-1">
                      Apple
                    </div>
                    {filteredGPUs.apple.map((gpu) => (
                      <ComboboxOption
                        key={gpu.id}
                        value={gpu}
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                            active ? 'bg-blue-600 text-white' : 'text-gray-900 dark:text-gray-100'
                          }`
                        }
                      >
                        {({ selected, active }) => (
                          <>
                            <div className="flex items-center justify-between">
                              <span
                                className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                              >
                                {gpu.name}
                              </span>
                              <div className="flex items-center gap-2 shrink-0 ml-2">
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.vram_gb} GB
                                </span>
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.memory_bandwidth_gbps} GB/s
                                </span>
                                <span
                                  className={`text-xs px-1.5 py-0.5 rounded ${active ? 'bg-blue-700' : 'bg-gray-200 dark:bg-gray-700'}`}
                                >
                                  {getTierBadge(gpu.tier)}
                                </span>
                              </div>
                            </div>
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
                      </ComboboxOption>
                    ))}
                  </>
                )}

                {/* Intel GPUs */}
                {filteredGPUs.intel.length > 0 && (
                  <>
                    <div className="px-4 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider bg-gray-100 dark:bg-gray-900 mt-1">
                      Intel
                    </div>
                    {filteredGPUs.intel.map((gpu) => (
                      <ComboboxOption
                        key={gpu.id}
                        value={gpu}
                        className={({ active }) =>
                          `relative cursor-pointer select-none py-2 pl-10 pr-4 ${
                            active ? 'bg-blue-600 text-white' : 'text-gray-900 dark:text-gray-100'
                          }`
                        }
                      >
                        {({ selected, active }) => (
                          <>
                            <div className="flex items-center justify-between">
                              <span
                                className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                              >
                                {gpu.name}
                              </span>
                              <div className="flex items-center gap-2 shrink-0 ml-2">
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.vram_gb} GB
                                </span>
                                <span
                                  className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                                >
                                  {gpu.memory_bandwidth_gbps} GB/s
                                </span>
                                <span
                                  className={`text-xs px-1.5 py-0.5 rounded ${active ? 'bg-blue-700' : 'bg-gray-200 dark:bg-gray-700'}`}
                                >
                                  {getTierBadge(gpu.tier)}
                                </span>
                              </div>
                            </div>
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
                      </ComboboxOption>
                    ))}
                  </>
                )}

                {/* Custom GPU option */}
                <ComboboxOption
                  value={null}
                  className={({ active }) =>
                    `relative cursor-pointer select-none py-2 pl-10 pr-4 border-t border-gray-200 dark:border-gray-700 mt-1 ${
                      active ? 'bg-blue-600 text-white' : 'text-gray-900 dark:text-gray-100'
                    }`
                  }
                >
                  {({ active }) => (
                    <span className={`block ${active ? 'font-medium' : 'font-normal'}`}>
                      Custom GPU...
                    </span>
                  )}
                </ComboboxOption>
              </>
            )}
          </ComboboxOptions>
        </div>
      </Combobox>

      {/* Selected GPU metadata */}
      {selectedGPU && (
        <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 px-1">
          <div className="flex items-center gap-3">
            <span>{selectedGPU.memory_bandwidth_gbps} GB/s</span>
            <span>{selectedGPU.memory_type}</span>
            {selectedGPU.fp16_tflops && <span>{selectedGPU.fp16_tflops} TFLOPS</span>}
            {selectedGPU.tdp_watts && <span>{selectedGPU.tdp_watts}W TDP</span>}
          </div>
          {selectedGPU.spec_url && (
            <a
              href={selectedGPU.spec_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 dark:text-blue-400 hover:underline"
            >
              Spec sheet ↗
            </a>
          )}
        </div>
      )}

      {/* Interconnect variant selector (only shown for GPUs with multiple options) */}
      <InterconnectSelector />

      {/* Custom GPU form */}
      {showCustomForm && (
        <div className="mt-4 p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-3">
          <h3 className="text-sm font-medium text-gray-900 dark:text-white">Custom GPU</h3>

          <div>
            <label
              htmlFor="custom-gpu-name"
              className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
            >
              Name *
            </label>
            <input
              type="text"
              id="custom-gpu-name"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
              placeholder="My Custom GPU"
              required
            />
          </div>

          <div>
            <label
              htmlFor="custom-gpu-vram"
              className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
            >
              VRAM (GB) *
            </label>
            <input
              type="number"
              id="custom-gpu-vram"
              value={customVRAM}
              onChange={(e) => setCustomVRAM(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
              placeholder="24"
              min="1"
              step="1"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label
                htmlFor="custom-gpu-bandwidth"
                className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
              >
                Bandwidth (GB/s)
              </label>
              <input
                type="number"
                id="custom-gpu-bandwidth"
                value={customBandwidth}
                onChange={(e) => setCustomBandwidth(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
                placeholder="900"
              />
            </div>

            <div>
              <label
                htmlFor="custom-gpu-flops"
                className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
              >
                FP16 TFLOPS
              </label>
              <input
                type="number"
                id="custom-gpu-flops"
                value={customFLOPS}
                onChange={(e) => setCustomFLOPS(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
                placeholder="312"
              />
            </div>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleCustomGPUSubmit}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              Add GPU
            </button>
            <button
              type="button"
              onClick={() => setShowCustomForm(false)}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
