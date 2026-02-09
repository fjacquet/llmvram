import modelsData from '@data/models.json'
import {
  Combobox,
  ComboboxButton,
  ComboboxInput,
  ComboboxOption,
  ComboboxOptions,
} from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { useUIStore } from '@store/uiStore'
import { type CustomModelInput, createCustomModel, type Model } from '@types/model'
import { validateModels } from '@utils/schemas'
import { useState } from 'react'

// Validate models data at module load
const models = validateModels(modelsData)

export function ModelSelector() {
  const { selectedModel, setSelectedModel } = useUIStore()
  const [query, setQuery] = useState('')
  const [showCustomForm, setShowCustomForm] = useState(false)

  // Custom model form state
  const [customName, setCustomName] = useState('')
  const [customParams, setCustomParams] = useState('')
  const [customHiddenSize, setCustomHiddenSize] = useState('')
  const [customLayers, setCustomLayers] = useState('')
  const [customHeads, setCustomHeads] = useState('')

  // Filter models based on search query
  const filteredModels =
    query === ''
      ? models
      : models.filter((model) => model.name.toLowerCase().includes(query.toLowerCase()))

  const handleCustomModelSubmit = () => {
    const paramsNum = Number.parseFloat(customParams)
    if (!customName || !paramsNum || paramsNum < 0.1) {
      return
    }

    const input: CustomModelInput = {
      name: customName,
      num_parameters_billion: paramsNum,
      hidden_size: customHiddenSize ? Number.parseInt(customHiddenSize, 10) : undefined,
      num_hidden_layers: customLayers ? Number.parseInt(customLayers, 10) : undefined,
      num_attention_heads: customHeads ? Number.parseInt(customHeads, 10) : undefined,
    }

    const customModel = createCustomModel(input)
    setSelectedModel(customModel)

    // Reset form
    setCustomName('')
    setCustomParams('')
    setCustomHiddenSize('')
    setCustomLayers('')
    setCustomHeads('')
    setShowCustomForm(false)
  }

  return (
    <div className="space-y-2">
      <label
        htmlFor="model-selector"
        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        Model
      </label>

      <Combobox
        value={selectedModel}
        onChange={(model) => {
          if (model === null) {
            // "Custom model..." option selected
            setShowCustomForm(true)
            setQuery('')
          } else {
            setSelectedModel(model)
            setShowCustomForm(false)
          }
        }}
      >
        <div className="relative">
          <div className="relative w-full cursor-default overflow-hidden rounded-lg bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-left shadow-sm focus-within:ring-2 focus-within:ring-blue-600">
            <ComboboxInput
              id="model-selector"
              className="w-full border-none py-2 pl-3 pr-10 text-sm leading-5 text-gray-900 dark:text-white bg-transparent focus:ring-0 focus:outline-none"
              displayValue={(model: Model | null) => model?.name || ''}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search models..."
            />
            <ComboboxButton className="absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
            </ComboboxButton>
          </div>

          <ComboboxOptions className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-lg bg-white dark:bg-gray-800 py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            {filteredModels.length === 0 && query !== '' ? (
              <div className="relative cursor-default select-none py-2 px-4 text-gray-700 dark:text-gray-300">
                No models found.
              </div>
            ) : (
              <>
                {filteredModels.map((model) => (
                  <ComboboxOption
                    key={model.id}
                    value={model}
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
                            {model.name}
                          </span>
                          <span
                            className={`text-xs ml-2 ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                          >
                            ({model.num_parameters_billion}B)
                          </span>
                        </div>
                        {model.architecture === 'moe' && (
                          <span
                            className={`text-xs ${active ? 'text-blue-200' : 'text-gray-500 dark:text-gray-400'}`}
                          >
                            MoE
                          </span>
                        )}
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

                {/* Custom model option */}
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
                      Custom model...
                    </span>
                  )}
                </ComboboxOption>
              </>
            )}
          </ComboboxOptions>
        </div>
      </Combobox>

      {/* Custom model form */}
      {showCustomForm && (
        <div className="mt-4 p-4 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-800/50 space-y-3">
          <h3 className="text-sm font-medium text-gray-900 dark:text-white">Custom Model</h3>

          <div>
            <label
              htmlFor="custom-model-name"
              className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
            >
              Name *
            </label>
            <input
              type="text"
              id="custom-model-name"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
              placeholder="My Custom Model"
              required
            />
          </div>

          <div>
            <label
              htmlFor="custom-model-params"
              className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
            >
              Parameter Count (billions) *
            </label>
            <input
              type="number"
              id="custom-model-params"
              value={customParams}
              onChange={(e) => setCustomParams(e.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
              placeholder="7"
              min="0.1"
              step="0.1"
              required
            />
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div>
              <label
                htmlFor="custom-model-hidden"
                className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
              >
                Hidden Size
              </label>
              <input
                type="number"
                id="custom-model-hidden"
                value={customHiddenSize}
                onChange={(e) => setCustomHiddenSize(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
                placeholder="4096"
              />
            </div>

            <div>
              <label
                htmlFor="custom-model-layers"
                className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
              >
                Layers
              </label>
              <input
                type="number"
                id="custom-model-layers"
                value={customLayers}
                onChange={(e) => setCustomLayers(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
                placeholder="32"
              />
            </div>

            <div>
              <label
                htmlFor="custom-model-heads"
                className="block text-xs text-gray-700 dark:text-gray-300 mb-1"
              >
                Heads
              </label>
              <input
                type="number"
                id="custom-model-heads"
                value={customHeads}
                onChange={(e) => setCustomHeads(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
                placeholder="32"
              />
            </div>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleCustomModelSubmit}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              Add Model
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
