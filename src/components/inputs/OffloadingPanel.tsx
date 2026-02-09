import { useUIStore } from '@store/uiStore'

/**
 * Offloading configuration panel
 *
 * Allows users to configure CPU/RAM or NVMe offloading to reduce GPU VRAM usage.
 * Includes:
 * - Enable toggle
 * - Offload target selector (CPU/RAM vs NVMe)
 * - Offload mode toggle (By Percentage vs By Number of Layers)
 * - Percentage slider (0-100%) or layer count slider
 * - KV cache offload checkbox
 *
 * Reference UI: https://apxml.com/tools/vram-calculator
 */
export function OffloadingPanel() {
  const offloadingEnabled = useUIStore((s) => s.offloadingEnabled)
  const setOffloadingEnabled = useUIStore((s) => s.setOffloadingEnabled)
  const offloadTarget = useUIStore((s) => s.offloadTarget)
  const setOffloadTarget = useUIStore((s) => s.setOffloadTarget)
  const offloadMode = useUIStore((s) => s.offloadMode)
  const setOffloadMode = useUIStore((s) => s.setOffloadMode)
  const offloadPercentage = useUIStore((s) => s.offloadPercentage)
  const setOffloadPercentage = useUIStore((s) => s.setOffloadPercentage)
  const offloadLayers = useUIStore((s) => s.offloadLayers)
  const setOffloadLayers = useUIStore((s) => s.setOffloadLayers)
  const kvCacheOffload = useUIStore((s) => s.kvCacheOffload)
  const setKVCacheOffload = useUIStore((s) => s.setKVCacheOffload)
  const selectedModel = useUIStore((s) => s.selectedModel)

  const totalLayers = selectedModel?.num_hidden_layers ?? 80

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Offloading Configuration
        </h3>
        <label className="flex items-center gap-2 cursor-pointer">
          <span className="text-sm text-gray-700 dark:text-gray-300">Enable Offloading</span>
          <input
            type="checkbox"
            checked={offloadingEnabled}
            onChange={(e) => setOffloadingEnabled(e.target.checked)}
            className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
          />
        </label>
      </div>

      {offloadingEnabled && (
        <div className="space-y-4 pl-4 border-l-2 border-blue-200 dark:border-blue-800">
          {/* Offload Target Selector */}
          <div>
            <div className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Offload Target
            </div>
            <div className="grid grid-cols-1 gap-3">
              {/* CPU/RAM */}
              <button
                type="button"
                onClick={() => setOffloadTarget('cpu-ram')}
                className={`text-left p-3 border-2 rounded-lg transition-colors ${
                  offloadTarget === 'cpu-ram'
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <input
                    type="radio"
                    checked={offloadTarget === 'cpu-ram'}
                    onChange={() => setOffloadTarget('cpu-ram')}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="font-medium text-gray-900 dark:text-white">CPU/RAM</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 ml-6">
                  Offload to system memory via PCIe. Reduces VRAM usage but results in slower
                  inference speeds.
                </p>
              </button>

              {/* NVMe SSD */}
              <button
                type="button"
                onClick={() => setOffloadTarget('nvme')}
                className={`text-left p-3 border-2 rounded-lg transition-colors ${
                  offloadTarget === 'nvme'
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <input
                    type="radio"
                    checked={offloadTarget === 'nvme'}
                    onChange={() => setOffloadTarget('nvme')}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="font-medium text-gray-900 dark:text-white">NVMe SSD</span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 ml-6">
                  Offload to NVMe storage. Slower than CPU/RAM but allows offloading more data when
                  system memory is limited.
                </p>
              </button>
            </div>
          </div>

          {/* Offload Model Weights Section */}
          <div>
            <div className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Offload Model Weights
            </div>

            {/* Mode Toggle */}
            <div className="flex gap-2 mb-3">
              <button
                type="button"
                onClick={() => setOffloadMode('percentage')}
                className={`flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                  offloadMode === 'percentage'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                By Percentage
              </button>
              <button
                type="button"
                onClick={() => setOffloadMode('layers')}
                className={`flex-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                  offloadMode === 'layers'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                By Number of Layers
              </button>
            </div>

            {/* Percentage Mode */}
            {offloadMode === 'percentage' && (
              <div>
                <label
                  htmlFor="offload-percentage"
                  className="block text-sm text-gray-700 dark:text-gray-300 mb-1"
                >
                  Percentage of Model to Offload
                </label>
                <div className="flex items-center gap-4">
                  <input
                    id="offload-percentage"
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={offloadPercentage}
                    onChange={(e) => setOffloadPercentage(Number(e.target.value))}
                    className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <span className="text-sm font-semibold text-gray-900 dark:text-white w-12 text-right tabular-nums">
                    {offloadPercentage}%
                  </span>
                </div>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Offloading {offloadPercentage}% of model weights
                </p>
              </div>
            )}

            {/* Layers Mode */}
            {offloadMode === 'layers' && (
              <div>
                <label
                  htmlFor="offload-layers"
                  className="block text-sm text-gray-700 dark:text-gray-300 mb-1"
                >
                  Number of Layers to Offload
                </label>
                <div className="flex items-center gap-4">
                  <input
                    id="offload-layers"
                    type="range"
                    min={0}
                    max={totalLayers}
                    step={1}
                    value={offloadLayers}
                    onChange={(e) => setOffloadLayers(Number(e.target.value))}
                    className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <span className="text-sm font-semibold text-gray-900 dark:text-white w-16 text-right tabular-nums">
                    {offloadLayers}
                  </span>
                </div>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Offloading {offloadLayers} of {totalLayers} layers
                </p>
              </div>
            )}
          </div>

          {/* KV Cache Offload */}
          <div>
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={kvCacheOffload}
                onChange={(e) => setKVCacheOffload(e.target.checked)}
                className="mt-0.5 w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
              />
              <div>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  Offload KV Cache to CPU/RAM
                </span>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                  Moves the entire KV cache to system memory. Adds per-token latency during
                  generation.
                </p>
              </div>
            </label>
          </div>
        </div>
      )}
    </div>
  )
}
