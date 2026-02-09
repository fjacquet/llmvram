import { INTERCONNECT_SPECS } from '@engines/constants'
import { resolveInterconnect } from '@engines/multi-gpu'
import { useUIStore } from '@store/uiStore'

/**
 * Sharding strategy selector (Tensor Parallel vs Pipeline Parallel)
 *
 * Only visible when numGPUs > 1. Shows:
 * - Radio buttons for TP/PP selection with descriptions
 * - Interconnect information badge from selected GPU
 * - Performance warnings for suboptimal configurations
 */
export function ShardingStrategySelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)
  const setShardingStrategy = useUIStore((s) => s.setShardingStrategy)
  const selectedGPU = useUIStore((s) => s.selectedGPU)

  // Only render when multi-GPU is active
  if (numGPUs <= 1) {
    return null
  }

  // Resolve interconnect type and spec
  const interconnectType = selectedGPU ? resolveInterconnect(selectedGPU) : 'none'
  const interconnectSpec = INTERCONNECT_SPECS[interconnectType]

  // Determine badge color based on interconnect type
  let badgeColorClass = 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  if (interconnectType.startsWith('nvlink')) {
    badgeColorClass = 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
  } else if (interconnectType.startsWith('pcie')) {
    badgeColorClass = 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
  } else if (interconnectType === 'none') {
    badgeColorClass = 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }

  // Check if TP degree exceeds recommended maximum
  const tpExceedsMax =
    shardingStrategy === 'tensor-parallel' && numGPUs > interconnectSpec.recommendedMaxTPDegree

  return (
    <div className="space-y-3">
      <div className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        Sharding Strategy
      </div>

      {/* Strategy selection cards */}
      <div className="grid grid-cols-1 gap-3">
        {/* Tensor Parallel */}
        <button
          type="button"
          onClick={() => setShardingStrategy('tensor-parallel')}
          className={`text-left p-3 border-2 rounded-lg transition-colors ${
            shardingStrategy === 'tensor-parallel'
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`}
        >
          <div className="flex items-center gap-2 mb-1">
            <input
              type="radio"
              checked={shardingStrategy === 'tensor-parallel'}
              onChange={() => setShardingStrategy('tensor-parallel')}
              className="text-blue-600 focus:ring-blue-500"
            />
            <span className="font-medium text-gray-900 dark:text-white">Tensor Parallel</span>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-400 ml-6">
            Splits model layers horizontally across GPUs. Best for single-node with fast
            interconnect (NVLink).
          </p>
        </button>

        {/* Pipeline Parallel */}
        <button
          type="button"
          onClick={() => setShardingStrategy('pipeline-parallel')}
          className={`text-left p-3 border-2 rounded-lg transition-colors ${
            shardingStrategy === 'pipeline-parallel'
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`}
        >
          <div className="flex items-center gap-2 mb-1">
            <input
              type="radio"
              checked={shardingStrategy === 'pipeline-parallel'}
              onChange={() => setShardingStrategy('pipeline-parallel')}
              className="text-blue-600 focus:ring-blue-500"
            />
            <span className="font-medium text-gray-900 dark:text-white">Pipeline Parallel</span>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-400 ml-6">
            Splits model layers sequentially into pipeline stages. KV cache is NOT shared -- each
            GPU stores full cache.
          </p>
        </button>
      </div>

      {/* Interconnect information badge */}
      <div
        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${badgeColorClass}`}
      >
        {interconnectType === 'none' ? (
          'No interconnect detected. Multi-GPU may not be supported for this GPU.'
        ) : interconnectType.startsWith('nvlink') ? (
          <>
            {interconnectType.toUpperCase()}: {interconnectSpec.bandwidthGBps} GB/s -- Excellent for
            TP up to {interconnectSpec.recommendedMaxTPDegree} GPUs
          </>
        ) : (
          <>
            {interconnectType.toUpperCase()}: {interconnectSpec.bandwidthGBps} GB/s -- TP
            recommended up to {interconnectSpec.recommendedMaxTPDegree} GPUs
          </>
        )}
      </div>

      {/* Warning for TP degree exceeding recommended max */}
      {tpExceedsMax && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3">
          <p className="text-xs text-amber-800 dark:text-amber-200">
            ⚠ Tensor Parallel with {numGPUs} GPUs may experience performance degradation on{' '}
            {interconnectType.toUpperCase()}. Recommended maximum:{' '}
            {interconnectSpec.recommendedMaxTPDegree} GPUs.
          </p>
        </div>
      )}
    </div>
  )
}
