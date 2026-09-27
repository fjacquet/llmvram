import { InfoTip } from '@components/common/InfoTip'
import { INTERCONNECT_LABELS, INTERCONNECT_SPECS } from '@engines/constants'
import { interconnectLabel as linkLabel, resolveInterconnect } from '@engines/multi-gpu'
import { useUIStore } from '@store/uiStore'

/**
 * Sharding strategy selector (Tensor Parallel vs Pipeline Parallel)
 *
 * Only visible when there is more than one GPU per server. This selects the
 * INTRA-node strategy; across servers the strategy is always pipeline parallel.
 * Shows:
 * - Radio buttons for TP/PP selection with descriptions
 * - Interconnect information badge from selected GPU
 * - Performance warnings for suboptimal configurations
 */
export function ShardingStrategySelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)
  const setShardingStrategy = useUIStore((s) => s.setShardingStrategy)
  const selectedGPU = useUIStore((s) => s.selectedGPU)
  const isMoE = useUIStore((s) => s.selectedModel?.architecture === 'moe')

  // Only render when multi-GPU is active
  if (numGPUs <= 1) {
    return null
  }

  // Resolve the link for THIS group size: an NVLink bridge only carries a group that fits it
  const interconnectType = selectedGPU ? resolveInterconnect(selectedGPU, numGPUs) : 'none'
  const interconnectSpec = INTERCONNECT_SPECS[interconnectType]
  const bridgeSize =
    selectedGPU?.nvlink_bridge && numGPUs <= selectedGPU.nvlink_bridge.size
      ? selectedGPU.nvlink_bridge.size
      : null

  // Determine badge color based on interconnect type
  let badgeColorClass = 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  if (interconnectType.startsWith('nvlink')) {
    badgeColorClass = 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
  } else if (interconnectType === 'infinity-fabric') {
    // AMD's high-bandwidth scale-up fabric — same tier as NVLink (8-way fully
    // connected, comparable GB/s), so it gets its own color close to NVLink's
    // rather than falling through to the unknown-interconnect gray.
    badgeColorClass = 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200'
  } else if (interconnectType.startsWith('pcie')) {
    badgeColorClass = 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
  } else if (interconnectType === 'none') {
    badgeColorClass = 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }

  // The badge renders bandwidthGBps separately, so only the name part of the label is used
  const interconnectLabel = (
    selectedGPU ? linkLabel(selectedGPU, numGPUs) : (INTERCONNECT_LABELS.none ?? 'None')
  ).split(' — ')[0]

  // Check if TP degree exceeds recommended maximum
  const tpExceedsMax =
    shardingStrategy === 'tensor-parallel' && numGPUs > interconnectSpec.recommendedMaxTPDegree

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1 mb-2">
        <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Intra-server sharding strategy
        </div>
        <InfoTip text="Tensor Parallel splits each layer across GPUs (needs fast NVLink). Pipeline Parallel assigns whole layers to GPUs (works over PCIe but adds pipeline bubbles)." />
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
            Splits model layers sequentially into pipeline stages. Each GPU holds only its own
            layers&apos; KV cache; decode speeds up only with enough concurrent sequences.
          </p>
        </button>

        {/* Expert Parallel + DP attention (MoE only) */}
        {isMoE && (
          <button
            type="button"
            onClick={() => setShardingStrategy('expert-parallel')}
            className={`text-left p-3 border-2 rounded-lg transition-colors ${
              shardingStrategy === 'expert-parallel'
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              <input
                type="radio"
                checked={shardingStrategy === 'expert-parallel'}
                onChange={() => setShardingStrategy('expert-parallel')}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="font-medium text-gray-900 dark:text-white">
                Expert Parallel + DP attention
              </span>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 ml-6">
              Splits the experts across GPUs and replicates attention; each GPU serves its own
              sessions, so MLA KV cache is not duplicated. How DeepSeek, Kimi and GLM are served.
            </p>
          </button>
        )}
      </div>

      {/* Interconnect information badge */}
      <div
        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${badgeColorClass}`}
      >
        {interconnectType === 'none' ? (
          'No interconnect detected. Multi-GPU may not be supported for this GPU.'
        ) : interconnectType.startsWith('nvlink') || interconnectType === 'infinity-fabric' ? (
          <>
            {interconnectLabel}: {interconnectSpec.bandwidthGBps} GB/s ·{' '}
            {Math.round(interconnectSpec.tpScalingEfficiency * 100)}% TP efficiency — Excellent for
            TP up to {bridgeSize ?? interconnectSpec.recommendedMaxTPDegree} GPUs
          </>
        ) : (
          <>
            {interconnectLabel}: {interconnectSpec.bandwidthGBps} GB/s ·{' '}
            {Math.round(interconnectSpec.tpScalingEfficiency * 100)}% TP efficiency — TP recommended
            up to {interconnectSpec.recommendedMaxTPDegree} GPUs
          </>
        )}
      </div>

      {/* Warning for TP degree exceeding recommended max */}
      {tpExceedsMax && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3">
          <p className="text-xs text-amber-800 dark:text-amber-200">
            ⚠ Tensor Parallel with {numGPUs} GPUs per server may experience performance degradation
            on {interconnectLabel}. Recommended maximum: {interconnectSpec.recommendedMaxTPDegree}{' '}
            GPUs.
          </p>
        </div>
      )}
    </div>
  )
}
