import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

/**
 * Server count selector with range slider (1-8 servers)
 *
 * Pipeline parallelism runs across servers. Tensor parallelism across a server
 * boundary is not offered: its per-layer allreduce over a network fabric an
 * order of magnitude slower than NVLink is not a viable configuration.
 */
export function NodeCountSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const numNodes = useUIStore((s) => s.numNodes)
  const setNumNodes = useUIStore((s) => s.setNumNodes)

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="node-count"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Number of servers
        </label>
        <InfoTip text="Identical servers in the cluster. Model layers are split across them with pipeline parallelism, so more servers fit larger models but add a network hop between layer groups." />
      </div>
      <div className="flex items-center gap-4">
        <input
          id="node-count"
          type="range"
          min={1}
          max={8}
          step={1}
          value={numNodes}
          onChange={(e) => setNumNodes(Number(e.target.value))}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <span className="text-lg font-semibold text-gray-900 dark:text-white w-8 text-center tabular-nums">
          {numNodes}
        </span>
      </div>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        {numNodes * numGPUs} GPUs total
        {numNodes > 1 ? ' · pipeline parallel across servers' : ''}
      </p>
    </div>
  )
}
