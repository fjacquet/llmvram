import { InfoTip } from '@components/common/InfoTip'
import { FABRIC_SPECS, perNodeFabricGBps } from '@engines/fabric'
import type { FabricType } from '@engines/types'
import { useUIStore } from '@store/uiStore'

/**
 * Scale-out fabric selector, shown only when the cluster spans servers
 *
 * Labels show per-node aggregate bandwidth, not port speed. The standard AI
 * node has one NIC per GPU and collectives stripe across all of them, so an
 * 8-GPU node on 800GbE has 800 GB/s of scale-out, not 100. Showing only the
 * port speed would understate node bandwidth eightfold.
 */
export function InterNodeFabricSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const numNodes = useUIStore((s) => s.numNodes)
  const interNodeFabric = useUIStore((s) => s.interNodeFabric)
  const setInterNodeFabric = useUIStore((s) => s.setInterNodeFabric)
  const customFabric = useUIStore((s) => s.customFabric)
  const setCustomFabric = useUIStore((s) => s.setCustomFabric)

  if (numNodes <= 1) {
    return null
  }

  const options = Object.values(FABRIC_SPECS)

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="inter-node-fabric"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          Network between servers
        </label>
        <InfoTip text="The scale-out fabric carrying activations between servers. Assumes one NIC per GPU, the standard build, so per-server bandwidth is the port speed times the GPU count." />
      </div>
      <select
        id="inter-node-fabric"
        value={interNodeFabric}
        onChange={(e) => setInterNodeFabric(e.target.value as FabricType)}
        className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white"
      >
        {options.map((spec) => (
          <option key={spec.type} value={spec.type}>
            {spec.label} — {perNodeFabricGBps(spec.portGBps, numGPUs)} GB/s per server
          </option>
        ))}
        <option value="custom">Custom…</option>
      </select>

      {interNodeFabric === 'custom' && (
        <div className="mt-2">
          <label
            htmlFor="custom-fabric-gbps"
            className="block text-xs text-gray-600 dark:text-gray-400 mb-1"
          >
            Port bandwidth, GB/s (unidirectional, per NIC)
          </label>
          <input
            id="custom-fabric-gbps"
            type="number"
            min={0.1}
            max={10000}
            step={0.1}
            value={customFabric?.port_gbps ?? ''}
            onChange={(e) => {
              const value = Number(e.target.value)
              setCustomFabric(
                Number.isFinite(value) && value > 0
                  ? { name: 'Custom fabric', port_gbps: value }
                  : null,
              )
            }}
            className="w-full rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white"
          />
          {customFabric && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {perNodeFabricGBps(customFabric.port_gbps, numGPUs)} GB/s per server across {numGPUs}{' '}
              NICs
            </p>
          )}
        </div>
      )}

      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        Efficiency figures are derived from bandwidth, not measured benchmarks.
      </p>
    </div>
  )
}
