import { InfoTip } from '@components/common/InfoTip'
import { useAllowedOptions } from '@hooks/useAllowedOptions'
import { useUIStore } from '@store/uiStore'
import { maxGPUsFor } from '@utils/gpuLimits'

/**
 * GPU count selector, bounded by the selected GPU's scale-up domain
 *
 * In inference mode this is the PER-NODE count: total GPUs is this times the
 * server count from NodeCountSelector. The upper bound is the selected GPU's
 * max_gpus_per_node — 8 for an HGX or OAM baseboard, 72 for a GB300 NVL72
 * rack, 2 for a pair of DGX Sparks, 1 for Apple Silicon and the GB300 Desktop
 * Superchip.
 *
 * In training mode there is no server concept: useTrainingCalculation reads
 * this value directly as the total GPU count for ZeRO data-parallel sharding
 * (multi-node training is an explicit spec Non-Goal, and NodeCountSelector is
 * hidden in this mode). The same per-GPU bound still applies, because the
 * GPUs still have to share one node.
 */
export function GPUCountSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const setNumGPUs = useUIStore((s) => s.setNumGPUs)
  const selectedGPU = useUIStore((s) => s.selectedGPU)
  const mode = useUIStore((s) => s.mode)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)
  const { gpuCounts } = useAllowedOptions()

  const isTraining = mode === 'training'
  const maxGPUs = maxGPUsFor(selectedGPU)
  const label = isTraining ? 'Number of GPUs' : 'GPUs per server'

  const tooltip = isTraining
    ? 'GPUs used for data-parallel training, e.g. DeepSpeed ZeRO sharding. Multi-node training is not modelled, so this is the total GPU count.'
    : `GPUs inside one server. Tensor or pipeline parallelism runs at this level, over NVLink, Infinity Fabric or PCIe. Capped at ${maxGPUs} — the largest GPU count this hardware forms in one node.`

  if (maxGPUs === 1) {
    return (
      <div>
        <div className="flex items-center gap-1 mb-1">
          <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </span>
          <InfoTip text={tooltip} />
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Single GPU — {selectedGPU?.name ?? 'this part'} ships as a single GPU; no multi-GPU
          configuration exists for it.
        </p>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center gap-1 mb-1">
        <label
          htmlFor="gpu-count"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300"
        >
          {label}
        </label>
        <InfoTip text={tooltip} />
      </div>
      <div className="flex items-center gap-4">
        {/* The slider moves over the allowed counts only (R1 + R14): a tensor-parallel
            degree vLLM refuses is not selectable. */}
        <input
          id="gpu-count"
          type="range"
          min={0}
          max={gpuCounts.length - 1}
          step={1}
          value={Math.max(0, gpuCounts.indexOf(numGPUs))}
          aria-valuetext={`${numGPUs} GPUs`}
          onChange={(e) => setNumGPUs(gpuCounts[Number(e.target.value)] ?? 1)}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <span className="text-lg font-semibold text-gray-900 dark:text-white w-10 text-center tabular-nums">
          {numGPUs}
        </span>
      </div>
      {numGPUs > 1 && (
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          {isTraining
            ? `${numGPUs} GPUs`
            : `${numGPUs} GPUs per server, ${
                shardingStrategy === 'tensor-parallel' ? 'tensor parallel' : 'pipeline parallel'
              }`}
        </p>
      )}
    </div>
  )
}
