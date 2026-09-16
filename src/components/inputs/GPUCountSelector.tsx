import { InfoTip } from '@components/common/InfoTip'
import { useUIStore } from '@store/uiStore'

/**
 * GPU count selector with range slider (1-8 GPUs per server)
 *
 * In inference mode this is the PER-NODE count: total GPUs is this times the
 * server count from NodeCountSelector, and the 1-8 bound is real — both
 * NVLink and Infinity Fabric top out at an 8-GPU fully connected domain.
 *
 * In training mode there is no server concept: useTrainingCalculation reads
 * this value directly as the total GPU count for ZeRO data-parallel sharding
 * (multi-node training is an explicit spec Non-Goal, and NodeCountSelector is
 * hidden in this mode). The label and copy switch accordingly rather than
 * claiming a per-server meaning training does not honor.
 */
export function GPUCountSelector() {
  const numGPUs = useUIStore((s) => s.numGPUs)
  const setNumGPUs = useUIStore((s) => s.setNumGPUs)
  const mode = useUIStore((s) => s.mode)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)

  const isTraining = mode === 'training'
  const label = isTraining ? 'Number of GPUs' : 'GPUs per server'
  const tooltip = isTraining
    ? 'GPUs used for data-parallel training, e.g. DeepSpeed ZeRO sharding. Multi-node training is not modelled, so this is the total GPU count.'
    : 'GPUs inside one server. Tensor or pipeline parallelism runs at this level, over NVLink or Infinity Fabric. Capped at 8 — that is the size of a fully connected GPU domain in current hardware.'

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
        <input
          id="gpu-count"
          type="range"
          min={1}
          max={8}
          step={1}
          value={numGPUs}
          onChange={(e) => setNumGPUs(Number(e.target.value))}
          className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
        />
        <span className="text-lg font-semibold text-gray-900 dark:text-white w-8 text-center tabular-nums">
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
