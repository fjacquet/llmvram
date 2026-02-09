import type { ConfigSnapshot } from '@store/comparisonStore'
import { useComparisonStore } from '@store/comparisonStore'
import { useMemo, useState } from 'react'

interface ComparisonColumnProps {
  snapshot: ConfigSnapshot
  allSnapshots: ConfigSnapshot[]
  onRemove: () => void
}

interface FieldProps {
  label: string
  value: string | number
  isDifferent: boolean
}

/**
 * Field row with optional diff highlighting
 */
function Field({ label, value, isDifferent }: FieldProps) {
  return (
    <div
      className={`flex justify-between py-1 text-sm ${
        isDifferent
          ? 'bg-amber-50 dark:bg-amber-900/20 border-l-2 border-amber-400 dark:border-amber-500 pl-2'
          : ''
      }`}
    >
      <span className="text-gray-500 dark:text-gray-400">{label}</span>
      <span className="font-medium text-gray-900 dark:text-white">{value}</span>
    </div>
  )
}

/**
 * Single configuration snapshot column
 *
 * Renders a card with config details, VRAM breakdown, and performance.
 * Fields that differ from other snapshots are highlighted with amber/yellow.
 */
export function ComparisonColumn({ snapshot, allSnapshots, onRemove }: ComparisonColumnProps) {
  const [isEditingLabel, setIsEditingLabel] = useState(false)
  const [editedLabel, setEditedLabel] = useState(snapshot.label)
  const { updateLabel } = useComparisonStore()

  // Compute diff map: which fields differ from other snapshots
  const diffMap = useMemo(() => {
    const others = allSnapshots.filter((s) => s.id !== snapshot.id)
    if (others.length === 0) {
      // Only one snapshot, nothing to compare
      return {
        modelName: false,
        gpuName: false,
        gpuVramGb: false,
        quantization: false,
        sequenceLength: false,
        batchSize: false,
        numGPUs: false,
        kvQuantization: false,
        shardingStrategy: false,
        offloadingEnabled: false,
        offloadTarget: false,
        offloadPercentage: false,
        totalVRAM: false,
        fits: false,
        tokensPerSecond: false,
      }
    }

    return {
      modelName: others.some((s) => s.config.modelName !== snapshot.config.modelName),
      gpuName: others.some((s) => s.config.gpuName !== snapshot.config.gpuName),
      gpuVramGb: others.some((s) => s.config.gpuVramGb !== snapshot.config.gpuVramGb),
      quantization: others.some((s) => s.config.quantization !== snapshot.config.quantization),
      sequenceLength: others.some(
        (s) => s.config.sequenceLength !== snapshot.config.sequenceLength,
      ),
      batchSize: others.some((s) => s.config.batchSize !== snapshot.config.batchSize),
      numGPUs: others.some((s) => s.config.numGPUs !== snapshot.config.numGPUs),
      kvQuantization: others.some(
        (s) => s.config.kvQuantization !== snapshot.config.kvQuantization,
      ),
      shardingStrategy: others.some(
        (s) => s.config.shardingStrategy !== snapshot.config.shardingStrategy,
      ),
      offloadingEnabled: others.some(
        (s) => s.config.offloadingEnabled !== snapshot.config.offloadingEnabled,
      ),
      offloadTarget: others.some((s) => s.config.offloadTarget !== snapshot.config.offloadTarget),
      offloadPercentage: others.some(
        (s) => s.config.offloadPercentage !== snapshot.config.offloadPercentage,
      ),
      totalVRAM: others.some((s) => s.results.totalVRAM !== snapshot.results.totalVRAM),
      fits: others.some((s) => s.results.fits !== snapshot.results.fits),
      tokensPerSecond: others.some(
        (s) => s.results.tokensPerSecond !== snapshot.results.tokensPerSecond,
      ),
    }
  }, [snapshot, allSnapshots])

  // Calculate effective utilization (per-GPU if multi-GPU)
  const effectiveVRAM =
    snapshot.results.perGPUTotal !== null
      ? snapshot.results.perGPUTotal
      : snapshot.results.totalVRAM
  const effectiveCapacity =
    snapshot.config.numGPUs > 1 ? snapshot.config.gpuVramGb : snapshot.config.gpuVramGb
  const utilPercent = (effectiveVRAM / effectiveCapacity) * 100

  // Determine utilization color
  const getUtilColor = () => {
    if (!snapshot.results.fits || utilPercent > 100) return 'bg-red-500'
    if (utilPercent > 90) return 'bg-red-500'
    if (utilPercent > 70) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  // Handle label save
  const handleLabelSave = () => {
    if (editedLabel.trim() && editedLabel !== snapshot.label) {
      updateLabel(snapshot.id, editedLabel.trim())
    } else {
      setEditedLabel(snapshot.label) // Reset if empty or unchanged
    }
    setIsEditingLabel(false)
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4 space-y-3">
      {/* Header: Label + Remove button */}
      <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        {isEditingLabel ? (
          <input
            type="text"
            value={editedLabel}
            onChange={(e) => setEditedLabel(e.target.value)}
            onBlur={handleLabelSave}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleLabelSave()
              if (e.key === 'Escape') {
                setEditedLabel(snapshot.label)
                setIsEditingLabel(false)
              }
            }}
            className="text-lg font-semibold text-gray-900 dark:text-white bg-gray-50 dark:bg-gray-900 border border-blue-300 dark:border-blue-600 rounded px-2 py-1 flex-1 mr-2"
            // biome-ignore lint/a11y/noAutofocus: Intentional UX for inline editing
            autoFocus
          />
        ) : (
          <button
            type="button"
            onClick={() => setIsEditingLabel(true)}
            className="text-lg font-semibold text-gray-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 flex-1 text-left"
          >
            {snapshot.label}
          </button>
        )}
        <button
          type="button"
          onClick={onRemove}
          className="text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-colors"
          aria-label="Remove snapshot"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <title>Remove</title>
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Configuration fields */}
      <div className="space-y-1">
        <Field label="Model" value={snapshot.config.modelName} isDifferent={diffMap.modelName} />
        <Field label="GPU" value={snapshot.config.gpuName} isDifferent={diffMap.gpuName} />
        <Field
          label="Quantization"
          value={snapshot.config.quantization}
          isDifferent={diffMap.quantization}
        />
        <Field
          label="Seq Length"
          value={snapshot.config.sequenceLength.toLocaleString()}
          isDifferent={diffMap.sequenceLength}
        />
        <Field
          label="Batch Size"
          value={snapshot.config.batchSize}
          isDifferent={diffMap.batchSize}
        />
        <Field
          label="KV Quant"
          value={snapshot.config.kvQuantization}
          isDifferent={diffMap.kvQuantization}
        />

        {/* Multi-GPU fields */}
        {snapshot.config.numGPUs > 1 && (
          <Field
            label="GPUs"
            value={`${snapshot.config.numGPUs}x ${snapshot.config.shardingStrategy === 'tensor-parallel' ? 'TP' : 'PP'}`}
            isDifferent={diffMap.numGPUs || diffMap.shardingStrategy}
          />
        )}

        {/* Offloading fields */}
        {snapshot.config.offloadingEnabled && (
          <Field
            label="Offload"
            value={`${snapshot.config.offloadPercentage}% to ${snapshot.config.offloadTarget === 'cpu-ram' ? 'CPU' : 'NVMe'}`}
            isDifferent={
              diffMap.offloadingEnabled || diffMap.offloadPercentage || diffMap.offloadTarget
            }
          />
        )}
      </div>

      {/* VRAM section */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-3 space-y-2">
        <Field
          label="Total VRAM"
          value={`${snapshot.results.totalVRAM.toFixed(2)} GB`}
          isDifferent={diffMap.totalVRAM}
        />
        <div className="text-xs text-gray-500 dark:text-gray-400 space-y-0.5">
          <div className="flex justify-between">
            <span>Weights:</span>
            <span>{snapshot.results.modelWeights.toFixed(2)} GB</span>
          </div>
          <div className="flex justify-between">
            <span>KV Cache:</span>
            <span>{snapshot.results.kvCache.toFixed(2)} GB</span>
          </div>
          <div className="flex justify-between">
            <span>Activations:</span>
            <span>{snapshot.results.activations.toFixed(2)} GB</span>
          </div>
        </div>

        {/* Utilization bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Utilization</span>
            <span>{Math.min(utilPercent, 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className={`h-full ${getUtilColor()} transition-all duration-300`}
              style={{ width: `${Math.min(utilPercent, 100)}%` }}
            />
          </div>
        </div>

        {/* Fit status badge */}
        <div className="flex justify-center pt-1">
          {snapshot.results.fits ? (
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <title>Success</title>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              FITS
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <title>Warning</title>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              EXCEEDS
            </span>
          )}
        </div>
      </div>

      {/* Performance section */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
        <div className="text-sm space-y-1">
          <Field
            label="Performance"
            value={`${snapshot.results.tokensPerSecond.toFixed(1)} tok/s`}
            isDifferent={diffMap.tokensPerSecond}
          />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>TTFT:</span>
            <span>{snapshot.results.timeToFirstToken.toFixed(0)} ms</span>
          </div>
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Bottleneck:</span>
            <span className="capitalize">{snapshot.results.bottleneck}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
