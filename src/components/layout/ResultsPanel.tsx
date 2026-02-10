import { FitIndicator } from '@components/outputs/FitIndicator'
import { MemoryBreakdownTable } from '@components/outputs/MemoryBreakdownTable'
import { MultiGPUBreakdownChart } from '@components/outputs/MultiGPUBreakdownChart'
import { Recommendations } from '@components/outputs/Recommendations'
import { TrainingBreakdownChart } from '@components/outputs/TrainingBreakdownChart'
import { TrainingBreakdownTable } from '@components/outputs/TrainingBreakdownTable'
import { VRAMBreakdownChart } from '@components/outputs/VRAMBreakdownChart'
import type { OffloadingConfig } from '@engines/types'
import { PlusIcon } from '@heroicons/react/24/outline'
import { useInferenceCalculation } from '@hooks/useInferenceCalculation'
import { useTrainingCalculation } from '@hooks/useTrainingCalculation'
import type { ConfigSnapshot } from '@store/comparisonStore'
import { useComparisonStore } from '@store/comparisonStore'
import { useUIStore } from '@store/uiStore'
import { useEffect, useMemo } from 'react'
import { toast } from 'sonner'

/**
 * Results panel - the critical integration point
 *
 * Connects Zustand store selections to useInferenceCalculation hook
 * and renders output components with proper state handling.
 * Supports multi-GPU and offloading configurations.
 */
export function ResultsPanel() {
  // Read all selections from store
  const {
    mode,
    selectedModel,
    selectedGPU,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization,
    numGPUs,
    shardingStrategy,
    offloadingEnabled,
    offloadTarget,
    offloadMode,
    offloadPercentage,
    offloadLayers,
    kvCacheOffload,
  } = useUIStore()

  // Read comparison store for save functionality
  const { snapshots, addSnapshot } = useComparisonStore()

  // Build offloading config if enabled (memoized to prevent render loops)
  const offloadingConfig = useMemo<OffloadingConfig | undefined>(
    () =>
      offloadingEnabled
        ? {
            enabled: offloadingEnabled,
            target: offloadTarget,
            mode: offloadMode,
            offloadPercentage,
            offloadLayers,
            kvCacheOffload,
          }
        : undefined,
    [
      offloadingEnabled,
      offloadTarget,
      offloadMode,
      offloadPercentage,
      offloadLayers,
      kvCacheOffload,
    ],
  )

  // Call the calculation hook with multi-GPU and offloading params
  const { result, loading, error } = useInferenceCalculation(
    selectedModel,
    selectedGPU,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization,
    numGPUs,
    shardingStrategy,
    offloadingConfig,
  )

  // Call training calculation hook (unconditional - React hooks cannot be conditional)
  const { result: trainingResult, error: trainingError } = useTrainingCalculation()

  // Show toast on error (only once per error change)
  useEffect(() => {
    if (error) {
      toast.error(error)
    }
  }, [error])

  // Show toast on training error
  useEffect(() => {
    if (trainingError) {
      toast.error(trainingError)
    }
  }, [trainingError])

  // State 1: No model or GPU selected
  if (!selectedModel || !selectedGPU) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
        <div className="text-gray-500 dark:text-gray-400">
          <svg
            className="mx-auto h-12 w-12 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            role="img"
            aria-label="Calculator icon"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
            />
          </svg>
          <p className="text-lg font-medium">
            Select a model and GPU to calculate VRAM requirements
          </p>
        </div>
      </div>
    )
  }

  // State 2: Loading
  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="space-y-6 animate-pulse">
          <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded" />
          <div className="h-48 bg-gray-200 dark:bg-gray-700 rounded" />
        </div>
      </div>
    )
  }

  // State 3: Error
  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6">
        <div className="flex items-start">
          <svg
            className="h-6 w-6 text-red-600 dark:text-red-400 mr-3 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            role="img"
            aria-label="Error icon"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div>
            <h3 className="text-lg font-semibold text-red-900 dark:text-red-200 mb-1">
              Calculation Error
            </h3>
            <p className="text-red-800 dark:text-red-300">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  // State 4: Results
  if (!result) {
    return null
  }

  // Determine which breakdown to use for FitIndicator and display
  const displayBreakdown = result.offloading ? result.offloading.onDevice : result.vram

  // Determine doesNotFit logic based on active features
  let doesNotFit = false
  if (result.multiGPU) {
    // Multi-GPU: check if per-GPU total exceeds GPU capacity
    doesNotFit = result.multiGPU.totalPerGPU.greaterThan(selectedGPU.vram_gb)
  } else if (result.offloading) {
    // Offloading only: check if on-device total exceeds GPU capacity
    doesNotFit = result.offloading.onDevice.total.greaterThan(selectedGPU.vram_gb)
  } else {
    // Single GPU, no offloading: check if total exceeds GPU capacity
    doesNotFit = result.vram.total.greaterThan(selectedGPU.vram_gb)
  }

  /**
   * Generate descriptive label for snapshot
   */
  const generateLabel = (): string => {
    const modelName = selectedModel.name
    const gpuName = selectedGPU.name
    const quantName = quantization.toUpperCase()
    return `${modelName} / ${gpuName} / ${quantName}`
  }

  /**
   * Save current configuration and results to comparison store
   */
  const handleSaveToComparison = () => {
    if (!selectedModel || !selectedGPU || !result) return

    // Convert all Decimal results to plain numbers for serialization
    const snapshotData: Omit<ConfigSnapshot, 'id' | 'timestamp'> = {
      label: generateLabel(),
      config: {
        modelName: selectedModel.name,
        modelId: selectedModel.id,
        gpuName: selectedGPU.name,
        gpuId: selectedGPU.id,
        gpuVramGb: selectedGPU.vram_gb,
        quantization,
        sequenceLength,
        batchSize,
        kvQuantization,
        numGPUs,
        shardingStrategy,
        offloadingEnabled,
        offloadTarget,
        offloadPercentage,
      },
      results: {
        totalVRAM: result.vram.total.toNumber(),
        modelWeights: result.vram.modelWeights.toNumber(),
        kvCache: result.vram.kvCache.toNumber(),
        activations: result.vram.activations.toNumber(),
        frameworkOverhead: result.vram.frameworkOverhead.toNumber(),
        tokensPerSecond: result.performance.tokensPerSecond.toNumber(),
        timeToFirstToken: result.performance.timeToFirstToken.toNumber(),
        bottleneck: result.performance.bottleneck,
        fits: !doesNotFit,
        perGPUTotal: result.multiGPU?.totalPerGPU.toNumber() ?? null,
        utilizationPercent: result.multiGPU?.utilizationPercent.toNumber() ?? null,
      },
    }

    addSnapshot(snapshotData)
    toast.success('Configuration saved to comparison')
  }

  /**
   * Render training results (when mode is 'training')
   */
  const renderTrainingResults = () => {
    // Error state
    if (trainingError) {
      return (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6">
          <div className="flex items-start">
            <svg
              className="h-6 w-6 text-red-600 dark:text-red-400 mr-3 flex-shrink-0"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              role="img"
              aria-label="Error icon"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <h3 className="text-lg font-semibold text-red-900 dark:text-red-200 mb-1">
                Calculation Error
              </h3>
              <p className="text-red-800 dark:text-red-300">{trainingError}</p>
            </div>
          </div>
        </div>
      )
    }

    // No result yet
    if (!trainingResult || !selectedModel || !selectedGPU) {
      return null
    }

    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div className="space-y-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Training VRAM Requirements
            </h2>
            <div className="space-y-6">
              {/* Fit Indicator */}
              <FitIndicator totalVRAM={trainingResult.total} availableVRAM={selectedGPU.vram_gb} />

              {/* Training method badge */}
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Method:
                </span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300">
                  {trainingResult.method === 'full'
                    ? 'Full Fine-Tuning'
                    : trainingResult.method === 'lora'
                      ? 'LoRA'
                      : 'QLoRA'}
                </span>
              </div>

              {/* Training breakdown chart */}
              <TrainingBreakdownChart breakdown={trainingResult} />

              {/* Training breakdown table */}
              <TrainingBreakdownTable breakdown={trainingResult} />
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Branch on mode: training vs inference
  if (mode === 'training') {
    return renderTrainingResults()
  }

  // Continue with inference mode (everything below is unchanged)
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="space-y-6">
        {/* VRAM Section */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              VRAM Requirements
            </h2>
            <button
              type="button"
              onClick={handleSaveToComparison}
              disabled={snapshots.length >= 3}
              className="flex items-center gap-1.5 text-sm px-3 py-1.5 rounded-md bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 border border-blue-200 dark:border-blue-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {snapshots.length >= 3 ? (
                'Max 3 saved'
              ) : (
                <>
                  <PlusIcon className="h-4 w-4" />
                  Save to Compare
                </>
              )}
            </button>
          </div>
          <div className="space-y-6">
            {/* Fit Indicator - shows per-GPU utilization for multi-GPU, on-device for offloading */}
            {result.multiGPU ? (
              <FitIndicator
                totalVRAM={result.multiGPU.totalPerGPU}
                availableVRAM={selectedGPU.vram_gb}
              />
            ) : (
              <FitIndicator
                totalVRAM={displayBreakdown.total}
                availableVRAM={selectedGPU.vram_gb}
              />
            )}

            {/* Offloading Summary */}
            {result.offloading && (
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-200 mb-2">
                  Offloading Active
                </h4>
                <div className="text-sm text-blue-800 dark:text-blue-300 space-y-1">
                  <p>
                    <span className="font-medium">Offloaded:</span>{' '}
                    {result.offloading.offloaded.total.toFixed(2)} GB to{' '}
                    {offloadTarget === 'cpu-ram' ? 'CPU/RAM' : 'NVMe'}
                  </p>
                  <p>
                    <span className="font-medium">On-device:</span>{' '}
                    {result.offloading.onDevice.total.toFixed(2)} GB
                  </p>
                  <p className="text-xs italic">{result.offloading.performanceImpact}</p>
                </div>
              </div>
            )}

            {/* Single-GPU Breakdown Chart and Table */}
            <VRAMBreakdownChart breakdown={displayBreakdown} />
            <MemoryBreakdownTable breakdown={displayBreakdown} />

            {/* Multi-GPU Breakdown Chart */}
            {result.multiGPU && (
              <MultiGPUBreakdownChart breakdown={result.multiGPU} gpuVRAM={selectedGPU.vram_gb} />
            )}

            {/* Interconnect Warning */}
            {result.interconnectWarning && (
              <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  ⚠ {result.interconnectWarning}
                </p>
              </div>
            )}

            {/* Recommendations */}
            {doesNotFit && (
              <Recommendations
                gpu={selectedGPU}
                breakdown={result.vram}
                currentQuantization={quantization}
                currentSequenceLength={sequenceLength}
                numGPUs={numGPUs}
                multiGPUBreakdown={result.multiGPU}
              />
            )}
          </div>
        </div>

        {/* Performance Section */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Performance Estimate
          </h2>
          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Decode Speed</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {result.performance.tokensPerSecond.toFixed(1)} tokens/sec
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Time to First Token</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-white">
                  {result.performance.timeToFirstToken.toFixed(1)} ms
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">Bottleneck</p>
                <p
                  className={`text-lg font-semibold ${
                    result.performance.bottleneck === 'balanced'
                      ? 'text-green-600 dark:text-green-400'
                      : result.performance.bottleneck === 'memory'
                        ? 'text-yellow-600 dark:text-yellow-400'
                        : 'text-blue-600 dark:text-blue-400'
                  }`}
                >
                  {result.performance.bottleneck === 'balanced'
                    ? 'Balanced'
                    : result.performance.bottleneck === 'memory'
                      ? 'Memory bandwidth'
                      : 'Compute'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
