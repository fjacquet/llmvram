import { FitIndicator } from '@components/outputs/FitIndicator'
import { MemoryBreakdownTable } from '@components/outputs/MemoryBreakdownTable'
import { Recommendations } from '@components/outputs/Recommendations'
import { VRAMBreakdownChart } from '@components/outputs/VRAMBreakdownChart'
import { useInferenceCalculation } from '@hooks/useInferenceCalculation'
import { useUIStore } from '@store/uiStore'
import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * Results panel - the critical integration point
 *
 * Connects Zustand store selections to useInferenceCalculation hook
 * and renders output components with proper state handling.
 */
export function ResultsPanel() {
  // Read selections from store
  const { selectedModel, selectedGPU, quantization, sequenceLength, batchSize, kvQuantization } =
    useUIStore()

  // Call the calculation hook
  const { result, loading, error } = useInferenceCalculation(
    selectedModel,
    selectedGPU,
    quantization,
    sequenceLength,
    batchSize,
    kvQuantization,
  )

  // Show toast on error (only once per error change)
  useEffect(() => {
    if (error) {
      toast.error(error)
    }
  }, [error])

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

  const doesNotFit = result.vram.total.greaterThan(selectedGPU.vram_gb)

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="space-y-6">
        {/* VRAM Section */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            VRAM Requirements
          </h2>
          <div className="space-y-6">
            <FitIndicator totalVRAM={result.vram.total} availableVRAM={selectedGPU.vram_gb} />
            <VRAMBreakdownChart breakdown={result.vram} />
            <MemoryBreakdownTable breakdown={result.vram} />
            {doesNotFit && (
              <Recommendations
                gpu={selectedGPU}
                breakdown={result.vram}
                currentQuantization={quantization}
                currentSequenceLength={sequenceLength}
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
