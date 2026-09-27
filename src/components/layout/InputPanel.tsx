import { BatchSizeInput } from '@components/inputs/BatchSizeInput'
import { ConcurrentUsersInput } from '@components/inputs/ConcurrentUsersInput'
import { GPUCountSelector } from '@components/inputs/GPUCountSelector'
import { GPUSelector } from '@components/inputs/GPUSelector'
import { InterconnectSelector } from '@components/inputs/InterconnectSelector'
import { InterNodeFabricSelector } from '@components/inputs/InterNodeFabricSelector'
import { KVQuantizationPicker } from '@components/inputs/KVQuantizationPicker'
import { KVTierPanel } from '@components/inputs/KVTierPanel'
import { ModelSelector } from '@components/inputs/ModelSelector'
import { ModeToggle } from '@components/inputs/ModeToggle'
import { NodeCountSelector } from '@components/inputs/NodeCountSelector'
import { OffloadingPanel } from '@components/inputs/OffloadingPanel'
import { QuantizationPicker } from '@components/inputs/QuantizationPicker'
import { SequenceLengthInput } from '@components/inputs/SequenceLengthInput'
import { ShardingStrategySelector } from '@components/inputs/ShardingStrategySelector'
import { TrainingPanel } from '@components/inputs/TrainingPanel'
import { FRAMEWORK_PRESETS } from '@engines/frameworks'
import { useUIStore } from '@store/uiStore'
import { useEffect, useRef, useState } from 'react'
import { countAdvancedChanges } from './advancedChanges'

/**
 * Input panel: essential inputs always visible, advanced ones in a native <details>
 * that opens itself whenever one of them differs from its default (ADR 0005).
 * Inputs that are inert in training are hidden, never reset (spec Section 1).
 */
export function InputPanel() {
  const selectedGPU = useUIStore((s) => s.selectedGPU)
  const mode = useUIStore((s) => s.mode)
  const frameworkPreset = useUIStore((s) => s.frameworkPreset)
  const resetAdvancedSettings = useUIStore((s) => s.resetAdvancedSettings)
  const isInference = mode === 'inference'
  // In training, more than one GPU only matters with a DeepSpeed ZeRO preset
  const showGPUCount =
    selectedGPU !== null && (isInference || FRAMEWORK_PRESETS[frameworkPreset].zeroStage !== null)

  // Derived selector: re-renders only when the count itself changes, not on every
  // store write (UIState extends UIConfig, so countAdvancedChanges' shape typechecks).
  const changed = useUIStore(countAdvancedChanges)
  const [advancedOpen, setAdvancedOpen] = useState(changed > 0)
  // Auto-open only on the 0 -> >0 transition, so a user who closed it while settings
  // stayed non-default (e.g. 1 -> 2) is not overridden.
  const prevChanged = useRef(changed)
  useEffect(() => {
    if (prevChanged.current === 0 && changed > 0) setAdvancedOpen(true)
    prevChanged.current = changed
  }, [changed])

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div className="space-y-6">
        <ModeToggle />

        <hr className="border-gray-200 dark:border-gray-700" />

        <section aria-label="Essential settings" className="space-y-4">
          <ModelSelector />
          {isInference && <QuantizationPicker />}
          <GPUSelector />
          {showGPUCount && <GPUCountSelector />}
          {/* Multi-node is inference-only (multi-node training is a spec Non-Goal) */}
          {selectedGPU && isInference && <NodeCountSelector />}
          <SequenceLengthInput />
          {isInference && <ConcurrentUsersInput />}
        </section>

        {!isInference && (
          <>
            <hr className="border-gray-200 dark:border-gray-700" />
            <TrainingPanel />
          </>
        )}

        <hr className="border-gray-200 dark:border-gray-700" />

        <details
          data-testid="advanced-settings"
          open={advancedOpen}
          onToggle={(e) => setAdvancedOpen(e.currentTarget.open)}
        >
          <summary className="cursor-pointer text-lg font-semibold text-gray-900 dark:text-white">
            Advanced
            {changed > 0 && (
              <span className="ml-2 text-sm font-normal text-blue-600 dark:text-blue-400">
                {changed} setting{changed === 1 ? '' : 's'} changed
              </span>
            )}
          </summary>
          <div className="mt-4 space-y-4">
            {/* Spec Section 4: resets batch, KV precision, strategy, fabric, interconnect
                variant, offloading and KV tier — keeps model, GPU, GPU count, servers,
                format, context and concurrent users. Goes through commit()/normalizeConfig
                (resetAdvancedSettings), so the "Reset to defaults" toast is the existing
                useConfigNotices pipeline, not a button-local one. */}
            <button
              type="button"
              onClick={() => resetAdvancedSettings()}
              className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
            >
              Reset advanced settings
            </button>
            <BatchSizeInput />
            {isInference && (
              <>
                <KVQuantizationPicker />
                {selectedGPU && <ShardingStrategySelector />}
                <InterNodeFabricSelector />
                <InterconnectSelector />
                {selectedGPU && <OffloadingPanel />}
                {selectedGPU && <KVTierPanel />}
              </>
            )}
          </div>
        </details>
      </div>
    </div>
  )
}
