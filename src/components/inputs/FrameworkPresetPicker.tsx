import { FRAMEWORK_PRESETS, type FrameworkPreset } from '@engines/frameworks'
import { useUIStore } from '@store/uiStore'

/**
 * Framework preset picker for training/inference optimization
 *
 * Allows selecting from 7 framework presets including DeepSpeed ZeRO stages,
 * Unsloth optimizations, and inference-only frameworks (vLLM, TGI).
 *
 * Selecting a preset auto-configures optimizations and may switch modes.
 */
export function FrameworkPresetPicker() {
  const { frameworkPreset, setFrameworkPreset } = useUIStore()

  const selectedConfig = FRAMEWORK_PRESETS[frameworkPreset]

  return (
    <div className="space-y-2">
      <label
        htmlFor="framework-preset"
        className="block text-sm font-medium text-gray-700 dark:text-gray-300"
      >
        Framework Preset
      </label>
      <select
        id="framework-preset"
        value={frameworkPreset}
        onChange={(e) => setFrameworkPreset(e.target.value as FrameworkPreset)}
        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
      >
        <option value="none">None (Manual Configuration)</option>
        <option value="deepspeed-zero1">DeepSpeed ZeRO-1 (2x memory savings)</option>
        <option value="deepspeed-zero2">DeepSpeed ZeRO-2 (4x memory savings)</option>
        <option value="deepspeed-zero3">DeepSpeed ZeRO-3 (8-10x memory savings)</option>
        <option value="unsloth">Unsloth (optimized single-GPU)</option>
        <option value="vllm">vLLM (inference only)</option>
        <option value="tgi">TGI (inference only)</option>
      </select>

      {/* Description text */}
      <p className="text-xs text-gray-600 dark:text-gray-400">{selectedConfig.description}</p>

      {/* Auto-optimization badges */}
      {frameworkPreset.startsWith('deepspeed-') && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-2">
          <p className="text-xs text-blue-800 dark:text-blue-300">
            <span className="font-semibold">Auto-enabled:</span> Gradient Checkpointing, Flash
            Attention
          </p>
        </div>
      )}

      {frameworkPreset === 'unsloth' && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-2">
          <p className="text-xs text-blue-800 dark:text-blue-300">
            <span className="font-semibold">Auto-enabled:</span> Gradient Checkpointing, Flash
            Attention, 8-bit Optimizer
          </p>
        </div>
      )}

      {(frameworkPreset === 'vllm' || frameworkPreset === 'tgi') && (
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-md p-2">
          <p className="text-xs text-amber-800 dark:text-amber-300 font-medium">
            ⚠ Inference-only framework — switches to inference mode
          </p>
        </div>
      )}
    </div>
  )
}
