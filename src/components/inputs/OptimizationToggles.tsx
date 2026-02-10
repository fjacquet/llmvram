import { InfoTip } from '@components/common/InfoTip'
import { Description, Field, Label, Switch } from '@headlessui/react'
import { useUIStore } from '@store/uiStore'

/**
 * Optimization toggles for gradient checkpointing and Flash Attention
 *
 * Each toggle provides contextual information about memory savings and trade-offs
 */
export function OptimizationToggles() {
  const {
    gradientCheckpointing,
    setGradientCheckpointing,
    flashAttention,
    setFlashAttention,
    sequenceLength,
  } = useUIStore()

  // Calculate Flash Attention benefit based on sequence length
  const getFlashAttentionBenefit = (): string => {
    if (sequenceLength < 2048) {
      return '~15% activation reduction at current sequence length'
    }
    if (sequenceLength < 8192) {
      return '~50% activation reduction at current sequence length'
    }
    return '~70% activation reduction at current sequence length'
  }

  return (
    <div className="space-y-4">
      {/* Gradient Checkpointing Toggle */}
      <Field>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Gradient Checkpointing
            </Label>
            <InfoTip text="Recompute activations during backward pass instead of storing them. Saves ~60% activation memory at the cost of 20-25% more compute time." />
          </div>
          <Switch
            checked={gradientCheckpointing}
            onChange={setGradientCheckpointing}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
              gradientCheckpointing ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
                gradientCheckpointing ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </Switch>
        </div>
        <Description className="text-xs text-gray-600 dark:text-gray-400 mt-1">
          {gradientCheckpointing
            ? '~60% activation memory reduction, +20-25% training time'
            : 'Recompute activations to save memory (recommended for large models)'}
        </Description>
      </Field>

      {/* Flash Attention Toggle */}
      <Field>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <Label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Flash Attention
            </Label>
            <InfoTip text="Reduces attention memory from O(n^2) to O(n). Benefit scales with sequence length: ~15% at 2K, ~50% at 8K, ~70% at 32K+." />
          </div>
          <Switch
            checked={flashAttention}
            onChange={setFlashAttention}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
              flashAttention ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
                flashAttention ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </Switch>
        </div>
        <Description className="text-xs text-gray-600 dark:text-gray-400 mt-1">
          {flashAttention
            ? `${getFlashAttentionBenefit()} (benefit scales with sequence length)`
            : 'O(n²) to O(n) attention memory, also 2-4x faster (benefit scales with sequence length)'}
        </Description>
      </Field>
    </div>
  )
}
