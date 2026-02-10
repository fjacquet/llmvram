import { InfoTip } from '@components/common/InfoTip'
import { Description, Field, Label, Switch } from '@headlessui/react'
import { useUIStore } from '@store/uiStore'

/**
 * CPU offload toggle for DeepSpeed optimizer states
 *
 * Only visible when a DeepSpeed preset is selected. Enables offloading
 * optimizer states from GPU VRAM to CPU RAM at the cost of throughput.
 */
export function CPUOffloadToggle() {
  const { frameworkPreset, cpuOffloadOptimizer, setCpuOffloadOptimizer } = useUIStore()

  // Only show for DeepSpeed presets
  if (!frameworkPreset.startsWith('deepspeed-')) {
    return null
  }

  return (
    <Field>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1">
          <Label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            CPU Offload Optimizer
          </Label>
          <InfoTip text="Move optimizer states to CPU RAM during training. Reduces GPU VRAM by the size of optimizer states but slows training by 15-30% due to PCIe transfers." />
        </div>
        <Switch
          checked={cpuOffloadOptimizer}
          onChange={setCpuOffloadOptimizer}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
            cpuOffloadOptimizer ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-200 ease-in-out ${
              cpuOffloadOptimizer ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </Switch>
      </div>
      <Description className="text-xs text-gray-600 dark:text-gray-400 mt-1">
        {cpuOffloadOptimizer
          ? 'Offload optimizer states to CPU RAM (15-30% throughput reduction). Requires sufficient system RAM.'
          : 'Keep all training state on GPU (maximum throughput)'}
      </Description>
    </Field>
  )
}
