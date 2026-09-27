import { type AllowedOptions, allowedOptions } from '@engines/config-rules'
import { useUIStore } from '@store/uiStore'
import { useMemo } from 'react'

/** What each input may offer for the current selection (ADR 0004: never offer an impossible value) */
export function useAllowedOptions(): AllowedOptions {
  const mode = useUIStore((s) => s.mode)
  const shardingStrategy = useUIStore((s) => s.shardingStrategy)
  const offloadingEnabled = useUIStore((s) => s.offloadingEnabled)
  const kvCacheOffload = useUIStore((s) => s.kvCacheOffload)
  const frameworkPreset = useUIStore((s) => s.frameworkPreset)
  const model = useUIStore((s) => s.selectedModel)
  const gpu = useUIStore((s) => s.selectedGPU)
  return useMemo(
    () =>
      allowedOptions(
        { mode, shardingStrategy, offloadingEnabled, kvCacheOffload, frameworkPreset },
        model,
        gpu,
      ),
    [mode, shardingStrategy, offloadingEnabled, kvCacheOffload, frameworkPreset, model, gpu],
  )
}
