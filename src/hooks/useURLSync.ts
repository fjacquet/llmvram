import type {
  KVCachePrecision,
  OffloadMode,
  OffloadTarget,
  QuantizationFormat,
  ShardingStrategy,
} from '@engines/types'
import { findGPUById, findModelById, useUIStore } from '@store/uiStore'
import { deserializeFromURL, serializeToURL } from '@store/urlSerializer'
import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * Hook that provides bidirectional sync between Zustand store and URL hash
 *
 * On mount:
 * - Reads URL hash and deserializes to restore configuration
 * - Hydrates store with model/GPU from database or creates custom objects
 * - Shows toast if referenced model/GPU not found
 *
 * On store changes:
 * - Debounces changes by 300ms
 * - Serializes current state to compressed URL hash
 * - Updates URL without triggering navigation
 * - Warns if URL exceeds recommended length
 */
export function useURLSync() {
  const store = useUIStore()

  // Hydrate store from URL hash on mount
  // biome-ignore lint/correctness/useExhaustiveDependencies: Zustand store setters are stable and don't need to be in deps
  useEffect(() => {
    const hash = window.location.hash.slice(1)
    if (!hash) {
      return
    }

    const urlState = deserializeFromURL(hash)
    if (!urlState) {
      // Invalid/corrupted URL - show toast and continue with defaults
      toast.error('Could not restore configuration from URL')
      return
    }

    // Restore model
    if (urlState.modelId) {
      const model = findModelById(urlState.modelId)
      if (model) {
        store.setSelectedModel(model)
      } else if (urlState.customModel) {
        // Model ID not found but custom params available - restore custom
        store.setSelectedModel({
          id: 'custom-restored',
          architecture: 'dense',
          ...urlState.customModel,
        })
      } else {
        // Model ID not found and no custom params
        toast.warning('Model from shared link not found in database')
      }
    } else if (urlState.customModel) {
      // Custom model without ID
      store.setSelectedModel({
        id: 'custom-restored',
        architecture: 'dense',
        ...urlState.customModel,
      })
    }

    // Restore GPU
    if (urlState.gpuId) {
      const gpu = findGPUById(urlState.gpuId)
      if (gpu) {
        store.setSelectedGPU(gpu)
      } else if (urlState.customGPU) {
        // GPU ID not found but custom params available - restore custom
        store.setSelectedGPU({
          id: 'custom-restored',
          manufacturer: 'nvidia',
          memory_type: 'Custom',
          bus_width: 0,
          tier: 'consumer',
          interconnect: 'none',
          ...urlState.customGPU,
        })
      } else {
        // GPU ID not found and no custom params
        toast.warning('GPU from shared link not found in database')
      }
    } else if (urlState.customGPU) {
      // Custom GPU without ID
      store.setSelectedGPU({
        id: 'custom-restored',
        manufacturer: 'nvidia',
        memory_type: 'Custom',
        bus_width: 0,
        tier: 'consumer',
        interconnect: 'none',
        ...urlState.customGPU,
      })
    }

    // Restore calculation parameters
    store.setQuantization(urlState.q as QuantizationFormat)
    store.setSequenceLength(urlState.sl)
    store.setBatchSize(urlState.bs)
    store.setKVQuantization(urlState.kvq as KVCachePrecision)
    store.setNumGPUs(urlState.ng)
    store.setShardingStrategy(urlState.ss as ShardingStrategy)

    // Restore offloading parameters (only if enabled)
    if (urlState.oe) {
      store.setOffloadingEnabled(true)
      if (urlState.ot) store.setOffloadTarget(urlState.ot as OffloadTarget)
      if (urlState.om) store.setOffloadMode(urlState.om as OffloadMode)
      if (urlState.op !== undefined) store.setOffloadPercentage(urlState.op)
      if (urlState.ol !== undefined) store.setOffloadLayers(urlState.ol)
      if (urlState.ko !== undefined) store.setKVCacheOffload(urlState.ko)
    }
  }, []) // Empty deps - only run on mount

  // Sync store changes to URL hash (debounced)
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined

    const unsubscribe = useUIStore.subscribe((state) => {
      // Clear previous timeout
      if (timeoutId) {
        clearTimeout(timeoutId)
      }

      // Debounce by 300ms
      timeoutId = setTimeout(() => {
        const compressed = serializeToURL(state)

        // Skip if URL hasn't changed (avoid infinite loop)
        const currentHash = window.location.hash.slice(1)
        if (compressed === currentHash) {
          return
        }

        // Warn if URL is getting large
        if (compressed.length > 1800) {
          console.warn(`URL state exceeds recommended limit: ${compressed.length} characters`)
        }

        // Update URL without navigation
        window.history.replaceState(null, '', `#${compressed}`)
      }, 300)
    })

    // Cleanup
    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
      unsubscribe()
    }
  }, [])
}
