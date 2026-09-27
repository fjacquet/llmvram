import { findGPUById, findModelById, isAtDefaults, useUIStore } from '@store/uiStore'
import { deserializeFromURL, serializeToURL, urlStateToConfig } from '@store/urlSerializer'
import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * Hook that provides bidirectional sync between Zustand store and URL hash
 *
 * On mount:
 * - Reads URL hash and deserializes it
 * - Builds the whole configuration (urlStateToConfig) and applies it in ONE store
 *   action, normalized once; corrections surface as "Shared link adjusted"
 * - Warns when a referenced model/GPU is not in the database
 *
 * On store changes:
 * - Debounces changes by 300ms
 * - Serializes current state to compressed URL hash
 * - Updates URL without triggering navigation
 * - Warns if URL exceeds recommended length
 */
export function useURLSync() {
  // Hydrate store from URL hash on mount
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

    const { patch, missing } = urlStateToConfig(urlState, {
      findModel: findModelById,
      findGPU: findGPUById,
    })
    for (const message of missing) toast.warning(message)
    useUIStore.getState().restoreConfig(patch)
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

        // resetAll() (spec Section 4) clears the hash itself, but this debounced
        // effect still fires from that same state change: without this, it would
        // write `#<encoded defaults>` ~300ms later, and the hash would visibly
        // reappear even though it decodes to the same configuration. Keep it cleared
        // for as long as the state stays at the defaults.
        if (isAtDefaults(state)) {
          if (window.location.hash) {
            window.history.replaceState(null, '', window.location.pathname + window.location.search)
          }
          return
        }

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
