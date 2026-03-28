import { useUIStore } from '@store/uiStore'
import { useEffect } from 'react'

/**
 * Hook to manage dark mode state and sync it to the DOM
 *
 * Reads isDarkMode from Zustand store and applies/removes 'dark' class
 * on document.documentElement. This is the ONLY place that manipulates
 * the dark class - components should call toggleDarkMode instead.
 */
export function useDarkMode() {
  const isDarkMode = useUIStore((state) => state.isDarkMode)
  const toggleDarkMode = useUIStore((state) => state.toggleDarkMode)
  const setIsDarkMode = useUIStore((state) => state.setIsDarkMode)

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [isDarkMode])

  // Track live OS theme changes and mirror them in the store
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const handler = (e: MediaQueryListEvent) => setIsDarkMode(e.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [setIsDarkMode])

  return { isDarkMode, toggleDarkMode }
}
