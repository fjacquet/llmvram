import { Layout } from '@components/layout/Layout'
import { useDarkMode } from '@hooks/useDarkMode'
import { useURLSync } from '@hooks/useURLSync'
import { Toaster } from 'sonner'

/**
 * Root application component
 *
 * Initializes dark mode DOM sync, URL sync for state sharing, and renders the main layout with toast notifications
 */
export default function App() {
  // Initialize dark mode DOM sync
  useDarkMode()

  // Sync calculator state with URL hash for sharing
  useURLSync()

  return (
    <>
      <Layout />
      <Toaster
        richColors
        position="top-right"
        toastOptions={{
          className: 'dark:bg-gray-800 dark:text-white',
        }}
      />
    </>
  )
}
