import { Layout } from '@components/layout/Layout'
import { useDarkMode } from '@hooks/useDarkMode'
import { Toaster } from 'sonner'

/**
 * Root application component
 *
 * Initializes dark mode DOM sync and renders the main layout with toast notifications
 */
export default function App() {
  // Initialize dark mode DOM sync
  useDarkMode()

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
