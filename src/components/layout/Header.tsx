import { DarkModeToggle } from '@components/common/DarkModeToggle'
import { LinkIcon } from '@heroicons/react/24/outline'
import { toast } from 'sonner'

/**
 * App header with branding, share button, and dark mode toggle
 */
export function Header() {
  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href)
      toast.success('Link copied to clipboard!')
    } catch {
      // Fallback for browsers without clipboard API or when permissions denied
      toast.error('Could not copy link. Copy the URL from your browser address bar.')
    }
  }

  return (
    <header className="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              LLM VRAM Calculator
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Estimate memory requirements for running large language models
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleShare}
              className="p-2 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Copy shareable link"
            >
              <LinkIcon className="w-5 h-5" />
            </button>
            <DarkModeToggle />
          </div>
        </div>
      </div>
    </header>
  )
}
