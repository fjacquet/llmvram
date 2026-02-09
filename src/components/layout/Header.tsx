import { DarkModeToggle } from '@components/common/DarkModeToggle'

/**
 * App header with branding and dark mode toggle
 */
export function Header() {
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
          <DarkModeToggle />
        </div>
      </div>
    </header>
  )
}
