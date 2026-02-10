import { ComparisonView } from '@components/comparison/ComparisonView'
import { GuidePage } from '@components/guide/GuidePage'
import { Header } from '@components/layout/Header'
import { InputPanel } from '@components/layout/InputPanel'
import { ResultsPanel } from '@components/layout/ResultsPanel'
import { BookOpenIcon } from '@heroicons/react/24/outline'
import { useComparisonStore } from '@store/comparisonStore'
import { useState } from 'react'

/**
 * Responsive layout component
 *
 * Mobile: Stacked (inputs on top, results below)
 * Desktop (lg+): Two columns side-by-side (input panel 5/12, results panel 7/12)
 * Input panel is sticky on desktop for better UX while scrolling results
 *
 * Tab navigation allows switching between Calculator and Comparison views.
 */
export function Layout() {
  const [activeTab, setActiveTab] = useState<'calculator' | 'comparison' | 'guide'>('calculator')
  const snapshotCount = useComparisonStore((s) => s.snapshots.length)

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <Header />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Tab navigation */}
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-6">
          <button
            type="button"
            onClick={() => setActiveTab('calculator')}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === 'calculator'
                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            Calculator
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('comparison')}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === 'comparison'
                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            Comparison
            {snapshotCount > 0 && (
              <span className="ml-2 inline-flex items-center rounded-full bg-blue-100 dark:bg-blue-900 px-2 py-0.5 text-xs font-medium text-blue-700 dark:text-blue-300">
                {snapshotCount}
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('guide')}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors flex items-center gap-1.5 ${
              activeTab === 'guide'
                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'
            }`}
          >
            <BookOpenIcon className="h-4 w-4" />
            Guide
          </button>
        </div>

        {/* Conditional content based on active tab */}
        {activeTab === 'calculator' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5">
              <div className="lg:sticky lg:top-6">
                <InputPanel />
              </div>
            </div>
            <div className="lg:col-span-7">
              <ResultsPanel />
            </div>
          </div>
        ) : activeTab === 'comparison' ? (
          <ComparisonView />
        ) : (
          <GuidePage />
        )}
      </main>
    </div>
  )
}
