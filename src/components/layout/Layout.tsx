import { Header } from '@components/layout/Header'
import { InputPanel } from '@components/layout/InputPanel'
import { ResultsPanel } from '@components/layout/ResultsPanel'

/**
 * Responsive layout component
 *
 * Mobile: Stacked (inputs on top, results below)
 * Desktop (lg+): Two columns side-by-side (input panel 5/12, results panel 7/12)
 * Input panel is sticky on desktop for better UX while scrolling results
 */
export function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <Header />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
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
      </main>
    </div>
  )
}
