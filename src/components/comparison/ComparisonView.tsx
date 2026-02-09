import { useComparisonStore } from '@store/comparisonStore'
import { ComparisonColumn } from './ComparisonColumn'

/**
 * Main comparison view component
 *
 * Displays saved configuration snapshots in a responsive grid
 * with diff highlighting to show differences between configs.
 *
 * Empty state instructs users to save configs from results panel.
 */
export function ComparisonView() {
  const { snapshots, clearAll } = useComparisonStore()

  // Empty state
  if (snapshots.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
        <div className="text-gray-500 dark:text-gray-400">
          <svg
            className="mx-auto h-12 w-12 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            role="img"
            aria-label="Compare icon"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          <p className="text-lg font-medium mb-2">No configurations saved for comparison</p>
          <p className="text-sm">
            Use the "Save to Compare" button in the results panel to add configurations here.
          </p>
        </div>
      </div>
    )
  }

  // Grid with snapshots
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Compare Configurations</h2>
        <button
          type="button"
          onClick={() => {
            if (window.confirm('Clear all saved configurations?')) {
              clearAll()
            }
          }}
          className="text-sm text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300 font-medium transition-colors"
        >
          Clear All
        </button>
      </div>

      {/* Grid of comparison columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {snapshots.map((snapshot) => (
          <ComparisonColumn
            key={snapshot.id}
            snapshot={snapshot}
            allSnapshots={snapshots}
            onRemove={() => useComparisonStore.getState().removeSnapshot(snapshot.id)}
          />
        ))}
      </div>
    </div>
  )
}
