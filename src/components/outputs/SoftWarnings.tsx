import type { SoftWarning } from '@engines/config-rules'

/** Soft rules (W3, W6, W8): shown inline, never corrected (spec Section 1) */
export function SoftWarnings({ warnings }: { warnings: SoftWarning[] }) {
  if (warnings.length === 0) return null
  return (
    <>
      {warnings.map((w) => (
        <div
          key={w.id}
          data-testid={`soft-warning-${w.id}`}
          className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4"
        >
          <p className="text-sm text-amber-800 dark:text-amber-200">⚠ {w.message}</p>
        </div>
      ))}
    </>
  )
}
