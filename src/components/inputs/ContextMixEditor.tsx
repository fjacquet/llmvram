import { InfoTip } from '@components/common/InfoTip'
import { MIN_CONTEXT } from '@engines/config-rules'
import { CONTEXT_PRESETS, isDefaultMix, normalizeMix } from '@engines/context-mix'
import { useUIStore } from '@store/uiStore'
import { MAX_SEQUENCE_LENGTH } from '@utils/schemas'

const MAX_CLASSES = 3
const FIELD =
  'px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white'

const pct = (share: number) => Math.round(share * 100)

/** Context classes as rows; the last share is the remainder so shares always sum to 100% */
export function ContextMixEditor() {
  const contextMix = useUIStore((s) => s.contextMix)
  const sequenceLength = useUIStore((s) => s.sequenceLength)
  const setContextMix = useUIStore((s) => s.setContextMix)
  const lastIndex = contextMix.length - 1

  const setTokens = (i: number, raw: number) => {
    if (!Number.isFinite(raw)) return
    const tokens = Math.min(MAX_SEQUENCE_LENGTH, Math.max(MIN_CONTEXT, Math.round(raw)))
    setContextMix(contextMix.map((c, j) => (j === i ? { ...c, tokens } : c)))
  }

  const setShare = (i: number, raw: number) => {
    if (!Number.isFinite(raw)) return
    // The last class keeps at least 1%: leave room for it
    const others = contextMix
      .slice(0, lastIndex)
      .reduce((s, c, j) => s + (j === i ? 0 : pct(c.share)), 0)
    const p = Math.min(99 - others, Math.max(1, Math.round(raw)))
    setContextMix(normalizeMix(contextMix.map((c, j) => (j === i ? { ...c, share: p / 100 } : c))))
  }

  const addClass = () => {
    const prev = contextMix[lastIndex]
    if (!prev) return
    // The old last class gives half its share to the new last one
    const keep = Math.floor(pct(prev.share) / 2)
    const used = new Set(contextMix.map((c) => c.tokens))
    const tokens = CONTEXT_PRESETS.find((p) => !used.has(p.tokens))?.tokens ?? MIN_CONTEXT
    setContextMix([
      ...contextMix.slice(0, lastIndex),
      { tokens: prev.tokens, share: keep / 100 },
      { tokens, share: (pct(prev.share) - keep) / 100 },
    ])
  }

  const removeClass = (i: number) =>
    setContextMix(normalizeMix(contextMix.filter((_, j) => j !== i)))

  const canAdd = contextMix.length < MAX_CLASSES && pct(contextMix[lastIndex]?.share ?? 0) >= 2

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1">
        <span className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Context mix
        </span>
        <InfoTip text="Up to 3 prompt-size classes with the share of requests each carries. Memory is sized on the mix average; one session at the largest class must still fit. The TTFT of a burst is the share-weighted mean prefill times the number of prompts." />
      </div>

      {contextMix.map((c, i) => (
        // Row identity is its position: tokens and shares are edited in place
        // biome-ignore lint/suspicious/noArrayIndexKey: rows have no stable id
        <div key={i} className="space-y-1.5">
          <div className="flex flex-wrap gap-1.5">
            {CONTEXT_PRESETS.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setTokens(i, p.tokens)}
                className={`px-2 py-0.5 text-xs font-medium rounded-md transition-colors ${
                  c.tokens === p.tokens
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              key={c.tokens}
              type="number"
              min={MIN_CONTEXT}
              max={MAX_SEQUENCE_LENGTH}
              defaultValue={c.tokens}
              onBlur={(e) => setTokens(i, Number(e.target.value))}
              className={`w-28 ${FIELD}`}
              aria-label={`Class ${i + 1} tokens`}
            />
            <span className="text-xs text-gray-500 dark:text-gray-400">tokens</span>
            <input
              type="number"
              min={1}
              max={99}
              {...(i === lastIndex
                ? { value: pct(c.share), readOnly: true }
                : {
                    key: `share-${pct(c.share)}`,
                    defaultValue: pct(c.share),
                    onBlur: (e: React.FocusEvent<HTMLInputElement>) =>
                      setShare(i, Number(e.target.value)),
                  })}
              className={`w-16 ${FIELD} read-only:opacity-70`}
              aria-label={`Class ${i + 1} share`}
            />
            <span className="text-xs text-gray-500 dark:text-gray-400">%</span>
            {contextMix.length > 1 && (
              <button
                type="button"
                onClick={() => removeClass(i)}
                className="text-xs text-red-600 dark:text-red-400 hover:underline"
                aria-label={`Remove class ${i + 1}`}
              >
                Remove
              </button>
            )}
          </div>
        </div>
      ))}

      <button
        type="button"
        onClick={addClass}
        disabled={!canAdd}
        className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline disabled:opacity-50 disabled:no-underline"
      >
        Add class
      </button>

      {!isDefaultMix(contextMix, sequenceLength) && (
        <p className="text-xs text-gray-500 dark:text-gray-400">memory sized on the mix average</p>
      )}
    </div>
  )
}
