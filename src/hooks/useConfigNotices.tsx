import { useUIStore } from '@store/uiStore'
import { useEffect } from 'react'
import { toast } from 'sonner'

/**
 * Show the store's correction notice as one toast per action ("Adjusted for ..." or
 * "Shared link adjusted"), one line per correction, then clear it. Mounted once in App.
 */
export function useConfigNotices(): void {
  const pendingNotice = useUIStore((s) => s.pendingNotice)
  const clearNotice = useUIStore((s) => s.clearNotice)

  useEffect(() => {
    if (!pendingNotice) return
    toast.warning(pendingNotice.title, {
      description: (
        <ul className="list-disc pl-4">
          {pendingNotice.lines.map((line, i) => (
            // biome-ignore lint/suspicious/noArrayIndexKey: lines has no stable id and can repeat the same text (e.g. two "KV tier turned off" corrections); index+text keeps keys unique where text alone would collide.
            <li key={`${i}-${line}`}>{line}</li>
          ))}
        </ul>
      ),
    })
    clearNotice()
  }, [pendingNotice, clearNotice])
}
