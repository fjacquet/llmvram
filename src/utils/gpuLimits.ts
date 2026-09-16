import { MAX_GPUS_PER_NODE } from '@engines/constants'
import type { GPU } from '@utils/schemas'

/**
 * Clamp a per-node GPU count to what the selected GPU can actually form
 *
 * The bound is GPU.max_gpus_per_node — a hard limit, min(coherent interconnect
 * limit, largest shipping chassis slot count). With no GPU selected there is
 * nothing to bound against, so the engine's flat sanity bound applies.
 *
 * Clamping is silent by design: no toast, no warning. A shared link carrying a
 * count above the bound will render a different number than its sender saw.
 */
export function clampGPUCount(numGPUs: number, gpu: GPU | null): number {
  const max = gpu?.max_gpus_per_node ?? MAX_GPUS_PER_NODE
  return Math.min(Math.max(1, Math.trunc(numGPUs)), max)
}
