import { MAX_GPUS_PER_NODE } from '@engines/constants'
import type { GPU } from '@utils/schemas'

/**
 * The per-node GPU bound for a selection, or the engine's sanity bound with none.
 *
 * Single definition on purpose: config-rules R1 and allowedOptions read it, so the
 * UI never offers a count the store corrects.
 */
export function maxGPUsFor(gpu: GPU | null): number {
  return gpu?.max_gpus_per_node ?? MAX_GPUS_PER_NODE
}

/**
 * Clamp a GPU count to what the selected GPU can form (GPU.max_gpus_per_node).
 *
 * The store no longer calls this: config-rules R1 corrects numGPUs and always shows
 * the correction. Recommendations uses it to bound a suggested count.
 */
export function clampGPUCount(numGPUs: number, gpu: GPU | null): number {
  return Math.min(Math.max(1, Math.trunc(numGPUs)), maxGPUsFor(gpu))
}
