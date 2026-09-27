import { BYTES_PER_GB } from './constants'
import type { ShardingStrategy } from './types'

/**
 * KV storage tier: park idle sessions' KV cache off the GPU and reload it on resume.
 *
 * Parked sessions hold no HBM, so memory and decode engines are unchanged; this
 * module only answers how many sessions a configuration holds, how long a resume
 * takes versus recomputing the prefill, and whether the tier sustains the churn.
 * Spec: docs/superpowers/specs/2026-09-26-kv-storage-tier-design.md
 */

export const KV_TIER_TYPES = ['none', 'host-grace', 'host-pcie', 'local-nvme', 'network'] as const

export type KVTierType = (typeof KV_TIER_TYPES)[number]

export interface KVTierSettings {
  tier: KVTierType
  /** Overrides the preset when positive (GB/s per GPU, read) */
  customGBps: number | null
  /** Fraction of sessions decoding at any moment (0-1) */
  activeShare: number
  /** Mean length of one active burst before the session parks again */
  burstSeconds: number
  /** Tier capacity in TB; null = unlimited */
  capacityTB: number | null
}

export const DEFAULT_KV_TIER: KVTierSettings = {
  tier: 'none',
  customGBps: null,
  activeShare: 0.25,
  burstSeconds: 30,
  capacityTB: null,
}

/**
 * Per-GPU read bandwidth presets, GB/s. All estimates; basis in each comment.
 *
 * host-grace's gbpsPerGPU (225, the NVL72 figure) is a fallback only, used
 * when no GPU id is available to resolve the real per-GPU figure — see
 * graceLinkGBps, which callers should prefer whenever a GPU id is in hand.
 */
export const KV_TIER_PRESETS: Record<
  Exclude<KVTierType, 'none'>,
  { label: string; gbpsPerGPU: number }
> = {
  'host-grace': { label: 'Host memory (Grace NVLink-C2C)', gbpsPerGPU: 225 },
  // PCIe 5 x16 ~64 GB/s theoretical, ~50 practical.
  'host-pcie': { label: 'Host memory (PCIe 5)', gbpsPerGPU: 50 },
  // GB300 tray: 4 E1.S Gen5 drives for 4 GPUs (NVIDIA NVL72 reference architecture).
  'local-nvme': { label: 'Local NVMe', gbpsPerGPU: 12 },
  // One 400 GbE storage NIC share per GPU; Dell ObjectScale anchor: >= 51 GB/s per
  // 4-GPU server = 12.8 per GPU.
  // No Dell Lightning FS preset: it targets > 16K GPUs; size cluster storage in raidy.
  network: { label: 'Network storage (CMX, PowerScale, ObjectScale)', gbpsPerGPU: 12.5 },
}

/**
 * Fixed cost of a resume beyond the transfer: Dell measured offload at 113-129 ms
 * against 91 ms recompute at 4K tokens, where the transfer itself is ~15 ms.
 */
const KV_TIER_RESUME_OVERHEAD_S = 0.03

/**
 * Engine KV sizes are GiB (BYTES_PER_GB = 1024^3, like vram_gb); tier capacity (TB)
 * and bandwidth (GB/s) are decimal, as storage and network vendors quote them.
 *
 * Exported so every GiB -> decimal-GB conversion in the codebase (offloaded
 * memory vs. host capacity, tier capacity, etc.) shares this one constant.
 */
export const DECIMAL_GB_PER_GIB = BYTES_PER_GB.toNumber() / 1e9

/**
 * Bounds for tier settings, applied once at the store boundary (uiStore.setKVTier),
 * so the engine below can trust its input: active share 1-100%, burst >= 1 s, and a
 * custom bandwidth or capacity that is positive and finite, else null.
 */
export function clampKVTier(s: KVTierSettings): KVTierSettings {
  const positive = (n: number | null) => (n !== null && Number.isFinite(n) && n > 0 ? n : null)
  return {
    ...s,
    customGBps: positive(s.customGBps),
    activeShare: Number.isFinite(s.activeShare)
      ? Math.min(1, Math.max(0.01, s.activeShare))
      : DEFAULT_KV_TIER.activeShare,
    burstSeconds: Number.isFinite(s.burstSeconds)
      ? Math.max(1, s.burstSeconds)
      : DEFAULT_KV_TIER.burstSeconds,
    capacityTB: positive(s.capacityTB),
  }
}

/**
 * Per-GPU read bandwidth (GB/s) between a GPU and its Grace host's memory
 * over NVLink-C2C, for the `host-grace` KV tier preset; null when the GPU has
 * no Grace host at all.
 *
 * NVIDIA quotes 900 GB/s bidirectional per Grace-GPU superchip link, i.e. 450
 * GB/s in one direction. GB300 NVL72 pairs 2 GPUs to each Grace CPU, sharing
 * that one-direction figure: 225 GB/s each (NVIDIA publishes no per-GPU
 * figure; estimate). The GB300 Desktop Superchip pairs 1 GPU to 1 Grace CPU,
 * so the link itself isn't the bottleneck — the Grace LPDDR5X behind it is:
 * "DGX Station GB300: 900 GB/s NVLink-C2C, 496 GB LPDDR5X at 396 GB/s
 * (NVIDIA / Tom's Hardware); memory-bound."
 *
 * HGX B300 (`nvidia-gb300-288gb`) is an x86 host, not Grace, despite the
 * GB300 name. GB10 (`nvidia-gb10`) has no separate host tier: its Grace
 * memory is already one unified pool shared with the GPU, not a second tier
 * to park KV cache into.
 */
export function graceLinkGBps(gpuId: string): number | null {
  if (gpuId === 'nvidia-gb300-nvl72') return 225
  if (gpuId === 'nvidia-gb300-desktop-252gb') return 396
  return null
}

/**
 * Falls the KV tier back to `none` when it no longer matches a Grace-host GPU.
 *
 * Called from both uiStore.setSelectedGPU (the GPU changes while `host-grace`
 * is already active) AND uiStore.setKVTier (a patch sets `host-grace` while a
 * non-Grace GPU is already current). Both guards are required: useURLSync
 * restores the GPU first and the tier second from the same hash, so a shared
 * link with a `host-grace` tier for a non-Grace GPU only surfaces as the
 * latter — the setSelectedGPU guard alone would miss it, since at the moment
 * the GPU is set the tier is still whatever it was before the restore.
 */
export function resetTierForGPU(tier: KVTierSettings, gpuId: string | null): KVTierSettings {
  if (tier.tier !== 'host-grace' || graceLinkGBps(gpuId ?? '') !== null) return tier
  return { ...tier, tier: 'none' }
}

/**
 * @param gpuId Resolves the exact host-grace figure (graceLinkGBps) when the
 *   tier is `host-grace` and no custom bandwidth is set; falls back to the
 *   generic KV_TIER_PRESETS figure if the id doesn't match a Grace GPU
 *   (shouldn't happen in practice — resetTierForGPU keeps the two in sync).
 */
export function tierBandwidthGBps(
  settings: KVTierSettings,
  gpuId: string | null = null,
): number | null {
  if (settings.tier === 'none') return null
  if (settings.tier === 'host-grace') {
    return (
      settings.customGBps ?? graceLinkGBps(gpuId ?? '') ?? KV_TIER_PRESETS['host-grace'].gbpsPerGPU
    )
  }
  return settings.customGBps ?? KV_TIER_PRESETS[settings.tier].gbpsPerGPU
}

/** @param kvPerSessionPerGPUGB GiB (engine unit); @param gbpsPerGPU decimal GB/s */
export function resumeSeconds(kvPerSessionPerGPUGB: number, gbpsPerGPU: number): number {
  return KV_TIER_RESUME_OVERHEAD_S + (kvPerSessionPerGPUGB * DECIMAL_GB_PER_GIB) / gbpsPerGPU
}

interface KVTierSummary {
  sessionsHeld: number
  resumeSeconds: number
  /** null when the prefill time is unknown (GPU without FLOPS data) */
  resumeFaster: boolean | null
  /** Tier read traffic needed to resume sessions at the configured churn */
  trafficGBps: number
  /** What the tier delivers across all GPUs */
  tierGBps: number
  /** The churn needs more than the tier delivers */
  overloaded: boolean
}

/** The slice of a multi-GPU breakdown that decides where a session's KV sits */
type SessionMulti = {
  strategy: ShardingStrategy
  gpusPerNode: number
  numNodes: number
  numGPUs: number
} | null

/**
 * Where one session's KV sits, for resume and tier traffic.
 *
 * `perGPUKVGB / concurrentUsers` is the average KV per session on a GPU. Under tensor
 * and pipeline parallelism every GPU holds a share of every session (all of it when
 * MLA KV is duplicated), so a resume reads that share on every GPU at once. Under
 * expert parallelism each session lives on one rank per node (sessions spread 1/N),
 * so one session is gpusPerNode times the average and reloads through that one link.
 */
export function sessionKVLayout(p: {
  perGPUKVGB: number
  concurrentUsers: number
  multi: SessionMulti
}): { kvPerSessionPerGPUGB: number; gpusPerSession: number } {
  const average = p.perGPUKVGB / Math.max(1, p.concurrentUsers)
  if (!p.multi || p.multi.numGPUs <= 1) return { kvPerSessionPerGPUGB: average, gpusPerSession: 1 }
  if (p.multi.strategy === 'expert-parallel') {
    return {
      kvPerSessionPerGPUGB: average * p.multi.gpusPerNode,
      gpusPerSession: p.multi.numNodes,
    }
  }
  return { kvPerSessionPerGPUGB: average, gpusPerSession: p.multi.numGPUs }
}

/**
 * Sessions held, resume vs recompute, and tier traffic, from the per-GPU and total KV
 * of the displayed breakdown (both sized for `concurrentUsers` sessions).
 * Settings are assumed clamped (clampKVTier).
 */
export function kvTierSummary(p: {
  settings: KVTierSettings
  maxHotSessions: number
  /** KV on one GPU of the displayed breakdown, all sessions */
  perGPUKVGB: number
  /** KV of all sessions, one stored copy each (tier capacity) */
  totalKVGB: number
  concurrentUsers: number
  multi: SessionMulti
  recomputeSeconds: number | null
  /** Selected GPU id, to resolve host-grace bandwidth (graceLinkGBps) */
  gpuId?: string | null
}): KVTierSummary | null {
  const bandwidth = tierBandwidthGBps(p.settings, p.gpuId ?? null)
  if (bandwidth === null) return null

  const { activeShare, burstSeconds, capacityTB } = p.settings
  const { kvPerSessionPerGPUGB, gpusPerSession } = sessionKVLayout(p)
  const kvPerSessionGB = p.totalKVGB / Math.max(1, p.concurrentUsers)

  // Active sessions occupy the HBM slots; the tier holds only the parked ones, so its
  // capacity bounds parked sessions, never the ones that already fit in HBM.
  const parkedCapacity =
    capacityTB && kvPerSessionGB > 0
      ? Math.floor((capacityTB * 1000) / (kvPerSessionGB * DECIMAL_GB_PER_GIB))
      : Number.POSITIVE_INFINITY
  const sessionsHeld = Math.min(
    Math.floor(p.maxHotSessions / activeShare),
    p.maxHotSessions + parkedCapacity,
  )

  const resume = resumeSeconds(kvPerSessionPerGPUGB, bandwidth)
  // Every GPU of a session fetches its own part; duplicated MLA KV is fetched once
  // per tensor-parallel rank (conservative: assumes no cross-rank de-duplication).
  const trafficGBps =
    ((sessionsHeld * activeShare) / burstSeconds) *
    kvPerSessionPerGPUGB *
    gpusPerSession *
    DECIMAL_GB_PER_GIB
  const tierGBps = bandwidth * (p.multi?.numGPUs ?? 1)
  return {
    sessionsHeld,
    resumeSeconds: resume,
    resumeFaster: p.recomputeSeconds === null ? null : resume < p.recomputeSeconds,
    trafficGBps,
    tierGBps,
    overloaded: trafficGBps > tierGBps,
  }
}
