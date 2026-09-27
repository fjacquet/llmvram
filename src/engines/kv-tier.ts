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

/** Per-GPU read bandwidth presets, GB/s. All estimates; basis in each comment. */
export const KV_TIER_PRESETS: Record<
  Exclude<KVTierType, 'none'>,
  { label: string; gbpsPerGPU: number }
> = {
  // NVLink-C2C 900 GB/s per Grace superchip, shared by 2 GPUs, one direction.
  // NVIDIA publishes no per-GPU figure; estimate.
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

export function tierBandwidthGBps(settings: KVTierSettings): number | null {
  if (settings.tier === 'none') return null
  return settings.customGBps ?? KV_TIER_PRESETS[settings.tier].gbpsPerGPU
}

export function resumeSeconds(kvPerSessionPerGPUGB: number, gbpsPerGPU: number): number {
  return KV_TIER_RESUME_OVERHEAD_S + kvPerSessionPerGPUGB / gbpsPerGPU
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
}): KVTierSummary | null {
  const bandwidth = tierBandwidthGBps(p.settings)
  if (bandwidth === null) return null

  const { activeShare, burstSeconds, capacityTB } = p.settings
  const { kvPerSessionPerGPUGB, gpusPerSession } = sessionKVLayout(p)
  const kvPerSessionGB = p.totalKVGB / Math.max(1, p.concurrentUsers)

  // Active sessions occupy the HBM slots; the tier holds only the parked ones, so its
  // capacity bounds parked sessions, never the ones that already fit in HBM.
  const parkedCapacity =
    capacityTB && kvPerSessionGB > 0
      ? Math.floor((capacityTB * 1000) / kvPerSessionGB)
      : Number.POSITIVE_INFINITY
  const sessionsHeld = Math.min(
    Math.floor(p.maxHotSessions / activeShare),
    p.maxHotSessions + parkedCapacity,
  )

  const resume = resumeSeconds(kvPerSessionPerGPUGB, bandwidth)
  // Every GPU of a session fetches its own part; duplicated MLA KV is fetched once
  // per tensor-parallel rank (conservative: assumes no cross-rank de-duplication).
  const trafficGBps =
    ((sessionsHeld * activeShare) / burstSeconds) * kvPerSessionPerGPUGB * gpusPerSession
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
