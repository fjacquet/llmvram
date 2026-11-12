import type { GPU, Model } from '@utils/schemas'
import type Decimal from 'decimal.js'
import { fitsInference, maxConcurrentSessions } from './concurrency'
import { isRackGPU, validTPDegrees } from './config-rules'
import { GPU_MEMORY_UTILIZATION } from './constants'
import { defaultMix, largestContext } from './context-mix'
import { calculateInferenceVRAM } from './inference'
import { clampKVTier, kvTierSummary } from './kv-tier'
import { calculateMultiNodeVRAM } from './multi-node'
import { estimatePerformance } from './performance'
import { burstTTFT, classPrefills, maxUsersWithinTarget } from './ttft'
import type { ReplicaCandidate, SizingInput, SizingResult, SizingUnit } from './types'

/** Every replica shape vLLM accepts: TP inside one server/rack, plus full servers as pipeline stages */
export function replicaCandidates(model: Model, gpu: GPU): Array<{ tp: number; nodes: number }> {
  const max = gpu.max_gpus_per_node
  const tps = validTPDegrees(model, max)
  const single = tps.map((tp) => ({ tp, nodes: 1 }))
  if (isRackGPU(gpu) || max === 1 || !tps.includes(max)) return single
  const stages = Array.from({ length: 7 }, (_, i) => ({ tp: max, nodes: i + 2 }))
  return [...single, ...stages]
}

/**
 * Per-GPU memory of one replica at `sessions` simultaneous sessions (the worker's composition).
 * `batch` sizes activations (vLLM runs the whole batch at once); KV follows `sessions`.
 */
export function perGPUBreakdown(
  input: SizingInput,
  tp: number,
  nodes: number,
  sessions: number,
  batch = sessions,
) {
  const { model, gpu, quantization, kvQuantization, sequenceLength, fabric, contextMix } = input
  const single = calculateInferenceVRAM({
    model,
    quantization,
    sequenceLength,
    batchSize: batch,
    kvQuantization,
    concurrentUsers: sessions,
    contextMix,
  })
  const multi =
    tp * nodes > 1
      ? calculateMultiNodeVRAM({
          singleGPU: single,
          model,
          gpuVramGB: gpu.vram_gb,
          gpusPerNode: tp,
          numNodes: nodes,
          intraNodeStrategy: 'tensor-parallel',
          gpu,
          fabric,
          batchSize: batch,
          sequenceLength,
          quantization,
        })
      : null
  return { perGPU: multi ? multi.perGPU : single, multi, single }
}

export function evaluateReplica(
  input: SizingInput,
  tp: number,
  nodes: number,
): ReplicaCandidate & { fits: boolean; lackingGB: number } {
  const { model, gpu, quantization, kvQuantization, sequenceLength, users, contextMix } = input
  // KV of one session, activations at the real batch (capped), so the count never assumes
  // room the activations take
  const one = perGPUBreakdown(input, tp, nodes, 1, Math.max(1, users))
  const totalPerGPUGB = one.perGPU.total.toNumber()
  const memorySessions = maxConcurrentSessions({
    totalPerGPUGB,
    kvPerGPUGB: one.perGPU.kvCache.toNumber(),
    concurrentUsers: 1,
    gpuVramGB: gpu.vram_gb,
  })
  // null: no KV on the GPU, memory does not bound sessions
  const hot = Math.max(1, Math.min(users, memorySessions ?? users))
  // One session at the largest class must fit as well, whatever the mix-weighted KV says
  const longest = largestContext(contextMix)
  const guard = perGPUBreakdown(
    { ...input, sequenceLength: longest, contextMix: defaultMix(longest) },
    tp,
    nodes,
    1,
    Math.max(1, users),
  )
  const fits =
    fitsInference(totalPerGPUGB, gpu.vram_gb) &&
    fitsInference(guard.perGPU.total.toNumber(), gpu.vram_gb) &&
    (memorySessions ?? 1) >= 1
  const classes = classPrefills(
    { model, gpu, quantization, kvQuantization, multiGPUResult: one.multi },
    contextMix,
  )

  const clamped = clampKVTier(input.kvTier)
  // capacityTB is per unit (server or rack): a replica gets its share; a multi-server replica
  // is credited one server (conservative)
  const replicasPerUnit = nodes === 1 ? Math.floor(gpu.max_gpus_per_node / tp) : 1
  const settings = {
    ...clamped,
    capacityTB: clamped.capacityTB === null ? null : clamped.capacityTB / replicasPerUnit,
  }
  // ceiling prefill at one session, as in Inference
  const recomputeSeconds =
    memorySessions === null
      ? null
      : (estimatePerformance({
          model,
          gpu,
          quantization,
          sequenceLength,
          batchSize: 1,
          kvQuantization,
          multiGPUResult: one.multi,
        }).prefillSeconds?.toNumber() ?? null)
  const tier =
    memorySessions === null
      ? null
      : kvTierSummary({
          settings,
          maxHotSessions: memorySessions,
          perGPUKVGB: one.perGPU.kvCache.toNumber(),
          totalKVGB: one.single.kvCache.toNumber(),
          concurrentUsers: 1,
          multi: {
            strategy: 'tensor-parallel',
            gpusPerNode: tp,
            numNodes: nodes,
            numGPUs: tp * nodes,
          },
          recomputeSeconds,
          gpuId: gpu.id,
        })
  // An overloaded tier cannot sustain the churn: keep the HBM-only count
  const held = tier && !tier.overloaded ? Math.min(users, tier.sessionsHeld) : hot
  // Nothing parked (open <= hot): every open session decodes, no active-share discount
  let used = held > hot ? tier : null
  let open = used ? held : hot
  let sessions = used ? Math.min(hot, Math.ceil(open * settings.activeShare)) : hot
  const memoryOpen = open

  // server branch unreachable from the UI (ADR 0010 amendment): no cap, the target is not applied
  const target = isRackGPU(gpu) ? input.ttftTargetSeconds : null
  const burstCap = target === null ? null : maxUsersWithinTarget(classes, target)
  if (burstCap !== null) {
    // The cap bounds decoding sessions: ceil(open x activeShare) <= cap  <=>  open <= floor(cap / activeShare)
    const cap = Math.max(1, burstCap)
    if (used) {
      open = Math.min(open, Math.floor(cap / settings.activeShare + 1e-9))
      sessions = Math.min(hot, Math.ceil(open * settings.activeShare))
    }
    if (!used || open <= hot) {
      used = null
      open = Math.min(open, cap)
      sessions = open
    }
  }

  const real = perGPUBreakdown(input, tp, nodes, sessions)
  const perf = estimatePerformance({
    model,
    gpu,
    quantization,
    sequenceLength,
    batchSize: sessions,
    kvQuantization,
    multiGPUResult: real.multi,
    contextMix,
  })
  const perUser = (aggregate: Decimal) => aggregate.div(sessions)
  return {
    tp,
    nodes,
    gpusPerReplica: tp * nodes,
    sessionsPerReplica: open,
    hotSessions: hot,
    decodeBatch: sessions,
    activeShare: settings.activeShare,
    tierUsed: used ? settings.tier : null,
    tierOverloaded: tier?.overloaded ?? false,
    resumeSeconds: used?.resumeSeconds ?? null,
    recomputeSeconds: used ? recomputeSeconds : null,
    perUserTokensPerSecond: perUser(perf.expected.tokensPerSecond),
    perUserCeilingTokensPerSecond: perUser(perf.tokensPerSecond),
    timeToFirstToken: perf.expected.timeToFirstToken,
    ceilingTimeToFirstToken: perf.timeToFirstToken,
    totalPerGPUGB,
    burstCap,
    ttft: burstTTFT(classes, sessions),
    classes,
    boundBy: open < memoryOpen ? 'ttft' : 'memory',
    fits,
    lackingGB: Math.max(0, totalPerGPUGB - gpu.vram_gb * GPU_MEMORY_UTILIZATION),
  }
}

/** Replicas, per-unit packing and unit count for a candidate that fits */
function pack(input: SizingInput, c: ReplicaCandidate) {
  const { gpu, users } = input
  const replicas = Math.ceil(users / c.sessionsPerReplica)
  const unit: SizingUnit = isRackGPU(gpu) ? 'rack' : 'server'
  const multiServer = unit === 'server' && c.nodes > 1
  const replicasPerUnit = multiServer ? 0 : Math.floor(gpu.max_gpus_per_node / c.tp)
  return {
    unit,
    replicas,
    replicasPerUnit,
    units: multiServer ? replicas * c.nodes : Math.ceil(replicas / replicasPerUnit),
    idleGPUsPerUnit: multiServer ? 0 : gpu.max_gpus_per_node - replicasPerUnit * c.tp,
  }
}

/** The fewest servers or racks that serve `users` at once, expected (not ceiling) speed */
export function sizeDeployment(input: SizingInput): SizingResult {
  const unit: SizingUnit = isRackGPU(input.gpu) ? 'rack' : 'server'
  const evaluated = replicaCandidates(input.model, input.gpu).map(({ tp, nodes }) =>
    evaluateReplica(input, tp, nodes),
  )
  const options = evaluated
    .filter((c) => c.fits && c.burstCap !== 0)
    .map((c) => ({ c, p: pack(input, c) }))
    .sort(
      (a, b) =>
        a.p.units - b.p.units ||
        a.p.replicas * a.c.gpusPerReplica - b.p.replicas * b.c.gpusPerReplica ||
        a.c.gpusPerReplica - b.c.gpusPerReplica,
    )
  const best = options[0]
  if (best) return { fits: true, chosen: best.c, closest: null, neverMeetsTarget: null, ...best.p }
  if (evaluated.some((c) => c.fits)) {
    // Memory holds, but one prompt at the largest class misses the target on every shape
    return {
      fits: true,
      unit,
      units: 0,
      replicas: 0,
      replicasPerUnit: 0,
      idleGPUsPerUnit: 0,
      chosen: null,
      closest: null,
      neverMeetsTarget: { largestClassTokens: largestContext(input.contextMix) },
    }
  }
  const closest = evaluated.reduce((a, b) => (b.lackingGB < a.lackingGB ? b : a))
  return {
    fits: false,
    unit,
    units: 0,
    replicas: 0,
    replicasPerUnit: 0,
    idleGPUsPerUnit: 0,
    chosen: null,
    closest: { candidate: closest, lackingGB: closest.lackingGB },
    neverMeetsTarget: null,
  }
}
