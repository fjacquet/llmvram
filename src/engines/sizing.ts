import type { GPU, Model } from '@utils/schemas'
import type Decimal from 'decimal.js'
import { fitsInference, maxConcurrentSessions } from './concurrency'
import { validTPDegrees } from './config-rules'
import { GPU_MEMORY_UTILIZATION } from './constants'
import { calculateInferenceVRAM } from './inference'
import { clampKVTier, kvTierSummary } from './kv-tier'
import { calculateMultiNodeVRAM } from './multi-node'
import { estimatePerformance } from './performance'
import type { ReplicaCandidate, SizingInput, SizingResult, SizingUnit } from './types'

/** A GPU sold as a rack (NVL72) rather than an 8-way server */
export function isRackGPU(gpu: GPU): boolean {
  return gpu.max_gpus_per_node > 8
}

/** Every replica shape vLLM accepts: TP inside one server/rack, plus full servers as pipeline stages */
export function replicaCandidates(model: Model, gpu: GPU): Array<{ tp: number; nodes: number }> {
  const max = gpu.max_gpus_per_node
  const tps = validTPDegrees(model, max)
  const single = tps.map((tp) => ({ tp, nodes: 1 }))
  if (isRackGPU(gpu) || max === 1 || !tps.includes(max)) return single
  const stages = Array.from({ length: 7 }, (_, i) => ({ tp: max, nodes: i + 2 }))
  return [...single, ...stages]
}

/** Per-GPU memory of one replica at `sessions` simultaneous sessions (the worker's composition) */
function perGPUBreakdown(input: SizingInput, tp: number, nodes: number, sessions: number) {
  const { model, gpu, quantization, kvQuantization, sequenceLength, fabric } = input
  const single = calculateInferenceVRAM({
    model,
    quantization,
    sequenceLength,
    batchSize: sessions,
    kvQuantization,
    concurrentUsers: sessions,
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
          batchSize: sessions,
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
  const { model, gpu, quantization, kvQuantization, sequenceLength, users } = input
  const one = perGPUBreakdown(input, tp, nodes, 1)
  const totalPerGPUGB = one.perGPU.total.toNumber()
  const memorySessions = maxConcurrentSessions({
    totalPerGPUGB,
    kvPerGPUGB: one.perGPU.kvCache.toNumber(),
    concurrentUsers: 1,
    gpuVramGB: gpu.vram_gb,
  })
  // null: no KV on the GPU, memory does not bound sessions
  const hot = Math.max(1, Math.min(users, memorySessions ?? users))
  const fits = fitsInference(totalPerGPUGB, gpu.vram_gb) && (memorySessions ?? 1) >= 1

  const settings = clampKVTier(input.kvTier)
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
          // ceiling prefill at one session, as in Inference
          recomputeSeconds:
            estimatePerformance({
              model,
              gpu,
              quantization,
              sequenceLength,
              batchSize: 1,
              kvQuantization,
              multiGPUResult: one.multi,
            }).prefillSeconds?.toNumber() ?? null,
          gpuId: gpu.id,
        })
  // An overloaded tier cannot sustain the churn: keep the HBM-only count
  const used = tier && !tier.overloaded ? tier : null
  const open = used ? Math.max(hot, Math.min(users, used.sessionsHeld)) : hot
  const sessions = used ? Math.min(hot, Math.ceil(open * settings.activeShare)) : hot

  const real = perGPUBreakdown(input, tp, nodes, sessions)
  const perf = estimatePerformance({
    model,
    gpu,
    quantization,
    sequenceLength,
    batchSize: sessions,
    kvQuantization,
    multiGPUResult: real.multi,
  })
  const perUser = (aggregate: Decimal) => aggregate.div(sessions)
  return {
    tp,
    nodes,
    gpusPerReplica: tp * nodes,
    sessionsPerReplica: open,
    hotSessions: hot,
    decodeBatch: sessions,
    tierUsed: used ? settings.tier : null,
    tierOverloaded: tier?.overloaded ?? false,
    resumeSeconds: used?.resumeSeconds ?? null,
    perUserTokensPerSecond: perUser(perf.expected.tokensPerSecond),
    perUserCeilingTokensPerSecond: perUser(perf.tokensPerSecond),
    timeToFirstToken: perf.expected.timeToFirstToken,
    ceilingTimeToFirstToken: perf.timeToFirstToken,
    totalPerGPUGB,
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
    .filter((c) => c.fits)
    .map((c) => ({ c, p: pack(input, c) }))
    .sort(
      (a, b) =>
        a.p.units - b.p.units ||
        a.p.replicas * a.c.gpusPerReplica - b.p.replicas * b.c.gpusPerReplica ||
        a.c.gpusPerReplica - b.c.gpusPerReplica,
    )
  const best = options[0]
  if (best) return { fits: true, chosen: best.c, closest: null, ...best.p }
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
  }
}
