import type { CustomFabricInput } from '@utils/schemas'
import type { FabricSpec, FabricType } from './types'

/**
 * Reference per-node bandwidth for the efficiency formulas, GB/s
 *
 * 1600 GB/s is an 8-GPU node on 1.6TbE — currently the fastest mainstream
 * scale-out build. Efficiency is expressed as a penalty in halvings below it.
 */
export const FABRIC_REFERENCE_GBPS = 1600

/**
 * Efficiency ceiling for a pipeline stage boundary
 *
 * A stage handoff is never free, even at unlimited bandwidth: there is a
 * serialization point and a synchronization. Caps the prefill formula.
 */
export const PP_BASE_EFFICIENCY = 0.95

/** Floor applied to both efficiency formulas, to keep pathological inputs sane */
const EFFICIENCY_FLOOR = 0.05

/**
 * Scale-out fabric presets
 *
 * portGBps is UNIDIRECTIONAL GB/s per port: 800 Gb/s = 100 GB/s.
 *
 * Grounded in current hardware — Broadcom Tomahawk 6 (102.4 Tb/s, 128x800G or
 * 64x1.6T) has shipped since October 2025 with hardened SONiC available, and
 * IEEE 802.3dj finalizes 1.6T optics mid-2026.
 */
export const FABRIC_SPECS: Record<Exclude<FabricType, 'custom'>, FabricSpec> = {
  'ethernet-1600g': {
    type: 'ethernet-1600g',
    label: '1.6TbE (SONiC / RoCEv2)',
    portGBps: 200,
    classFactor: 1.0,
  },
  'infiniband-xdr': {
    type: 'infiniband-xdr',
    label: 'InfiniBand XDR 800G',
    portGBps: 100,
    classFactor: 1.02,
  },
  'ethernet-800g': {
    type: 'ethernet-800g',
    label: '800GbE (SONiC / RoCEv2)',
    portGBps: 100,
    classFactor: 1.0,
  },
  'infiniband-ndr': {
    type: 'infiniband-ndr',
    label: 'InfiniBand NDR 400G',
    portGBps: 50,
    classFactor: 1.02,
  },
  'ethernet-400g': {
    type: 'ethernet-400g',
    label: '400GbE (RoCEv2)',
    portGBps: 50,
    classFactor: 1.0,
  },
  'ethernet-100g': {
    type: 'ethernet-100g',
    label: '100GbE',
    portGBps: 12.5,
    classFactor: 1.0,
  },
}

/**
 * Per-node aggregate scale-out bandwidth
 *
 * The standard AI-node build is one NIC per GPU, and collective libraries
 * (NCCL/RCCL) stripe a pipeline stage handoff across all of them. So node
 * bandwidth is port speed times GPU count, not a single port's speed — a
 * distinction worth three-to-eight times the answer.
 */
export function perNodeFabricGBps(portGBps: number, gpusPerNode: number): number {
  return portGBps * gpusPerNode
}

/** Halvings below the reference bandwidth; 0 at or above it */
function halvingsBelowReference(perNodeGBps: number): number {
  if (perNodeGBps <= 0) return Number.POSITIVE_INFINITY
  return Math.max(0, Math.log2(FABRIC_REFERENCE_GBPS / perNodeGBps))
}

function clamp(value: number): number {
  if (!Number.isFinite(value)) return EFFICIENCY_FLOOR
  return Math.min(1, Math.max(EFFICIENCY_FLOOR, value))
}

/**
 * Inter-node efficiency for the PREFILL roofline
 *
 * Prefill ships full-sequence activations across a stage boundary
 * (batch x seqlen x hidden x 2 bytes), so it is genuinely bandwidth-bound.
 *
 *   eff = PP_BASE * (1 - 0.06*L - 0.020*L^2) * classFactor,  L = halvings
 *
 * The quadratic term encodes that degradation is superlinear: the first halving
 * of bandwidth costs little, the fourth is severe.
 *
 * DERIVED FROM BANDWIDTH, NOT MEASURED — as with INTERCONNECT_SPECS.
 */
export function fabricPrefillEfficiency(perNodeGBps: number, classFactor: number): number {
  const l = halvingsBelowReference(perNodeGBps)
  return clamp(PP_BASE_EFFICIENCY * (1 - 0.06 * l - 0.02 * l * l) * classFactor)
}

/**
 * Inter-node efficiency for the DECODE roofline
 *
 * Decode ships one token's activations per stage handoff — batch x hidden x 2
 * bytes, kilobytes. The hop is latency-bound at roughly 10 microseconds against
 * a 10-20 ms decode step, so it is near-free and this formula is near-flat.
 * Applying the prefill number here would badly understate decode throughput.
 *
 * DERIVED FROM BANDWIDTH, NOT MEASURED.
 */
export function fabricDecodeEfficiency(perNodeGBps: number): number {
  return clamp(0.99 - 0.01 * halvingsBelowReference(perNodeGBps))
}

/**
 * Pipeline fill/drain ("bubble") efficiency
 *
 *   eff = M / (M + numNodes - 1),  M = max(batchSize, numNodes)
 *
 * Independent of bandwidth: a pipeline of S stages idles S-1 slots at the start
 * and end of every batch. Omitting this flatters deep pipelines badly — 4 nodes
 * at batch 1 loses about 43% to bubbles alone.
 *
 * The M assumption — microbatch count tracks batch size, floored at the stage
 * count — is a heuristic. Real serving stacks tune it independently.
 */
export function pipelineBubbleEfficiency(batchSize: number, numNodes: number): number {
  if (numNodes <= 1) return 1
  const microbatches = Math.max(batchSize, numNodes)
  return microbatches / (microbatches + numNodes - 1)
}

/**
 * Resolve a fabric selection to a concrete spec
 *
 * Custom input with no value falls back to the 800GbE preset rather than
 * throwing: the UI can hold 'custom' selected while the field is still empty.
 */
export function resolveFabricSpec(type: FabricType, custom: CustomFabricInput | null): FabricSpec {
  if (type === 'custom') {
    if (!custom) return FABRIC_SPECS['ethernet-800g']
    return {
      type: 'custom',
      label: `${custom.name} (${custom.port_gbps} GB/s/port)`,
      portGBps: custom.port_gbps,
      classFactor: 1.0,
    }
  }
  return FABRIC_SPECS[type]
}
