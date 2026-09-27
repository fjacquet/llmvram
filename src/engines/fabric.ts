import type { CustomFabricInput, GPU } from '@utils/schemas'
import type { FabricSpec, FabricType } from './types'

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
  },
  'infiniband-xdr': {
    type: 'infiniband-xdr',
    label: 'InfiniBand XDR 800G',
    portGBps: 100,
  },
  'ethernet-800g': {
    type: 'ethernet-800g',
    label: '800GbE (SONiC / RoCEv2)',
    portGBps: 100,
  },
  'infiniband-ndr': {
    type: 'infiniband-ndr',
    label: 'InfiniBand NDR 400G',
    portGBps: 50,
  },
  'ethernet-400g': {
    type: 'ethernet-400g',
    label: '400GbE (RoCEv2)',
    portGBps: 50,
  },
  // DGX Spark / DGX Station scale-out: ConnectX-7 at 200 Gb/s (NVIDIA Sync cluster assistant)
  'ethernet-200g': {
    type: 'ethernet-200g',
    label: '200GbE (ConnectX-7)',
    portGBps: 25,
  },
  'ethernet-100g': {
    type: 'ethernet-100g',
    label: '100GbE',
    portGBps: 12.5,
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
    }
  }
  return FABRIC_SPECS[type]
}

/**
 * Share of line rate a stage handoff reaches with GPUDirect RDMA (HGX-class nodes).
 * ASSUMPTION: the only hint is GH200 all_reduce at 45.4 of 50 GB/s. Results are
 * insensitive to it: doubling port speed changes decode by under 1% (vllm#6610:
 * 21.0 tok/s at 400G vs 21.1 at 800G on 2x GH200 PP=2).
 */
export const HGX_EFFECTIVE_FRACTION = 0.8

/**
 * Share of line rate on a GPU without GPUDirect RDMA (`gpu.gpudirect_rdma === false`;
 * DGX Spark / GB10 is the only measured case, Task 3a).
 * SECONDARY SOURCE, one measurement: NCCL send/recv ~9 GB/s vs 24.6 GB/s with RDMA on
 * a Spark 200G link (multimodalflow.net, DGX Spark dual-node NCCL RDMA).
 */
export const GB10_EFFECTIVE_FRACTION = 0.37

/**
 * Fixed cost of one stage handoff: the 10 us RDMA small-message floor (arXiv 2511.15076).
 * vLLM's metadata exchange is likely 50-200 us, still under 1% of a decode step.
 */
export const FABRIC_HOP_LATENCY_S = 10e-6

/**
 * eta: the fraction of the fabric's line rate a stage handoff reaches on this GPU's
 * node. Keyed by the GPU's own `gpudirect_rdma` data field (Task 3a), never by
 * `gpu.id`: a custom GPU or a future no-RDMA card gets the right value automatically.
 */
export function effectiveFraction(gpu: GPU): number {
  return gpu.gpudirect_rdma === false ? GB10_EFFECTIVE_FRACTION : HGX_EFFECTIVE_FRACTION
}

/**
 * Effective scale-out bandwidth of one node, GB/s: one NIC per GPU, and each TP rank
 * sends its 1/tp slice, so the node's aggregate applies, times eta.
 */
export function interNodeGBps(portGBps: number, gpusPerNode: number, fraction: number): number {
  return perNodeFabricGBps(portGBps, gpusPerNode) * fraction
}

/**
 * One pipeline stage-boundary transfer of `tokens` tokens, in seconds. vLLM sends
 * hidden_states and residual (2 tensors x 2 bytes, BF16) per token
 * (distributed/parallel_state.py send_tensor_dict).
 */
export function fabricHopSeconds(tokens: number, hiddenSize: number, gbps: number): number {
  return (tokens * hiddenSize * 2 * 2) / (gbps * 1e9) + FABRIC_HOP_LATENCY_S
}

/**
 * vLLM's default max_num_batched_tokens for the OpenAI server (engine/arg_utils.py):
 * 16384 on GPUs with >= 160 GB, 8192 on >= 70 GB other than A100, else 2048.
 */
export function maxNumBatchedTokens(gpu: GPU): number {
  if (gpu.vram_gb >= 160) return 16384
  if (gpu.vram_gb >= 70 && !/a100/i.test(gpu.name)) return 8192
  return 2048
}

/** Microbatches a burst of B prompts of T tokens splits into at chunk size C: ceil(B x T / C) */
export function prefillMicrobatches(
  batchSize: number,
  sequenceLength: number,
  chunkTokens: number,
): number {
  return Math.max(1, Math.ceil((batchSize * sequenceLength) / chunkTokens))
}

/**
 * GPipe fill of a prefill over `stages` pipeline stages, M / (M + N - 1). Times N it is
 * the speedup N x M / (M + N - 1); one microbatch (M = 1) walks the stages serially and
 * gains nothing (GPipe, arXiv 1811.06965).
 */
export function prefillPipelineFill(microbatches: number, stages: number): number {
  if (stages <= 1) return 1
  return microbatches / (microbatches + stages - 1)
}
