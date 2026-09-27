# 0007. Multi-node prefill and decode from bytes over the fabric

**Status:** Accepted, 2026-09-27
**Supersedes:** the prefill/decode efficiency part of
`docs/superpowers/specs/2026-09-16-multi-node-scaling-design.md`
**Spec:** `docs/superpowers/specs/2026-09-27-v2-config-rules-topology-ui-design.md` (Section 3b)

## Context

Inter-node efficiency came from a heuristic we introduced (commit 9f33ac3):
a quadratic penalty per halving of bandwidth below 1600 GB/s per node,
clamped at a 5% floor, never measured. Two errors:

1. It turned a transfer worth under 1.3% of prefill into a ~15x penalty on
   small nodes (2x DGX Spark: 195 s TTFT for Llama 3.1 70B at 8k, vs ~13 s).
   The only clean measurement (vllm#6610, 2x GH200 PP=2) shows no change
   between 400G and 800G; the heuristic predicted 2.2x.
2. The pipeline bubble term gave a single request a cross-node prefill
   speedup that vLLM does not deliver: one prompt of at most one scheduler
   step walks the stages serially.

## Decision

- Stage-boundary transfers cost bytes / (eta x port x GPUs per node) +
  latency, like decode everywhere else in the engine.
- Prefill pipelines microbatches M = ceil(B x T / C), C = vLLM's default
  `max_num_batched_tokens`; decode keeps its existing overlap factor.
- eta for GB10 (0.37) is a secondary source and eta for HGX (0.8) is an
  assumption; both are labelled as such. Results are insensitive to them
  (doubling port speed changes decode by under 1%).
- The TTFT figure is relabelled "Prefill per request (amortized over batch B)":
  for B > 1 it is a burst's prefill divided by B, not one request's delay.
- Rejected: modelling the burst explicitly (new calculation, against the
  feature freeze of ADR 0005); keeping and re-anchoring the heuristic (no
  source supports its shape).

## Consequences

Nearly every multi-node TTFT changes: small clusters and batched HGX get
faster, single-request HGX over fast fabrics gets slower (the old speedup did
not exist). Decode moves under 10%. No KV-tier verdict flips in 100 tested
configurations. Listed as breaking changes in v2.0.0.

## Sources

vLLM `config/vllm.py`, `v1/engine/core.py`, `v1/core/sched/scheduler.py`,
`engine/arg_utils.py`, `distributed/parallel_state.py`; GPipe
(arXiv 1811.06965); Megatron-LM (arXiv 2104.04473); vllm#6610; vllm#41685;
multimodalflow.net DGX Spark dual-node NCCL RDMA measurement; arXiv 2511.15076.
