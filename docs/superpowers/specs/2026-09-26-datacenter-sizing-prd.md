# PRD: Datacenter Sizing for Frontier MoE Models

**Date:** 2026-09-26
**Status:** Draft, 4a in design

## Problem

A team sizing long-context agentic serving of Kimi K3 or Qwen3.8 on GB300 NVL72
racks (e.g. Dell XE9712) cannot get a trustworthy "how many racks, how many
concurrent sessions" answer. The calculator models tensor parallelism only, which
duplicates MLA KV on every GPU, has no session-count answer, ignores
linear-attention state, and has no storage tier for cold KV (Dell Lightning).

## Users

Infrastructure architects and pre-sales engineers sizing on-premises AI
datacenters. They know the hardware, not every model's attention internals.

## Goals

1. Model the layout production actually uses for MoE + MLA models.
2. Answer "max concurrent sessions at context X on configuration Y" directly.
3. Count every byte a session holds, including constant-size state.
4. Show how a KV offload tier changes session capacity and resume latency.

## Non-goals

Training, cost / TCO, power and cooling, scheduling policy, accuracy of
quantized models.

## Scope, in order

| Piece | What | Depends on |
|---|---|---|
| 4a | Expert parallelism + DP-attention strategy (intra-node): experts / N, base replicated, KV per rank; all-to-all cost = latency + bytes / bandwidth | — |
| 4b | Max-session solver: sessions per GPU and per cluster at a given context, from free VRAM after weights | 4a |
| 4c | Linear-attention / Mamba state per session (Qwen3.8, Qwen3.6, Kimi K3 KDA, Kimi Linear) | — |
| 4d | Dell Lightning KV tier: hot KV in HBM, cold KV on the parallel file system; capacity gain vs reload bandwidth on resume | 4b, 4c |

## Success criteria

- Every new rule cites vLLM (or SGLang) plus a second source, as in the audit
  design (D1).
- Kimi K3 MXFP4 on one NVL72 at 262k: the tool reports sessions per rack, and the
  EP result is no longer bounded by duplicated MLA KV.
- A shared link reproduces the same answer.
- Tests, typecheck, lint and build green on every piece; one PR per piece.

## Open questions

- Lightning: published bandwidth per client and per rack, and how vLLM /
  LMCache / Dynamo address it (answer during 4d spec).
- Whether 4b should also solve for context length given a session target.
