# KV Storage Tier (Dell Lightning / CMX / PowerScale / ObjectScale)

**Date:** 2026-09-26
**Status:** Approved 2026-09-26
**PRD:** `2026-09-26-datacenter-sizing-prd.md`, piece 4d

## Context

Long-context agentic sessions spend most of their life idle: waiting on tools, on
the user, on other agents. While idle, their KV cache (7.2 GB per Kimi K3 session
at 262k) sits in HBM doing nothing. Serving stacks now park idle KV on an external
tier and reload it on resume instead of recomputing the prefill:

- vLLM KV connectors: LMCache, the native OffloadingConnector, NVIDIA Dynamo KVBM
  (GPU → host → local SSD → remote file/object storage).
- NVIDIA CMX ("G3.5"): Ethernet-attached flash behind BlueField-4 (800 Gb/s) for
  hot reusable KV.
- Dell targets KV offload at CMX, PowerScale (NFS over RDMA), ObjectScale (S3 over
  RDMA) and Lightning FS (parallel, NVMe-oF).

The calculator today answers "how many sessions fit in HBM" (4b). With a KV tier
the question becomes "how many sessions can a rack **hold**, and how long does a
parked session take to resume".

## Published figures (the only anchors)

| Source | Figure |
|---|---|
| Dell blog, KV Cache Offload to Object Storage | XE9680, 4× H100, TP4, Qwen3-Coder-30B-A3B, ObjectScale S3-over-RDMA, vLLM + LMCache 0.4.5 + NIXL 1.1: 235K-token request, 43 GB KV, TTFT 837 ms vs 11,223 ms recompute; below 8-16K tokens offload is slower than recompute |
| Dell storage engines blog | PowerScale / ObjectScale: 1 s TTFT at 131K vs 17 s |
| StorageReview, GTC 2026 | Lightning FS up to 150 GB/s per rack unit |
| Dell Lightning FS page | targets > 16K GPUs or 4 TB/s aggregate |
| NVIDIA CMX | BlueField-4, up to 800 Gb/s |

Dell publishes **no per-GPU KV reload bandwidth** for Lightning. The 837 ms anchor
implies ≥ 51 GB/s into one 4-GPU server (43 GB / 0.837 s, prefill of the last
chunk ignored). So the tier bandwidth must be an input, with presets labelled as
estimates.

## Model

Two tiers per session: **hot** (in HBM, decoding) and **parked** (on the tier).

Inputs (new "KV storage tier" panel, off by default):

1. **Tier**: none | host memory | local NVMe | network storage (Lightning, CMX,
   PowerScale, ObjectScale). Each preset sets a per-GPU read bandwidth; a custom
   value overrides it.
2. **Active share**: fraction of sessions decoding at any moment (default 25%,
   a typical agentic duty cycle; user-set).
3. **Tier capacity (TB)**: optional; unlimited when blank.

Outputs:

- **Sessions held** = min(maxHotSessions / activeShare, tierCapacity / KV-per-session),
  reusing `maxConcurrentSessions` (4b) for the hot count.
- **Resume time** = KV per session per GPU / tier bandwidth per GPU, shown next to
  the recompute time from the existing prefill model (`prefillSeconds` at the full
  context). The UI states which is faster; below the crossover (Dell: 8-16K
  tokens) recompute wins.
- **Tier traffic** = resumes per second × KV per session, to show whether the
  fabric can sustain the churn (resumes/s = active sessions / mean active burst,
  burst length user-set, default 30 s).

Preset bandwidths (per GPU, read, **estimates**, sources in code):

| Preset | GB/s per GPU | Basis |
|---|---|---|
| Host memory (Grace C2C) | 225 | NVLink-C2C 900 GB/s bidirectional per Grace, shared by 2 GPUs, one direction (**to verify** against NVIDIA GB300 docs before landing) |
| Host memory (PCIe 5) | 50 | PCIe 5 x16 ~64 GB/s, practical |
| Local NVMe | 12 | ~4 Gen5 drives per 8-GPU server striped |
| Network storage | 12.5 | one 400 GbE storage NIC share per GPU; matches the Dell anchor (≥ 51 GB/s per 4-GPU server = 12.8 per GPU) |
| Dell Lightning FS | 12.5 | same client-side limit as network storage: Dell publishes no per-GPU figure (150 GB/s per rack unit is the storage side); the UI notes this |

## Engine

- New pure `kv-tier.ts`: `sessionsHeld(...)`, `resumeSeconds(...)`, `tierTraffic(...)`.
  No change to memory or decode engines: parked sessions hold no HBM.
- New `KVTierConfig` type; store fields + URL serialization (short keys) + defaults.
- Existing `offloading.ts` stays as is: it moves the **active** KV off-device with
  a slowdown, a different question from parking idle sessions.

## Tests

- Resume time reproduces the Dell anchor: 43 GB at 51 GB/s ≤ 0.84 s.
- Sessions held = hot / activeShare; capped by capacity.
- Resume beats recompute at 235K, loses at 4K (Dell crossover) for the anchor model.
- Tier "none" leaves every existing output unchanged.
- URL round-trip keeps the tier settings.

## Out of scope

Prefix sharing across sessions (hit ratios), eviction policies, tier write
bandwidth and endurance, cost. Multi-tier hierarchies (host + storage) beyond one
chosen tier.

## Decisions (review 2026-09-26)

1. Defaults accepted: active share 25%, active burst 30 s.
2. A named "Dell Lightning FS" preset is added (12.5 GB/s per GPU, with a note that
   Dell publishes no per-GPU figure).
3. PDF/PPTX exports: later, not in this piece.
