# 0003. Offloaded memory is read over the host link and bounded by host capacity

**Status:** Accepted, 2026-09-27 (v1.13.0)

## Context

A user-exported deck (Mistral Small 4 119B, HGX B300 x8, 1M context, 4096
users, weights and KV offloaded to CPU RAM) showed 1622 tok/s (offloading
ignored), 1.01 GB per GPU and "fits" for 92 TB placed in a 2 TB host. The
offloading slowdown was a step function never applied, and nothing bounded
host memory.

## Decision

- Offloaded bytes are read over the host link every decode step:
  `hostLinkGBps` reuses the KV-tier presets (PCIe 5: 50 GB/s per GPU, Grace
  NVLink-C2C: 225 on NVL72 / 396 on GB300 Desktop, NVMe: 12). The offloaded
  share per GPU mirrors the on-device layout (MLA KV duplication,
  expert-parallel replicated base). The computed slowdown is shown.
- A "Host capacity per server (GB)" input bounds offloaded memory; exceeding
  capacity x servers is "Does not fit". Defaults from DGX B300 (2048 GB RAM,
  30720 GB NVMe) for datacenter GPUs, 128 GB / 2000 GB otherwise.
- The export states concurrent users, the offloading setup and the capacity.

## Consequences

- Configurations without offloading are unchanged.
- The Mistral case reads 10.1 tok/s (~160x slower) and "Does not fit".

## Sources

NVIDIA DGX B300 user guide and datasheet (system memory, E1.S NVMe);
KV-tier preset sources in `src/engines/kv-tier.ts`.
