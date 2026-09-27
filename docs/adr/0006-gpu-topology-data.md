# 0006. GPU topology: sold server sizes and NVLink groups verified per card

**Status:** Proposed, 2026-09-27 (pending the per-card verification table)

## Context

The data describes a GPU's scale-up link by one `interconnect` type and a
`max_gpus_per_node`. Real topologies impose specific card combinations:

- HGX/DGX H200 SXM ships only as 4-GPU or 8-GPU NVSwitch baseboards.
- H200 NVL (PCIe) uses 2-way or 4-way NVLink bridges; H100 NVL and A100 PCIe
  bridges pair 2 cards, so TP beyond the bridge crosses PCIe.
- One entry (`nvidia-h200-141gb`) mixes the SXM and NVL products.
- DGX Spark (GB10) is one GPU per node; two Sparks are two nodes over 200 GbE.
- The store field `numGPUs` means both "GPUs per server" and the parallel
  degree, while a real 8-GPU server can run several smaller TP groups.
- vLLM requires the attention heads to divide by the TP degree.

## Decision (proposed, to be finalized from the per-card table)

- Verify every GPU entry against a card-specific vendor source; no reasoning
  by analogy; unsourced entries are marked unverified.
- Add the smallest data needed to prevent impossible choices (candidates:
  valid server sizes as sold, NVLink group size), enforced through ADR 0004.
- Split entries that mix two products while keeping the existing id stable
  (ids are never renamed: shared links depend on them).
- Decide the meaning of `numGPUs` (server size vs parallel degree).

## Sources

- NVIDIA HGX AI Factory reference architecture, components:
  https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
- PNY H200 NVL datasheet:
  https://www.pny.com/file%20library/company/support/linecards/data-center-gpus/h200-nvl-datasheet.pdf
- HPE 2-way / 4-way NVLink bridge for H200 NVL: PSN1014857028PLEN, PSN1014856854VNEN
- NVIDIA DGX Spark cluster assistant: https://docs.nvidia.com/sync/latest/cluster-assistant.html
