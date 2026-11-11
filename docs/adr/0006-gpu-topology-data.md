# 0006. GPU topology: sold server sizes and NVLink groups verified per card

**Status:** Accepted, 2026-09-27
**Spec:** `docs/superpowers/specs/2026-09-27-v2-config-rules-topology-ui-design.md`

## Context

The data described a GPU's scale-up link by one `interconnect` type and a
`max_gpus_per_node`. Real topologies impose specific card combinations, and
a per-card audit (one vendor source per card) found:

- HGX/DGX H200 SXM ships only as 4-GPU or 8-GPU NVSwitch baseboards; H200 NVL
  (PCIe) uses 2-way or 4-way NVLink bridges. One entry mixed both products.
- H100 PCIe and A100 PCIe bridges pair 2 cards, so TP beyond 2 crosses PCIe;
  both were modelled as 8-way NVLink. H100 PCIe FP16 used the sparse figure.
- A100 SXM was priced at NVLink-4 bandwidth (900 GB/s) instead of 600.
- DGX Spark (GB10) is one GPU per node; two Sparks are two nodes over 200 GbE.
- vLLM refuses tensor-parallel degrees that don't divide the attention heads
  (and the KV-head rule); the engine computed them as valid.
- The store field `numGPUs` was both "GPUs per server" and the parallel degree.

## Decision

- `numGPUs` is the parallel degree of one replica inside one server, capped by
  `max_gpus_per_node`; the label becomes "GPUs per replica (in one server)".
  No purchasable-size field (`valid_gpu_counts`): an 8-GPU server running four
  TP-2 replicas is a real presales answer, and restricting sizes would change
  shared links.
- Add `nvlink_bridge: { type, size }`: NVLink bandwidth applies while the TP
  group fits the bridge, otherwise the card's PCIe generation
  (`resolveInterconnect(gpu, tpDegree)`). Add `nvlink-3` (600 GB/s) and
  `unified_memory`.
- Split `nvidia-h200-141gb` (now SXM only) from a new `nvidia-h200-nvl-141gb`;
  ids are never renamed, so old links resolve to SXM.
- Correct GB10 (1 per node, no scale-up link), A100 SXM (nvlink-3), H100/A100
  PCIe (PCIe + bridge), L40S and RTX 6000 Ada (pcie-4), H100 PCIe FP16 (756).
- Tensor-parallel degree validity is a hard rule (R14 in ADR 0004's rule set):
  invalid degrees are not selectable and are corrected with a notice.
- Deferred: fabric prefill floor for small clusters; AMD per-degree mesh
  bandwidth (MI300X/MI325X 896 GB/s priced at 1075).

## Consequences

Numbers change for H100/A100 PCIe (TP-8 decode up to -26%), A100 SXM, GB10,
L40S and RTX 6000 Ada TTFT; shipped as v2.0.0 with a breaking-changes list.

## Amendment 2026-11-11: baseboards are one machine

An HGX/OAM baseboard is an integrated 8-GPU unit: 1/8 of it cannot be activated.
`min_gpus_per_node` (optional, 8) marks exactly the SXM/OAM baseboards (H100 SXM,
H200 SXM, B200, HGX B300, HGX Rubin NVL8, A100 SXM, MI300X/MI325X/MI350X/MI355X).
Config-rules R1 raises `numGPUs` to it and `allowedOptions` offers only that count in
Inference and Training (Training only where the count is shown, i.e. with a ZeRO
preset). It wins over R14: where tensor parallel refuses 8 (heads % 8 != 0: GLM 4.7
Flash, Gemma 3 1B) the count stays 8 and soft warning W9 points to pipeline parallel,
because R14 and R1 would otherwise cycle. PCIe cards and workstation parts keep
1..`max_gpus_per_node`; racks (72) remain Sizing-only. Breaking: shared links with fewer
GPUs on these parts are corrected ("Shared link adjusted"). Sizing (ADR 0010) is
unchanged: it lists racks only.

## Sources

- NVIDIA HGX AI Factory reference architecture, components:
  https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
- PNY H200 NVL datasheet:
  https://www.pny.com/file%20library/company/support/linecards/data-center-gpus/h200-nvl-datasheet.pdf
- HPE 2-way / 4-way NVLink bridge for H200 NVL: PSN1014857028PLEN, PSN1014856854VNEN
- NVIDIA H100 datasheet; Lenovo LP1732 (H100 PCIe); NVIDIA A100 page;
  Lenovo LP1812 (L40S); NVIDIA RTX 6000 Ada datasheet
- NVIDIA DGX Spark and DGX Station pages; NVIDIA Sync cluster assistant:
  https://docs.nvidia.com/sync/latest/cluster-assistant.html
- vLLM `vllm/config/model.py` (heads % tp) and
  `vllm/model_executor/layers/linear.py` (KV-head divisibility)
