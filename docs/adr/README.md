# Architecture Decision Records

One file per decision: context, decision, consequences, sources. Status is
Proposed, Accepted or Superseded. Detailed designs live in
`docs/superpowers/specs/`; an ADR records why, the spec records how.

| # | Decision | Status |
|---|---|---|
| [0001](0001-measured-weight-sizes.md) | Weight sizes are measured per format, a formula only as fallback | Accepted (v1.12.0) |
| [0002](0002-moe-base-expert-split.md) | MoE base and experts priced at measured rates from the HF dtype summary | Accepted (v1.13.0) |
| [0003](0003-offload-over-host-link.md) | Offloaded memory is read over the host link and bounded by host capacity | Accepted (v1.13.0) |
| [0004](0004-single-configuration-rule-set.md) | One configuration rule set, enforced in the store, every correction shown | Accepted |
| [0005](0005-presales-audience-and-ui-scope.md) | Presales/datacenter audience: essential vs advanced inputs, feature freeze | Accepted |
| [0006](0006-gpu-topology-data.md) | GPU topology: sold server sizes and NVLink groups verified per card | Proposed |
