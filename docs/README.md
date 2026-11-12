# Documentation

Start with the [README](../README.md) and the in-app Guide. This index is for readers who want
the reasoning and the sources behind the figures.

## Why the figures are what they are

| Question                                                | Read                                                                                                                                           |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Which engine do the figures follow?                     | [ADR 0011, vLLM is the reference engine](adr/0011-vllm-reference-engine.md)                                                                    |
| Why is the speed "expected" with a "ceiling" beside it? | [ADR 0009](adr/0009-calibrated-expected-speed-and-sparse-attention.md), [calibration points](calibration/2026-11-11-vllm-efficiency-points.md) |
| How does Sizing count racks?                            | [ADR 0010, Sizing mode](adr/0010-sizing-mode.md)                                                                                               |
| Where do weight sizes come from?                        | [ADR 0001, measured weight sizes](adr/0001-measured-weight-sizes.md), [ADR 0002, MoE split](adr/0002-moe-base-expert-split.md)                 |
| How are multi-GPU and multi-server setups modelled?     | [ADR 0006, GPU topology](adr/0006-gpu-topology-data.md), [ADR 0007, multi-node](adr/0007-multi-node-prefill-model.md)                          |
| Why is a GPU's memory or TFLOPS what it is?             | [GPU specs reference](gpu-specs-reference.md), [ADR 0008](adr/0008-per-precision-compute-and-usable-rack-memory.md)                            |
| What estimation mistakes does the tool avoid?           | [VRAM calculation pitfalls](vram-calculation-pitfalls.md)                                                                                      |
| Why did a setting change by itself?                     | [ADR 0004, configuration rules](adr/0004-single-configuration-rule-set.md)                                                                     |

All decisions: [ADR index](adr/README.md).

## For developers

- [Developer guide](development.md): commands, stack, project layout
- [ARCHITECTURE.md](../ARCHITECTURE.md): engines, stores, data flow
- [CLAUDE.md](../CLAUDE.md): conventions and domain rules
- `superpowers/specs/` and `superpowers/plans/`: design documents and implementation plans
