# LLM VRAM Calculator

## What This Is

A browser-based calculator that estimates VRAM requirements and performance characteristics for running and fine-tuning large language models on various GPU configurations. Inspired by apxml.com's VRAM calculator, built as an internal tool following the same architecture patterns as raidy (React + TypeScript + Vite, pure client-side).

## Core Value

Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what quantization, sharding, or hardware changes would make it work.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Estimate VRAM for LLM inference (weights, KV cache, activations, overhead)
- [ ] Estimate VRAM for fine-tuning (full, LoRA, QLoRA)
- [ ] Support multiple quantization formats (FP32, FP16, BF16, INT8, INT4, GPTQ, AWQ, GGUF variants)
- [ ] Curated model database (popular LLMs with parameter counts, architectures, layer configs)
- [ ] Custom model input (manually specify parameters, layers, hidden size, etc.)
- [ ] Curated GPU database (NVIDIA datacenter, consumer, Apple Silicon with VRAM, bandwidth, FLOPS)
- [ ] Custom GPU input (manually specify VRAM, bandwidth)
- [ ] Multi-GPU support with interconnect awareness (NVLink, PCIe) and sharding strategies
- [ ] Memory breakdown visualization (weights, KV cache, activations, optimizer states, gradients)
- [ ] Performance estimation (tokens/second, time-to-first-token)
- [ ] Batch size and sequence length parameter controls
- [ ] Fit/no-fit indicator with recommendations when model doesn't fit
- [ ] Static site deployment (GitHub Pages / Vercel — no backend)

### Out of Scope

- Real-time benchmark submission/community data — complexity of backend, not needed internally
- Cloud pricing integration — pricing changes too fast, separate concern
- Energy/carbon cost estimation — nice to have but not core to VRAM decisions
- Mobile-native app — web-first, responsive design sufficient

## Context

- **Sister project:** raidy (storage calculator) at /Users/fjacquet/Projects/raidy — same team, same patterns
- **Inspiration:** apxml.com/tools/vram-calculator — feature-rich reference implementation
- **Raidy stack:** React 19, Vite 7, TypeScript strict, Zustand, Tailwind CSS v4, Recharts, Biome, Vitest
- **Raidy patterns:** engines/ for calculation logic, components/{inputs,outputs,common,layout}, data/ for JSON databases, store/ for Zustand, workers/ for heavy computation, hooks/ for custom hooks
- **Target audience:** Internal engineers evaluating GPU hardware for LLM deployment and fine-tuning

## Constraints

- **Tech stack**: Mirror raidy's stack (React 19, Vite, TypeScript, Zustand, Tailwind, Recharts) — consistency across internal tools, open to different libraries where better suited
- **Client-side only**: All computation in browser, no backend — enables static deployment
- **Linting**: Biome (matching raidy) — consistent code style across tools
- **Testing**: Vitest (matching raidy) — engine logic must be testable

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mirror raidy architecture | Consistency across internal tools, proven patterns | — Pending |
| Client-side only | Static deployment, no infrastructure cost | — Pending |
| Curated + custom model/GPU data | Convenience for common hardware, flexibility for niche configs | — Pending |
| Inference + fine-tuning in v1 | Both are core use cases for the team | — Pending |
| Multi-GPU with interconnect awareness | Essential for datacenter GPU planning | — Pending |

---
*Last updated: 2026-02-09 after initialization*
