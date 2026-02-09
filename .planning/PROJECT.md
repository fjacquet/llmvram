# LLM VRAM Calculator

## What This Is

A browser-based calculator that estimates VRAM requirements and performance characteristics for running large language models on various GPU configurations. Pure client-side (no backend), deployed as a static site. Supports single and multi-GPU setups with 22 quantization formats, CPU/RAM/NVMe offloading, and shareable configurations.

## Core Value

Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what quantization, sharding, or hardware changes would make it work.

## Requirements

### Validated

- ✓ Curated model database (37 models with accurate architecture params) — v1.0
- ✓ Curated GPU database (19 GPUs: NVIDIA DC/consumer, AMD MI300X, Apple Silicon) — v1.0
- ✓ Custom model input (parameter count, layers, hidden size, attention heads) — v1.0
- ✓ Custom GPU input (VRAM, bandwidth, FLOPS) — v1.0
- ✓ VRAM estimation for inference (weights, KV cache, activations, overhead) — v1.0
- ✓ 22 quantization formats (FP32, FP16, BF16, INT8, INT4, NF4, GPTQ, AWQ, GGUF Q2-Q8, NVFP4/6) — v1.0
- ✓ KV cache with GQA/MQA support and independent quantization — v1.0
- ✓ MoE architecture handling (total params for VRAM, active for compute) — v1.0
- ✓ Performance estimation (tokens/sec, TTFT via roofline model) — v1.0
- ✓ Multi-GPU support (1-8 GPUs, tensor/pipeline parallelism) — v1.0
- ✓ Interconnect awareness (NVLink, PCIe) with validation warnings — v1.0
- ✓ CPU/RAM/NVMe offloading with performance impact estimation — v1.0
- ✓ Memory breakdown visualization (Recharts donut chart + table) — v1.0
- ✓ Fit/no-fit indicator with actionable recommendations — v1.0
- ✓ URL hash sharing with lz-string compression — v1.0
- ✓ Side-by-side comparison (up to 3 configs) with diff highlighting — v1.0
- ✓ Responsive layout, dark mode, Web Workers for non-blocking calculations — v1.0
- ✓ Zod schemas, Biome linting, Vitest testing (144 tests), static deployment — v1.0

### Active

- [ ] Estimate VRAM for fine-tuning (full, LoRA, QLoRA with optimizer states and gradients)
- [ ] Gradient accumulation calculator (effective batch size trade-offs)
- [ ] Optimization framework presets (vLLM, TGI, Unsloth, DeepSpeed ZeRO)

### Out of Scope

- Real-time benchmark submission/community data — complexity of backend, not needed internally
- Cloud pricing integration — pricing changes too fast, separate concern
- Energy/carbon cost estimation — nice to have but not core to VRAM decisions
- Mobile-native app — web-first, responsive design sufficient
- AI-powered recommendations — rule-based suggestions are clearer and more predictable

## Context

- **Shipped:** v1.0 MVP (2026-02-09) — 9,017 LOC TypeScript, 144 tests, 57 source files
- **Sister project:** raidy (storage calculator) — same team, same architecture patterns
- **Tech stack:** React 19, Vite 7, TypeScript strict, Zustand, Tailwind CSS v4, Recharts, Decimal.js, Biome, Vitest
- **Architecture:** engines/ (pure calculation), components/{inputs,outputs,comparison,layout}, data/ (JSON databases), store/ (Zustand), workers/ (Web Worker), hooks/
- **Target audience:** Engineers evaluating GPU hardware for LLM deployment

## Constraints

- **Tech stack**: React 19, Vite, TypeScript strict, Zustand, Tailwind, Recharts — consistency with raidy
- **Client-side only**: All computation in browser, no backend — static deployment
- **Linting**: Biome — consistent code style
- **Testing**: Vitest with 75% coverage thresholds on engines/utils

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Mirror raidy architecture | Consistency across internal tools, proven patterns | ✓ Good — clean separation of concerns |
| Client-side only | Static deployment, no infrastructure cost | ✓ Good — zero ops burden |
| Curated + custom model/GPU data | Convenience for common hardware, flexibility for niche configs | ✓ Good — 37 models, 19 GPUs + custom input |
| Decimal.js for all arithmetic | Avoid floating-point precision errors | ✓ Good — consistent results |
| Web Workers for calculations | Prevent UI blocking on complex calculations | ✓ Good — with sync fallback |
| Zod schemas as type source of truth | Single source for validation + TypeScript types | ✓ Good — no type drift |
| Total MoE params for VRAM | All expert weights must fit in memory | ✓ Good — accurate VRAM estimates |
| URL hash (not localStorage) for config | Shareable links are more valuable than auto-persist | ✓ Good — lz-string compression keeps URLs manageable |
| Transient comparison store | Comparisons are ephemeral, not worth persisting | ✓ Good — simple implementation |
| Fine-tuning deferred to v2 | Inference is the primary use case, ship v1 faster | ✓ Good — delivered complete inference calculator |

---
*Last updated: 2026-02-09 after v1.0 milestone*
