# LLM VRAM Calculator

## What This Is

A browser-based calculator that estimates VRAM requirements and performance characteristics for running and fine-tuning large language models on various GPU configurations. Pure client-side (no backend), deployed as a static site. Supports single and multi-GPU setups with 22 quantization formats, CPU/RAM/NVMe offloading, fine-tuning estimation (Full/LoRA/QLoRA), framework presets (DeepSpeed ZeRO, Unsloth, vLLM, TGI), and shareable configurations.

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
- ✓ Zod schemas, Biome linting, Vitest testing — v1.0
- ✓ Fine-tuning VRAM estimation (Full, LoRA, QLoRA with optimizer states and gradients) — v1.1
- ✓ Training mode toggle with method/optimizer/precision selection — v1.1
- ✓ Gradient accumulation calculator (1-128 steps, effective batch size) — v1.1
- ✓ Gradient checkpointing (60% activation memory reduction) — v1.1
- ✓ Flash Attention (sequence-length-dependent KV cache reduction) — v1.1
- ✓ Training memory visualization (donut chart + breakdown table) — v1.1
- ✓ Framework presets (DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI) with auto-optimization — v1.1
- ✓ Multi-GPU training with ZeRO stage memory reduction (2x/4x/8x) — v1.1
- ✓ CPU offloading for optimizer states in training mode — v1.1

### Active

(No active requirements — plan next milestone with `/gsd:new-milestone`)

### Out of Scope

- Real-time benchmark submission/community data — complexity of backend, not needed internally
- Cloud pricing integration — pricing changes too fast, separate concern
- Energy/carbon cost estimation — nice to have but not core to VRAM decisions
- Mobile-native app — web-first, responsive design sufficient
- AI-powered recommendations — rule-based suggestions are clearer and more predictable
- FSDP / Megatron strategies — overwhelming complexity, start with DeepSpeed ZeRO only
- MoE-specific fine-tuning — niche, complex, low ROI
- Training throughput / time estimation — too many variables (data, hardware, framework)

## Context

- **Shipped:** v1.0 MVP (2026-02-09), v1.1 Fine-Tuning Estimation (2026-02-10)
- **Codebase:** 14,131 LOC TypeScript, 266 tests (18 test files), 73+ source files
- **Sister project:** raidy (storage calculator) — same team, same architecture patterns
- **Tech stack:** React 19, Vite 7, TypeScript strict, Zustand, Tailwind CSS v4, Recharts, Decimal.js, Biome, Vitest
- **Architecture:** engines/ (pure calculation), components/{inputs,outputs,comparison,layout}, data/ (JSON databases), store/ (Zustand), workers/ (Web Worker), hooks/
- **Target audience:** Engineers evaluating GPU hardware for LLM deployment and fine-tuning

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
| Separate training from inference calculation paths | Avoid 30-60% underestimation from mixing concerns | ✓ Good — accurate training VRAM estimates |
| Optimizer states always FP32 | Maintain numerical stability even in mixed precision | ✓ Good — matches real-world training behavior |
| LoRA optimizer states apply to adapters only | Not full model params — critical pitfall avoidance | ✓ Good — accurate LoRA VRAM estimates |
| QLoRA three-precision architecture | NF4 base (0.5B/param), FP16 adapters (2B/param), FP32 optimizer (8B/param) | ✓ Good — matches bitsandbytes implementation |
| DeepSpeed ZeRO stage memory: 2x/4x/8x | Not simple divide-by-N — stage-specific partitioning | ✓ Good — accurate multi-GPU training estimates |
| Framework preset auto-apply | Cascade optimization settings with mode enforcement | ✓ Good — reduces user configuration burden |
| ZeRO stage derived from frameworkPreset | Single source of truth — not stored separately | ✓ Good — prevents state inconsistency |

---
*Last updated: 2026-02-10 after v1.1 milestone completion*
