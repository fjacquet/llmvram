# Milestones

## v1.0 MVP (Shipped: 2026-02-09)

**Phases completed:** 5 phases, 18 plans
**LOC:** 9,017 TypeScript | **Tests:** 144 passing | **Source files:** 57

**Key accomplishments:**

- Zod-validated model (37) and GPU (19) databases with HuggingFace refresh scripts
- Precision VRAM calculation engine with 22 quantization formats, GQA/MQA KV cache, MoE architecture support
- Full calculator UI with responsive layout, dark mode, real-time Recharts visualization
- Multi-GPU engine with tensor/pipeline parallelism, interconnect validation, CPU/RAM/NVMe offloading
- URL hash sharing with lz-string compression and side-by-side comparison with diff highlighting
- Bonus: Quantization expansion (13 to 22 formats), offloading engine (originally v2 scope)

**Tech debt accepted:**

- `npm run build` tsconfig.node.json file list issue (workaround: `npx vite build`)
- 10 human verification items pending (visual/interactive browser testing)

---

## v1.1 Fine-Tuning Estimation (Shipped: 2026-02-10)

**Phases completed:** 5 phases (6-10), 12 plans
**LOC:** 14,131 TypeScript (+13,019 from v1.0) | **Tests:** 266 passing (18 test files) | **Source files:** 73 modified

**Key accomplishments:**

- Full/LoRA/QLoRA fine-tuning VRAM calculation engines with accurate optimizer states, gradients, activations, and adapter parameter sizing
- Training mode UI with method/optimizer/precision selection, URL-shareable configuration
- Memory optimization features: gradient accumulation (1-128 steps), gradient checkpointing (60% activation reduction), Flash Attention
- Training memory visualization with donut chart and breakdown table showing trainable vs frozen parameters
- Framework presets (DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI) with auto-optimization and CPU offloading
- Zero deviations across all 12 plans — ~40 minutes total execution time

**Tech debt accepted:**

- Summaries lack `one_liner` field (v1.1 plans used rapid TDD execution format)
- Human verification items pending (visual/interactive browser testing for training mode)

---

