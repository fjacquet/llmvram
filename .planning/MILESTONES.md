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
