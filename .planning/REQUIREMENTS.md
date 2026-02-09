# Requirements: LLM VRAM Calculator

**Defined:** 2026-02-09
**Core Value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Layer

- [ ] **DATA-01**: Curated GPU database sourced from open datasets (dbgpu/gpu-info-api) with VRAM, memory bandwidth, FLOPS, and interconnect specs
- [ ] **DATA-02**: Curated model database sourced from HuggingFace config.json with parameter count, hidden_size, num_layers, num_attention_heads, num_kv_heads, intermediate_size, architecture type (dense/MoE)
- [ ] **DATA-03**: User can enter custom model specs manually (parameter count, layers, hidden size, attention heads, KV heads, intermediate size, architecture type)
- [ ] **DATA-04**: User can enter custom GPU specs manually (VRAM amount, memory bandwidth, FLOPS)
- [ ] **DATA-05**: GPU database includes NVIDIA datacenter (A100, H100, H200, B200), NVIDIA consumer (RTX 3090, 4090, 5090), AMD (MI300X), and Apple Silicon (M1-M4 Ultra) with unified memory notation
- [ ] **DATA-06**: Model database includes 30+ popular models (LLaMA 2/3, Mistral, Mixtral, Qwen, Phi, DeepSeek, Gemma, Command-R) with accurate architecture params
- [ ] **DATA-07**: Build script or utility to refresh model specs from HuggingFace API and GPU specs from open databases

### Inference Calculation

- [ ] **INFER-01**: Calculate model weight memory by quantization format (FP32, FP16, BF16, INT8, INT4, NF4, GPTQ, AWQ, GGUF Q4/Q5/Q6/Q8)
- [ ] **INFER-02**: Calculate KV cache memory by sequence length, batch size, and number of KV heads (supporting GQA/MQA architectures)
- [ ] **INFER-03**: Calculate activation memory for inference
- [ ] **INFER-04**: Account for framework overhead (CUDA context, memory allocator fragmentation, temporary buffers)
- [ ] **INFER-05**: Support KV cache quantization (FP16, FP8, INT8, INT4) separately from weight quantization
- [ ] **INFER-06**: User can configure sequence length (512 to 131072 tokens) and batch size (1 to 64)
- [ ] **INFER-07**: Correctly handle MoE architectures (loaded params vs active params, expert memory)
- [ ] **INFER-08**: Estimate tokens/second throughput based on GPU compute and memory bandwidth (roofline model)
- [ ] **INFER-09**: Estimate time-to-first-token (TTFT) for prefill phase

### Multi-GPU

- [ ] **MGPU-01**: User can select number of GPUs (1-8) for model distribution
- [ ] **MGPU-02**: Calculate VRAM per GPU accounting for tensor parallelism memory distribution
- [ ] **MGPU-03**: Account for communication overhead and memory replication (10-20% overhead)
- [ ] **MGPU-04**: User can select interconnect type (NVLink, PCIe) with bandwidth impact on performance estimates
- [ ] **MGPU-05**: Support pipeline parallelism as sharding strategy with per-stage memory accounting
- [ ] **MGPU-06**: Show memory distribution across GPUs with per-GPU utilization

### Visualization & UX

- [ ] **VIZ-01**: Memory breakdown chart showing weights, KV cache, activations, overhead as stacked/donut visualization
- [ ] **VIZ-02**: Clear fit/no-fit indicator with total VRAM needed vs available, percentage used
- [ ] **VIZ-03**: When model doesn't fit, show actionable recommendations (lower quantization, reduce context, add GPUs)
- [ ] **VIZ-04**: URL hash persistence for sharing configurations (same pattern as raidy)
- [ ] **VIZ-05**: Side-by-side comparison of 2-3 configurations (different GPUs or quantization settings)
- [ ] **VIZ-06**: Responsive layout with input panel and results panel
- [ ] **VIZ-07**: Dark mode support (matching raidy)

### Infrastructure

- [ ] **INFRA-01**: React 19 + TypeScript strict + Vite + Zustand + Tailwind CSS + Recharts stack
- [ ] **INFRA-02**: Biome for linting/formatting (matching raidy)
- [ ] **INFRA-03**: Vitest for unit testing with engine calculation coverage
- [ ] **INFRA-04**: Decimal.js for precision arithmetic in all calculation engines
- [ ] **INFRA-05**: Zod schemas for model and GPU data validation
- [ ] **INFRA-06**: Static site deployment (Vercel/GitHub Pages — no backend)
- [ ] **INFRA-07**: Web Workers for heavy calculations to prevent UI blocking

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Fine-tuning

- **TRAIN-01**: Estimate VRAM for full fine-tuning (optimizer states + gradients + activations)
- **TRAIN-02**: Estimate VRAM for LoRA fine-tuning (adapter params, reduced optimizer states, rank configuration)
- **TRAIN-03**: Estimate VRAM for QLoRA (quantized base + LoRA adapters)
- **TRAIN-04**: Side-by-side comparison of Full vs LoRA vs QLoRA memory requirements
- **TRAIN-05**: Gradient accumulation calculator (effective batch size trade-offs)
- **TRAIN-06**: Dataset size memory estimation (samples, tokens, epochs)

### Advanced Features

- **ADV-01**: Optimization framework presets (vLLM, TGI, Unsloth, DeepSpeed ZeRO)
- **ADV-02**: CPU/NVMe offloading estimation with performance penalty
- **ADV-03**: Energy consumption and cost estimation
- **ADV-04**: Memory optimization suggestions engine (rule-based recommendations)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Model execution/inference | Calculator only — link to vLLM, Ollama, TGI |
| Model downloading/hosting | Link to HuggingFace instead |
| User accounts/authentication | Stateless browser tool, localStorage for preferences |
| Community benchmark submission | Requires backend + moderation — out of scope for static site |
| Real-time GPU monitoring | Different product — users have nvidia-smi |
| Mobile native app | Responsive web sufficient |
| Cloud pricing integration | Prices change too fast, separate concern |
| AI-powered recommendations | Rule-based suggestions are clearer and more predictable |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | — | Pending |
| DATA-02 | — | Pending |
| DATA-03 | — | Pending |
| DATA-04 | — | Pending |
| DATA-05 | — | Pending |
| DATA-06 | — | Pending |
| DATA-07 | — | Pending |
| INFER-01 | — | Pending |
| INFER-02 | — | Pending |
| INFER-03 | — | Pending |
| INFER-04 | — | Pending |
| INFER-05 | — | Pending |
| INFER-06 | — | Pending |
| INFER-07 | — | Pending |
| INFER-08 | — | Pending |
| INFER-09 | — | Pending |
| MGPU-01 | — | Pending |
| MGPU-02 | — | Pending |
| MGPU-03 | — | Pending |
| MGPU-04 | — | Pending |
| MGPU-05 | — | Pending |
| MGPU-06 | — | Pending |
| VIZ-01 | — | Pending |
| VIZ-02 | — | Pending |
| VIZ-03 | — | Pending |
| VIZ-04 | — | Pending |
| VIZ-05 | — | Pending |
| VIZ-06 | — | Pending |
| VIZ-07 | — | Pending |
| INFRA-01 | — | Pending |
| INFRA-02 | — | Pending |
| INFRA-03 | — | Pending |
| INFRA-04 | — | Pending |
| INFRA-05 | — | Pending |
| INFRA-06 | — | Pending |
| INFRA-07 | — | Pending |

**Coverage:**
- v1 requirements: 36 total
- Mapped to phases: 0
- Unmapped: 36 (pending roadmap creation)

---
*Requirements defined: 2026-02-09*
*Last updated: 2026-02-09 after initial definition*
