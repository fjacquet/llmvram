# Requirements: LLM VRAM Calculator

**Defined:** 2026-02-09
**Core Value:** Users can quickly determine whether a specific LLM fits on their available GPU hardware — and if not, what changes would make it work.

## v1.0 Requirements (Validated)

All v1.0 requirements shipped and validated. See MILESTONES.md for details.

- ✓ DATA-01 through DATA-07 — Curated GPU/model databases with custom input and refresh scripts
- ✓ INFER-01 through INFER-09 — Inference VRAM calculation with 22 quantization formats, KV cache, MoE, performance estimation
- ✓ MGPU-01 through MGPU-06 — Multi-GPU support with tensor/pipeline parallelism
- ✓ VIZ-01 through VIZ-07 — Visualization, fit indicator, URL sharing, comparison, responsive layout, dark mode
- ✓ INFRA-01 through INFRA-07 — Full tech stack, Biome, Vitest, Decimal.js, Zod, Web Workers

## v1.1 Requirements

Requirements for fine-tuning estimation milestone. Each maps to roadmap phases.

### Fine-Tuning Core

- [ ] **FTCORE-01**: User can toggle between Inference and Fine-tuning mode
- [ ] **FTCORE-02**: User can select fine-tuning method (Full / LoRA / QLoRA)
- [ ] **FTCORE-03**: User can select optimizer (AdamW, SGD, 8-bit Adam, Adafactor)
- [ ] **FTCORE-04**: User can see training memory breakdown (weights, optimizer states, gradients, activations, overhead)
- [ ] **FTCORE-05**: User can configure LoRA rank (4-256) and alpha
- [ ] **FTCORE-06**: User can configure LoRA target modules percentage (10-100%)
- [ ] **FTCORE-07**: User can toggle mixed precision training (FP32/FP16/BF16)

### Memory Optimization

- [ ] **OPTIM-01**: User can set gradient accumulation steps (1-128) and see effective batch size
- [ ] **OPTIM-02**: User can toggle gradient checkpointing and see memory impact (50-80% activation reduction)
- [ ] **OPTIM-03**: User can toggle Flash Attention and see KV cache reduction
- [ ] **OPTIM-04**: User can see effective batch size calculation (batch x accumulation steps x GPUs)

### Framework Presets

- [ ] **FWPRST-01**: User can select framework preset (DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI, None)
- [ ] **FWPRST-02**: Framework preset auto-configures optimization settings with memory impact shown
- [ ] **FWPRST-03**: User can toggle 8-bit optimizer (2x memory savings)
- [ ] **FWPRST-04**: QLoRA mode shows 4-bit base + 16-bit adapter memory split

### Multi-GPU Training

- [ ] **MGPUTR-01**: User can estimate multi-GPU training VRAM with DeepSpeed ZeRO stages
- [ ] **MGPUTR-02**: ZeRO stages show correct memory reduction (Stage 1: optimizer, Stage 2: +gradients, Stage 3: +weights)
- [ ] **MGPUTR-03**: User can enable CPU offloading for optimizer states in training mode

## v1.2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Performance

- **PERF-01**: User can see training throughput estimation (samples/sec)
- **PERF-02**: User can see training time estimation per epoch

### Advanced Training

- **ADVTR-01**: User can select FSDP as distributed training strategy
- **ADVTR-02**: User can configure MoE-specific fine-tuning (expert selection)
- **ADVTR-03**: User can see cost estimation per training run

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Model execution/inference | Calculator only — link to vLLM, Ollama, TGI |
| Hyperparameter tuning suggestions | Out of scope — domain-specific, not VRAM estimation |
| Dataset size recommendations | Too training-task specific |
| Training loss prediction | Impossible without knowing data quality |
| Learning rate scheduling | Unrelated to VRAM |
| Cost estimation per epoch | Too many variables (provider, region, spot pricing) |
| FSDP / Megatron strategies | Overwhelming complexity for v1.1 — start with DeepSpeed ZeRO only |
| MoE-specific fine-tuning | Niche, complex, low ROI for v1.1 |
| Data loading/preprocessing memory | Highly variable, framework-dependent |
| Community benchmark submission | Requires backend + moderation — out of scope for static site |
| Mobile native app | Responsive web sufficient |
| AI-powered recommendations | Rule-based suggestions are clearer and more predictable |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FTCORE-01 | — | Pending |
| FTCORE-02 | — | Pending |
| FTCORE-03 | — | Pending |
| FTCORE-04 | — | Pending |
| FTCORE-05 | — | Pending |
| FTCORE-06 | — | Pending |
| FTCORE-07 | — | Pending |
| OPTIM-01 | — | Pending |
| OPTIM-02 | — | Pending |
| OPTIM-03 | — | Pending |
| OPTIM-04 | — | Pending |
| FWPRST-01 | — | Pending |
| FWPRST-02 | — | Pending |
| FWPRST-03 | — | Pending |
| FWPRST-04 | — | Pending |
| MGPUTR-01 | — | Pending |
| MGPUTR-02 | — | Pending |
| MGPUTR-03 | — | Pending |

**Coverage:**
- v1.1 requirements: 18 total
- Mapped to phases: 0
- Unmapped: 18

---
*Requirements defined: 2026-02-09*
*Last updated: 2026-02-10 after v1.1 milestone requirements*
