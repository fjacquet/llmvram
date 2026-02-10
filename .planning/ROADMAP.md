# Roadmap: LLM VRAM Calculator

## Milestones

- ✅ **v1.0 MVP** — Phases 1-5 (shipped 2026-02-09)
- 🚧 **v1.1 Fine-Tuning Estimation** — Phases 6-10 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-5) — SHIPPED 2026-02-09</summary>

- [x] Phase 1: Foundation & Data (4/4 plans) — completed 2026-02-09
- [x] Phase 2: Inference Engine (4/4 plans) — completed 2026-02-09
- [x] Phase 3: Core UI (4/4 plans) — completed 2026-02-09
- [x] Phase 4: Multi-GPU Support (3/3 plans) — completed 2026-02-09
- [x] Phase 5: Sharing & Comparison (3/3 plans) — completed 2026-02-09

Full details: [milestones/v1.0-ROADMAP.md](milestones/v1.0-ROADMAP.md)

</details>

### 🚧 v1.1 Fine-Tuning Estimation (In Progress)

**Milestone Goal:** Enable users to estimate VRAM requirements for fine-tuning LLMs, including LoRA/QLoRA adapters, optimizer states, gradients, gradient accumulation, and optimization framework presets.

**Phase Numbering:**
- Integer phases (6, 7, 8, 9, 10): Planned milestone work
- Decimal phases (6.1, 6.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 6: Fine-Tuning Calculation Engines** - Pure calculation logic for full/LoRA/QLoRA training VRAM
- [ ] **Phase 7: Training State & Basic UI** - Training mode toggle and configuration state management
- [ ] **Phase 8: Memory Optimization Features** - Gradient accumulation, checkpointing, Flash Attention
- [ ] **Phase 9: Training Memory Visualization** - Extended charts and tables with training breakdown
- [ ] **Phase 10: Framework Presets & Multi-GPU Training** - Framework-specific optimizations and distributed training

## Phase Details

### Phase 6: Fine-Tuning Calculation Engines
**Goal:** Users can calculate accurate training VRAM for full/LoRA/QLoRA methods
**Depends on:** Nothing (builds on existing inference engines)
**Requirements:** FTCORE-02, FTCORE-03, FTCORE-05, FTCORE-06
**Success Criteria** (what must be TRUE):
  1. User can calculate full fine-tuning VRAM (weights + gradients + optimizer states + activations)
  2. User can calculate LoRA fine-tuning VRAM (frozen base + adapters with correct optimizer state sizing)
  3. User can calculate QLoRA fine-tuning VRAM (4-bit base + FP16 adapters)
  4. Optimizer memory calculation accounts for FP32 precision regardless of training precision
  5. LoRA adapter parameter count scales correctly with rank, alpha, and target modules
**Plans:** TBD

Plans:
- [ ] TBD (to be defined during phase planning)

### Phase 7: Training State & Basic UI
**Goal:** Users can toggle into training mode and select basic configuration
**Depends on:** Phase 6
**Requirements:** FTCORE-01, FTCORE-07
**Success Criteria** (what must be TRUE):
  1. User can switch between Inference and Fine-tuning modes
  2. User can select fine-tuning method (Full / LoRA / QLoRA)
  3. User can select optimizer type (AdamW, SGD, 8-bit Adam, Adafactor)
  4. User can toggle mixed precision training (FP32/FP16/BF16)
  5. Training configuration persists in URL hash for sharing
**Plans:** TBD

Plans:
- [ ] TBD (to be defined during phase planning)

### Phase 8: Memory Optimization Features
**Goal:** Users can optimize training memory through gradient accumulation and checkpointing
**Depends on:** Phase 7
**Requirements:** OPTIM-01, OPTIM-02, OPTIM-03, OPTIM-04
**Success Criteria** (what must be TRUE):
  1. User can set gradient accumulation steps and see effective batch size calculation
  2. User can toggle gradient checkpointing and see activation memory reduction (50-80%)
  3. User can toggle Flash Attention and see KV cache reduction for training
  4. Effective batch size calculation accounts for micro batch × accumulation steps × GPUs
**Plans:** TBD

Plans:
- [ ] TBD (to be defined during phase planning)

### Phase 9: Training Memory Visualization
**Goal:** Users can see detailed breakdown of training memory components
**Depends on:** Phase 6, 7, 8
**Requirements:** FTCORE-04
**Success Criteria** (what must be TRUE):
  1. User can see training memory breakdown chart with optimizer states, gradients, activations, weights
  2. User can see training memory breakdown table with trainable vs frozen parameters
  3. Memory breakdown updates in real-time as training configuration changes
  4. Fit indicator shows whether training configuration fits in available VRAM
**Plans:** TBD

Plans:
- [ ] TBD (to be defined during phase planning)

### Phase 10: Framework Presets & Multi-GPU Training
**Goal:** Users can use framework presets and estimate multi-GPU training VRAM
**Depends on:** Phase 6, 7, 8, 9
**Requirements:** FWPRST-01, FWPRST-02, FWPRST-03, FWPRST-04, MGPUTR-01, MGPUTR-02, MGPUTR-03
**Success Criteria** (what must be TRUE):
  1. User can select framework preset (DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI)
  2. Framework preset auto-applies optimization settings (gradient checkpointing, Flash Attention, 8-bit optimizer)
  3. Multi-GPU training estimation shows correct ZeRO stage memory reduction (2x/4x/8x)
  4. QLoRA mode displays separate memory for 4-bit base model and 16-bit adapters
  5. User can enable CPU offloading for optimizer states in training mode
**Plans:** TBD

Plans:
- [ ] TBD (to be defined during phase planning)

## Progress

**Execution Order:**
Phases execute in numeric order: 6 → 7 → 8 → 9 → 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation & Data | v1.0 | 4/4 | Complete | 2026-02-09 |
| 2. Inference Engine | v1.0 | 4/4 | Complete | 2026-02-09 |
| 3. Core UI | v1.0 | 4/4 | Complete | 2026-02-09 |
| 4. Multi-GPU Support | v1.0 | 3/3 | Complete | 2026-02-09 |
| 5. Sharing & Comparison | v1.0 | 3/3 | Complete | 2026-02-09 |
| 6. Fine-Tuning Calculation Engines | v1.1 | 0/TBD | Not started | - |
| 7. Training State & Basic UI | v1.1 | 0/TBD | Not started | - |
| 8. Memory Optimization Features | v1.1 | 0/TBD | Not started | - |
| 9. Training Memory Visualization | v1.1 | 0/TBD | Not started | - |
| 10. Framework Presets & Multi-GPU Training | v1.1 | 0/TBD | Not started | - |
