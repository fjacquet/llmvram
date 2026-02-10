# Feature Landscape: Fine-Tuning VRAM Estimation

**Domain:** LLM Fine-Tuning VRAM Calculator
**Milestone:** v1.1 - Adding fine-tuning estimation to existing inference calculator
**Researched:** 2026-02-10

## Executive Summary

This research focuses on features needed for v1.1, which adds fine-tuning VRAM estimation to the existing inference calculator. Fine-tuning requires estimating 4-8x more memory than inference due to optimizer states (8 bytes/param for AdamW), gradients (2-4 bytes/param), and activations that scale with batch size. Key differentiators are LoRA/QLoRA support (1-4% trainable parameters), gradient accumulation calculator, and framework presets (DeepSpeed ZeRO, Unsloth, vLLM/TGI).

## Table Stakes

Features users expect for fine-tuning estimation. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Dependencies | Notes |
|---------|--------------|------------|--------------|-------|
| Training mode toggle | Distinguish inference vs training | Low | None (new top-level control) | Simple radio: Inference / Fine-tuning |
| Fine-tuning method selector | Full/LoRA/QLoRA are standard methods | Low | Training mode active | Radio group: Full / LoRA / QLoRA |
| Optimizer selection | Different optimizers have vastly different memory | Medium | Training mode active | AdamW (8B/param), SGD (4B/param), 8-bit Adam (2B/param) |
| Batch size input | Critical for activation memory | Low | **Exists in v1.0** for inference | Reuse existing, expand validation range (1-256) |
| Sequence length input | Activation memory scales with seq len | Low | **Exists in v1.0** for inference | Reuse existing (512-131K range) |
| Training memory breakdown | Show WHERE memory goes in training | Medium | Memory visualization from v1.0 | Add: optimizer states, gradients to existing chart |
| LoRA rank/alpha inputs | Control LoRA adapter size | Low | Fine-tuning method = LoRA/QLoRA | Rank (4-256), alpha (2x rank typical) |
| Gradient accumulation steps | Standard memory-saving technique | Low | Training mode active | Input: 1-128 steps, show effective batch size |
| Mixed precision toggle | FP16/BF16 halves memory | Low | Training mode active | Checkbox + format selector (FP32/FP16/BF16) |

**Rationale:** Users cannot estimate training memory without knowing: (1) training method (full vs LoRA), (2) optimizer (AdamW vs 8-bit), (3) batch size, (4) whether gradient accumulation is used. These are non-negotiable for accurate estimation.

## Differentiators

Features that set product apart. Not expected, but highly valued.

| Feature | Value Proposition | Complexity | Dependencies | Notes |
|---------|-------------------|------------|--------------|-------|
| Framework presets | One-click accurate configs | Medium | Training calculations | DeepSpeed ZeRO-1/2/3, Unsloth, vLLM, TGI presets |
| Gradient checkpointing toggle | 50-80% activation memory reduction | Medium | Training calculations | Checkbox with memory impact shown |
| Flash Attention toggle | 50-80% KV cache reduction | Low | **KV cache engine from v1.0** | Checkbox, reduce KV by 0.5-0.8x |
| Effective batch size calculator | Understand gradient accumulation impact | Low | Gradient accumulation input | Formula: batch × accum_steps × num_gpus |
| LoRA target modules % | Adjust trainable param count | Low | LoRA method selected | Slider: 10-100% (default 50%, affects trainable params) |
| QLoRA base quantization | Show 4-bit base + 16-bit adapter split | Medium | QLoRA method, quantization engine from v1.0 | Auto-select 4-bit for base, show adapter separately |
| Multi-GPU training estimation | DeepSpeed ZeRO stage impact | High | Multi-GPU engine from v1.0 | ZeRO-1: partition optimizer, ZeRO-2: +gradients, ZeRO-3: +weights |
| Training speed estimation | Samples/sec with optimizations | High | Performance engine from v1.0 | Extend existing tokens/sec to training throughput |
| Optimizer state breakdown | Visualize momentum + variance | Low | Training memory breakdown | Sub-rows: first moment, second moment for AdamW |
| CPU offloading for training | DeepSpeed ZeRO-Offload / ZeRO-Infinity | Medium | Offloading engine from v1.0 | Extend existing CPU offload to optimizer states |
| 8-bit optimizer toggle | bitsandbytes paged optimizers | Low | Optimizer selection | Checkbox: "Use 8-bit optimizer" (2x memory savings) |
| Activation checkpointing detail | Show recomputation tradeoff | Low | Gradient checkpointing | Tooltip: "~20% slower, 50-80% memory savings" |

**Rationale:** Framework presets are the killer feature - users don't know optimal DeepSpeed ZeRO stage or Unsloth settings. Gradient checkpointing is table stakes in some communities, differentiator in others. Multi-GPU training with ZeRO is complex but extremely valuable for large models.

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Hyperparameter tuning suggestions | Out of scope, domain-specific | Show memory only, link to Weights & Biases guides |
| Dataset size recommendations | Too training-task specific | Provide batch size input, user decides iterations |
| Training loss prediction | Impossible without knowing data quality | Estimate memory/speed only, not model quality |
| Learning rate scheduling | Unrelated to VRAM | Focus on memory estimation |
| Cost estimation per epoch | Too many variables (provider, region, spot) | Maybe add in future, but not v1.1 priority |
| Distributed training strategies (FSDP, Megatron) | Overwhelming complexity for v1.1 | Start with DeepSpeed ZeRO only, expand later |
| Data loading/preprocessing memory | Highly variable, framework-dependent | Focus on model training memory |
| Evaluation during training | Adds another dimension of complexity | Assume training-only memory in v1.1 |
| Checkpoint saving memory | Transient, not sustained memory use | Ignore for VRAM estimation |
| Mixed expert fine-tuning (MoE-specific) | Niche, complex, low ROI for v1.1 | Support MoE inference only (already in v1.0) |

**Rationale:** v1.1 scope is VRAM estimation for training. Hyperparameters, data quality, cost optimization are separate problem domains. Keep focus tight on "will it fit, how fast will it train?"

## Feature Dependencies

Dependencies on existing v1.0 features:

```
v1.0 Existing Features (Reuse):
  ├─ Model Database → Reuse for parameter counts
  ├─ GPU Database → Reuse for VRAM capacity
  ├─ Quantization Engine → Extend to QLoRA (4-bit base)
  ├─ KV Cache Engine → Apply Flash Attention reduction
  ├─ Multi-GPU Engine → Extend to DeepSpeed ZeRO stages
  ├─ Offloading Engine → Extend to optimizer state offloading
  ├─ Performance Engine → Extend to training throughput
  ├─ Memory Breakdown Chart → Add optimizer/gradient/activation rows
  ├─ Batch Size Input → Reuse, expand validation range
  ├─ Sequence Length Input → Reuse for activation calculation
  └─ URL Serialization → Extend to include training params

v1.1 New Calculation Components:
  Fine-Tuning Engine (NEW)
    ├─ Full Fine-Tuning Mode
    │   ├─ Optimizer States (AdamW: 8B, SGD: 4B, 8-bit: 2B)
    │   ├─ Gradients (match weight precision)
    │   └─ Activations (batch × seq × hidden × layers)
    ├─ LoRA Mode
    │   ├─ Frozen base weights (quantized)
    │   ├─ LoRA adapters (rank × 2 × target_modules)
    │   ├─ Optimizer states (adapters only, ~1% of full)
    │   └─ Gradients (adapters only)
    └─ QLoRA Mode
        ├─ 4-bit quantized base (via quantization engine)
        ├─ 16-bit LoRA adapters
        └─ Optimizer/gradients (adapters only)

  Gradient Accumulation Calculator (NEW)
    └─ Effective batch size = batch × accum_steps × num_gpus

  Framework Preset Engine (NEW)
    ├─ DeepSpeed ZeRO Presets (1/2/3)
    ├─ Unsloth Preset (LoRA optimized)
    ├─ vLLM Preset (inference only, disable training)
    └─ TGI Preset (inference only, disable training)
```

## MVP Recommendation for v1.1

**Must-Have (Release Blockers):**

1. **Training mode toggle + method selector** (Full/LoRA/QLoRA)
2. **Optimizer selection** (AdamW, SGD, 8-bit Adam)
3. **Full fine-tuning calculation**
   - Optimizer states: `num_params × bytes_per_optimizer_format`
   - Gradients: `num_params × bytes_per_weight_format`
   - Activations: `2 × seq_len × batch × hidden × layers × 4 bytes` (FP32 default)
4. **LoRA calculation**
   - Base weights: Use existing quantization engine
   - LoRA adapters: `2 × rank × num_params × target_modules_pct × bytes_per_adapter_format`
   - Optimizer states: Apply only to adapters (~1% of params)
5. **QLoRA calculation**
   - Base weights: Force 4-bit quantization
   - LoRA adapters: Force FP16/BF16
   - Optimizer states: Apply only to adapters
6. **Training memory breakdown visualization**
   - Extend existing donut chart with: optimizer states, gradients, activations (expanded)
7. **Gradient accumulation input + effective batch size display**
8. **Mixed precision toggle** (FP16/BF16 reduces activations/gradients to 2B)

**Should-Have (High Value, Medium Effort):**

1. **Framework presets** (at least 3)
   - DeepSpeed ZeRO-2 (most common)
   - Unsloth (LoRA optimized, 70% memory reduction)
   - Basic preset (no optimizations)
2. **Gradient checkpointing toggle** (50-80% activation reduction)
3. **Flash Attention toggle** (50-80% KV cache reduction)
4. **LoRA rank/alpha inputs** (customize adapter size)

**Could-Have (Nice to Have, Defer if Tight):**

1. **8-bit optimizer toggle** (2x memory savings)
2. **Multi-GPU training with ZeRO** (stage 1/2/3 presets)
3. **Training speed estimation** (extend performance engine)
4. **LoRA target modules %** (advanced LoRA tuning)

**Won't-Have in v1.1:**

- CPU offloading for training (defer to v1.2)
- Training cost estimation (defer to v1.2)
- FSDP/Megatron strategies (DeepSpeed only for v1.1)
- Evaluation memory (training-only focus)

## Implementation Complexity

| Feature | Complexity | Effort | Notes |
|---------|------------|--------|-------|
| Training mode toggle | Low | 0.5 day | UI state management |
| Fine-tuning method selector | Low | 0.5 day | Radio group + conditional rendering |
| Optimizer selection | Low | 0.5 day | Dropdown with 3-4 options |
| Full fine-tuning calculation | Medium | 2 days | New engine, formula verification critical |
| LoRA calculation | Medium | 2 days | Adapter size math, percent trainable |
| QLoRA calculation | Medium | 1.5 days | Combine quantization + LoRA engines |
| Training memory breakdown | Medium | 2 days | Extend existing chart, new data structure |
| Gradient accumulation calculator | Low | 0.5 day | Simple formula, display effective batch |
| Mixed precision toggle | Low | 1 day | Adjust bytes/param in calculations |
| Framework presets | Medium | 2 days | JSON configs, preset application logic |
| Gradient checkpointing | Low | 1 day | Activation reduction multiplier (0.2-0.5x) |
| Flash Attention toggle | Low | 0.5 day | Reuse v1.0 KV cache engine, apply multiplier |
| LoRA rank/alpha inputs | Low | 1 day | Input validation, adapter size calculation |
| 8-bit optimizer toggle | Low | 0.5 day | Optimizer state bytes/param adjustment |
| Multi-GPU ZeRO | High | 4 days | Complex per-stage memory partitioning |
| Training speed estimation | High | 3 days | Benchmarking data, formula validation |

**Total MVP Effort:** ~12-15 days (must-have + should-have)

## Formulas for Implementation

### Full Fine-Tuning VRAM

```
VRAM_training = weights + kv_cache + activations + optimizer_states + gradients + overhead

weights = num_params × bytes_per_weight_format  // From v1.0 quantization engine
kv_cache = 2 × layers × hidden × seq_len × batch × kv_precision × gqa_ratio / 1e9  // From v1.0
activations = 2 × seq_len × batch × hidden × layers × 4 / 1e9  // Always FP32 (4 bytes)
optimizer_states = num_params × bytes_per_optimizer_format / 1e9
  - AdamW: 8 bytes (momentum + variance, both FP32)
  - SGD: 4 bytes (momentum only, FP32)
  - 8-bit Adam: 2 bytes (quantized states)
gradients = num_params × bytes_per_weight_format / 1e9  // Match weight precision
overhead = 1.0 + (0.2 × training_framework_multiplier)  // 20% training overhead minimum
```

### LoRA Fine-Tuning VRAM

```
trainable_params = num_params × target_modules_pct / 100
adapter_params = 2 × rank × trainable_params  // Two matrices A and B

VRAM_lora = base_weights + adapters + kv_cache + activations + optimizer_states + gradients + overhead

base_weights = num_params × bytes_per_quantization / 1e9  // Frozen, can be quantized
adapters = adapter_params × bytes_per_adapter_format / 1e9  // Usually FP16 (2 bytes)
kv_cache = [same as full fine-tuning]
activations = [same as full fine-tuning, but scales with batch size]
optimizer_states = adapter_params × bytes_per_optimizer_format / 1e9  // Only for adapters!
gradients = adapter_params × bytes_per_adapter_format / 1e9  // Only for adapters!
overhead = 1.0 + (0.15 × training_framework_multiplier)  // 15% LoRA overhead
```

**Typical LoRA values:**

- Rank: 4-16 (low), 32-64 (medium), 128-256 (high)
- Alpha: Usually 2× rank (e.g., rank=16, alpha=32)
- Target modules: 50% default (attention layers only) to 100% (all linear layers)
- Trainable params: ~1-4% of total parameters

### QLoRA Fine-Tuning VRAM

```
VRAM_qlora = base_weights_4bit + adapters_fp16 + kv_cache + activations + optimizer_states + gradients + overhead

base_weights_4bit = num_params × 0.5 × 1.2 / 1e9  // 4-bit with overhead (NF4 typical)
adapters_fp16 = adapter_params × 2 / 1e9  // Always FP16 or BF16
[kv_cache, activations, optimizer_states, gradients same as LoRA]
overhead = 1.0 + (0.2 × training_framework_multiplier)  // 20% QLoRA overhead
```

**Memory savings:**

- QLoRA uses ~4x less memory than LoRA for the base model
- Total VRAM for QLoRA typically ~40-60% of LoRA for same model

### Gradient Accumulation

```
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus

// Activation memory PER STEP (not accumulated):
activations = 2 × seq_len × batch_size × hidden × layers × 4 / 1e9

// Gradient accumulation does NOT multiply activation memory
// It only increases effective batch size for optimizer update frequency
```

**Important:** Gradient accumulation does NOT increase activation memory proportionally because activations are computed per micro-batch, then discarded after backward pass. Only gradients are accumulated (already counted in `gradients` term).

### Gradient Checkpointing

```
// With gradient checkpointing enabled:
activations_checkpointed = activations × 0.2  // 80% reduction typical
// Trade-off: ~20-30% slower training (recompute on backward)
```

### Flash Attention

```
// With Flash Attention enabled:
kv_cache_flash = kv_cache × 0.5  // 50% reduction typical (can be 20-80% depending on seq_len)
// No speed penalty, actually faster due to IO optimization
```

### DeepSpeed ZeRO Stages (Multi-GPU)

```
// ZeRO Stage 1: Partition optimizer states only
per_gpu_memory = weights + kv_cache + activations + gradients + (optimizer_states / num_gpus) + overhead

// ZeRO Stage 2: Partition optimizer states + gradients
per_gpu_memory = weights + kv_cache + activations + (gradients / num_gpus) + (optimizer_states / num_gpus) + overhead

// ZeRO Stage 3: Partition optimizer states + gradients + weights
per_gpu_memory = (weights / num_gpus) + (kv_cache / num_gpus) + activations + (gradients / num_gpus) + (optimizer_states / num_gpus) + overhead

// Note: ZeRO-3 has highest communication overhead (~25%), ZeRO-2 is balanced (~15%), ZeRO-1 is lowest (~10%)
```

### Mixed Precision Training

```
// FP16/BF16 mixed precision (most common):
weights = num_params × 2 / 1e9  // FP16 working copy
master_weights = num_params × 4 / 1e9  // FP32 master copy for optimizer
optimizer_states = num_params × bytes_per_optimizer / 1e9  // FP32 states
gradients = num_params × 2 / 1e9  // FP16 gradients
activations = 2 × seq_len × batch × hidden × layers × 2 / 1e9  // FP16 activations

VRAM_mixed = weights + master_weights + optimizer_states + gradients + activations + kv_cache + overhead

// Memory savings vs FP32: ~40-50% (not 50% because optimizer maintains FP32 states)
```

## Framework Presets Configuration

### Preset 1: DeepSpeed ZeRO-2 (Recommended Default)

**When to use:** Multi-GPU training (2-8 GPUs), balanced performance/memory

**Configuration:**

- Optimizer: AdamW 32-bit
- Mixed precision: BF16
- Gradient checkpointing: Enabled
- Flash Attention: Enabled (if supported)
- ZeRO stage: 2 (partition optimizer + gradients)
- Gradient accumulation: 4 steps (default)

**Memory impact:**

- Optimizer states: Distributed across GPUs
- Gradients: Distributed across GPUs
- Weights: Replicated
- Expected reduction: ~60% per-GPU memory vs single GPU

**Use case:** Most common production training setup

### Preset 2: Unsloth (LoRA Optimized)

**When to use:** LoRA/QLoRA training, single GPU, memory constrained

**Configuration:**

- Fine-tuning method: QLoRA
- Base quantization: 4-bit NF4
- Adapter format: BF16
- Optimizer: 8-bit AdamW (paged)
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: 50% (attention only)
- Gradient checkpointing: Unsloth optimized (30% reduction vs 80% standard)
- Flash Attention: Enabled

**Memory impact:**

- 70% memory reduction vs standard LoRA (per Unsloth benchmarks)
- 2x faster training vs standard LoRA
- Supports 9B models on 24GB VRAM (LoRA 16-bit) or 6.5GB VRAM (QLoRA 4-bit)

**Use case:** Consumer hardware, single GPU fine-tuning

### Preset 3: DeepSpeed ZeRO-3 (Maximum Memory Savings)

**When to use:** Very large models (70B+), many GPUs (4-8), memory critical

**Configuration:**

- Optimizer: AdamW 32-bit
- Mixed precision: BF16
- Gradient checkpointing: Enabled
- Flash Attention: Enabled
- ZeRO stage: 3 (partition optimizer + gradients + weights)
- ZeRO-Infinity: CPU offload (optional)
- Gradient accumulation: 8 steps

**Memory impact:**

- Everything distributed across GPUs
- Expected reduction: ~90% per-GPU memory vs single GPU
- Trade-off: ~25% slower due to communication overhead

**Use case:** 70B+ models, insufficient VRAM even with multi-GPU

### Preset 4: vLLM (Inference Only)

**When to use:** Deployment, serving, inference optimization

**Configuration:**

- Mode: Inference (disable training)
- Quantization: AWQ 4-bit (recommended)
- KV cache quantization: FP8
- Flash Attention: Enabled (PagedAttention)
- Continuous batching: Enabled

**Memory impact:**

- 50-75% memory reduction vs naive inference
- 14-24x throughput vs Hugging Face Transformers

**Use case:** Production serving, high throughput inference

### Preset 5: TGI (Inference Only)

**When to use:** Hugging Face ecosystem, inference serving

**Configuration:**

- Mode: Inference (disable training)
- Quantization: GPTQ 4-bit or bitsandbytes
- KV cache quantization: FP8
- Flash Attention: Enabled (PagedAttention)
- Tensor parallelism: Enabled (multi-GPU)

**Memory impact:**

- Similar to vLLM, ~50-75% reduction

**Use case:** Hugging Face integration, multi-framework support

### Preset 6: Basic (No Optimizations)

**When to use:** Debugging, baseline comparison, research

**Configuration:**

- Optimizer: AdamW 32-bit
- Mixed precision: Disabled (FP32)
- Gradient checkpointing: Disabled
- Flash Attention: Disabled
- Gradient accumulation: 1 (no accumulation)

**Memory impact:**

- Maximum memory usage (baseline)
- Useful for understanding optimization impact

**Use case:** Research, debugging, academic settings

## User Experience Considerations

### Progressive Disclosure

**Complexity levels:**

1. **Simple mode (default):** Training method (Full/LoRA/QLoRA) + Framework preset dropdown
   - Preset auto-configures optimizer, precision, checkpointing
   - 80% of users stop here
2. **Advanced mode (expandable):** Override preset settings
   - Optimizer selection
   - Gradient checkpointing toggle
   - Flash Attention toggle
   - Gradient accumulation steps
   - LoRA rank/alpha (if LoRA selected)
3. **Expert mode (separate tab?):** Full control
   - 8-bit optimizer
   - Mixed precision configuration
   - Multi-GPU ZeRO stage
   - CPU offloading

**Rationale:** Most users want "I'm using Unsloth" or "I'm using DeepSpeed ZeRO-2", not individual knobs. Power users need override capability.

### Input Validation

**Critical validations:**

- Batch size × seq_len × layers must not exceed activation memory safety threshold
- LoRA rank must be ≤ hidden_size
- Gradient accumulation steps: warn if > 32 (diminishing returns, staleness issues)
- Multi-GPU: ZeRO-3 requires ≥ 2 GPUs
- QLoRA: force 4-bit base quantization (user cannot change)
- 8-bit optimizer: only available with AdamW/Adam (not SGD)

### Result Display Enhancements

**Training-specific displays:**

1. **Training feasibility indicator:**
   - "✓ Fits in VRAM" (< 90% utilization)
   - "⚠ Tight fit" (90-95% utilization)
   - "✗ Insufficient VRAM" (> 95% utilization)
2. **Effective batch size display:**
   - `effective_batch_size = batch × accum_steps × num_gpus`
   - "Training with effective batch size of 64 (16 × 4 accum × 1 GPU)"
3. **Memory savings display:**
   - "LoRA reduces trainable parameters by 98.5% (145M → 2.1M)"
   - "Gradient checkpointing saves 12.3 GB (62% reduction in activations)"
4. **Training throughput estimate:**
   - "~450 tokens/sec" (existing performance engine)
   - "~28 samples/min (512 tokens/sample, batch=16)"

## Testing Strategy

**Critical test cases:**

1. **Full fine-tuning:** 7B model, AdamW, FP16, batch=16, seq=2048
   - Expected: ~112 GB VRAM (16 GB weights + 64 GB optimizer + 14 GB gradients + 18 GB activations)
2. **LoRA:** Same model, rank=16, 50% target modules
   - Expected: ~20-25 GB VRAM (4-6x reduction vs full)
3. **QLoRA:** Same model, 4-bit base, rank=16
   - Expected: ~8-10 GB VRAM (4x reduction vs LoRA)
4. **Gradient checkpointing:** Toggle on/off, verify 50-80% activation reduction
5. **Gradient accumulation:** Verify activation memory does NOT multiply with steps
6. **DeepSpeed ZeRO-2:** 4 GPUs, verify optimizer + gradients distributed
7. **Mixed precision:** FP16 vs FP32, verify ~40-50% reduction
8. **Edge cases:**
   - Batch size = 1 (minimum activations)
   - Seq length = 131K (maximum activations)
   - LoRA rank = 256 (large adapters approach full fine-tuning memory)

**Validation against benchmarks:**

- Compare calculations to Unsloth documentation (9B LoRA = 24 GB stated)
- Compare to Modal.com guide (7B full = ~56 GB stated, but varies by framework)
- Compare to DeepSpeed memory estimator outputs

## Sources

### HIGH Confidence (Official Documentation & Recent Research)

- [Modal.com: How much VRAM do I need for LLM model fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning) - Formula verification for full fine-tuning with AdamW
- [Orbit2x: Ultimate VRAM Calculator Guide 2026](https://orbit2x.com/blog/ultimate-vram-calculator-guide-gpu-memory-ai-models) - Comprehensive VRAM estimation formulas with 2026 updates
- [Medium: Calculating GPU Memory for LLM Fine-Tuning](https://medium.com/@imsanjoykb/calculating-gpu-memory-for-large-language-model-fine-tuning-a-practical-approach-91b7ea883516) - Practical approach with examples
- [Modal.com: LoRA vs QLoRA](https://modal.com/blog/lora-qlora) - Memory comparison and use cases
- [Runpod: Maximizing Efficiency with LoRA and QLoRA](https://www.runpod.io/articles/guides/maximizing-efficiency-fine-tuning-large-language-models-with-lora-and-qlora-on-runpod) - Concrete VRAM numbers for 7B/70B models
- [Unsloth Documentation: LoRA Hyperparameters](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) - Rank/alpha recommendations
- [GitHub: unslothai/unsloth](https://github.com/unslothai/unsloth) - 2x faster, 70% less VRAM claims
- [DeepSpeed Documentation: ZeRO](https://www.deepspeed.ai/tutorials/zero/) - Official ZeRO stage explanations
- [DeepSpeed Docs: Memory Requirements](https://deepspeed.readthedocs.io/en/latest/memory.html) - Per-stage memory formulas
- [PyTorch Lightning: DeepSpeed](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html) - Practical ZeRO usage

### MEDIUM Confidence (Community Best Practices)

- [Unsloth Blog: Bug Fixes in LLM Training - Gradient Accumulation](https://unsloth.ai/blog/gradient) - Gradient accumulation misconceptions
- [Lightning AI: Finetuning LLMs with Gradient Accumulation](https://lightning.ai/pages/blog/gradient-accumulation/) - Memory tradeoff analysis
- [Giles Thomas: Messing around with fine-tuning LLMs](https://www.gilesthomas.com/2024/07/fine-tuning-5) - Real-world memory usage exploration
- [vLLM GitHub](https://github.com/vllm-project/vllm) - PagedAttention memory management
- [Hugging Face: Text Generation Inference](https://huggingface.co/docs/text-generation-inference/en/index) - TGI optimization techniques
- [DigitalOcean: vLLM GPU Sizing](https://www.digitalocean.com/community/conceptual-articles/vllm-gpu-sizing-configuration-guide) - Quantization and parallelism strategies
- [Runpod: How can I fine-tune LLMs on a budget](https://www.runpod.io/articles/guides/how-to-fine-tune-large-language-models-on-a-budget) - Budget-friendly configurations
- [Kaitchup: TrainingArguments adamw_8bit](https://kaitchup.substack.com/p/fine-tuning-llms-with-32-bit-8-bit) - 8-bit optimizer comparison

### MEDIUM Confidence (Technical Papers & Benchmarks)

- [arXiv: 8-bit optimizers via block-wise quantization](https://arxiv.org/pdf/2110.02861) - bitsandbytes 8-bit Adam research
- [arXiv: Small Batch Size Training for Language Models](https://arxiv.org/pdf/2507.07101) - Gradient accumulation research (2025)
- [arXiv: FlashAttention](https://arxiv.org/abs/2205.14135) - 10-20x memory savings at 2K-4K seq lengths
- [Hazy Research: FlashAttention-2](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2) - 2x faster than FA1, 72% FLOP utilization
- [Runpod: LLM Fine-Tuning GPU Guide](https://www.runpod.io/blog/llm-fine-tuning-gpu-guide) - Practical GPU requirements

### LOW Confidence (Needs Validation)

- Training speed estimation formulas (high variance across frameworks, no single source)
- CPU offloading performance penalties (ZeRO-Offload stated as "10-30x slower" in one source, needs verification)
- Exact gradient checkpointing overhead percentages (sources range 20-30%, varies by model architecture)
- Multi-GPU communication overhead percentages (estimates vary 10-25% depending on interconnect)

### Gaps to Address

- No official DeepSpeed ZeRO-3 memory calculator found (formulas inferred from documentation)
- Unsloth specific optimizations are proprietary (claims validated against user reports, not source code)
- Training throughput estimation requires benchmarking data (not available in web search)
- MoE fine-tuning specifics (expert routing overhead during training) - deferred to future research

## Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| Full fine-tuning formulas | **HIGH** | Multiple authoritative sources agree on 8B/param for AdamW, 4B for SGD |
| LoRA/QLoRA memory savings | **HIGH** | Official docs and benchmarks consistent (LoRA: 4-6x, QLoRA: 12-16x reduction) |
| Gradient accumulation behavior | **HIGH** | Recent research (2025) clarifies misconceptions, activation memory does NOT multiply |
| DeepSpeed ZeRO stages | **HIGH** | Official documentation with formulas |
| Framework preset configurations | **MEDIUM** | Inferred from best practices, not always explicitly documented |
| Training throughput estimation | **LOW** | Requires benchmarking data not available from web search |
| CPU offloading penalties | **LOW** | Wide range of claims, hardware-dependent |
