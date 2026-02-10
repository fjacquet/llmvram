# Domain Pitfalls: Adding Fine-Tuning VRAM Estimation

**Domain:** Fine-tuning VRAM calculation for LLM VRAM Calculator (v2.0 milestone)
**Researched:** 2026-02-10
**Confidence:** HIGH (verified with Context7, official docs, and current 2026 sources)

**Context:** This document covers pitfalls specific to ADDING fine-tuning estimation features to an existing inference VRAM calculator. For inference-only pitfalls, see the v1.0 PITFALLS.md.

---

## Critical Pitfalls

These mistakes cause estimation errors >50% and lead to OOM failures or massive overestimation.

### Pitfall 1: Reusing Inference KV Cache Formula for Training

**What goes wrong:** Calculator reuses inference KV cache formula (`2 * n_layers * d_model * seq_len * n_kv_heads/n_heads`) for fine-tuning mode. Reality: Training processes batches in parallel with full attention matrices, not sequential generation. KV cache behavior is fundamentally different.

**Why it happens:**
- Assuming KV cache works the same way in training
- Not understanding that training doesn't cache keys/values the same way
- Trying to maximize code reuse between inference and training

**Consequences:**
- Training memory underestimated by 30-60% for long sequences
- Missing attention matrix materialization during forward/backward pass
- Users hit OOM when "calculator said it would fit"

**Prevention:**

```
Training attention memory (without Flash Attention):
- Forward pass attention matrix: batch * n_heads * seq_len² * 2 bytes (BF16)
- Backward pass gradients: same size
- Total: 2 * batch * n_heads * seq_len² * 2 bytes

Training attention memory (with Flash Attention v2+):
- Forward/backward uses tiling, no full matrix materialization
- Memory: O(batch * seq_len) instead of O(batch * seq_len²)
- Reduction: 10-20x for long sequences (8K+)

Note: Training does NOT use KV cache like inference. Don't reuse that calculation.
```

**Detection:**
- Training estimate looks almost identical to inference estimate
- Missing Flash Attention toggle for training mode
- Estimates don't explode with long sequences (they should without Flash Attention)

**Phase mapping:** Phase 1 (Fine-tuning MVP) - Must separate training from inference logic immediately

**Sources:**
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Out-of-the-Box Acceleration](https://pytorch.org/blog/out-of-the-box-acceleration/)

---

### Pitfall 2: Optimizer State Precision Assumption

**What goes wrong:** Calculator assumes optimizer states match training precision (BF16 training = 2 bytes/param optimizer state). Reality: AdamW and most optimizers maintain states in FP32 (4 bytes) for numerical stability, even when training in BF16/FP16.

**Why it happens:**
- Logical assumption: "training in BF16 = everything in BF16"
- Not reading optimizer implementation details
- Copying formulas from blog posts that simplify for clarity

**Consequences:**
- 2x underestimation of optimizer memory
- Total training memory off by 30-40%
- "18 bytes per parameter" becomes 14, completely wrong

**Prevention:**

```
AdamW optimizer states (standard):
- First moment (momentum): params * 4 bytes (FP32)
- Second moment (variance): params * 4 bytes (FP32)
- Total: 8 bytes per TRAINABLE parameter

Total training memory per parameter (mixed precision):
- Model weights (BF16): 2 bytes
- Gradients (BF16): 2 bytes
- Master weights (FP32): 4 bytes (copy of weights in FP32)
- Optimizer states (FP32): 8 bytes
- Total: ~16-18 bytes per trainable parameter

Alternative optimizers:
- SGD with momentum: 4 bytes/param (1x FP32 state)
- AdamW 8-bit: ~2 bytes/param (quantized states)
- Lion: 4 bytes/param (1x FP32 state)
```

**Detection:**
- Optimizer state memory changes when user switches training precision
- Missing "master weights" in mixed precision explanation
- Total training memory < 12 bytes/param (impossibly low for AdamW)

**Phase mapping:** Phase 1 (Fine-tuning MVP) - Core calculation must be accurate

**Sources:**
- [Memory Requirements (HBM, GPU RAM)](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/memory-requirements-hbm-gpu-ram)
- [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one)
- [Modern Optimizers: AdamW, Lion](https://medium.com/@spjosyula2005/modern-optimizers-adamw-lion-and-what-actually-works-at-scale-68ffc033713b)

---

### Pitfall 3: LoRA Adapter-Only Optimizer States Ignored

**What goes wrong:** Calculator multiplies optimizer state memory by total model parameters for LoRA fine-tuning. Reality: Only LoRA adapter parameters (typically <1% of model) are trainable and get optimizer states. Frozen base model weights have no optimizer states.

**Why it happens:**
- Not distinguishing trainable vs frozen parameters
- Applying full fine-tuning formula to LoRA
- Missing that LoRA freezes base model completely

**Consequences:**
- 100-200x overestimation for LoRA fine-tuning
- Calculator shows "need 400GB" when 24GB is sufficient
- Users think fine-tuning is impossible on consumer hardware

**Prevention:**

```
LoRA trainable parameters:
- Adapter matrices per layer: 2 * rank * d_model
- Number of target layers: typically attention layers (q_proj, k_proj, v_proj, o_proj)
- Total adapters: ~0.2-2% of base model parameters

Example (Llama 7B, rank 16, 4 targets per layer, 32 layers):
- Base model: 7B params frozen
- Adapters: 2 * 16 * 4096 * 4 * 32 = 16.8M trainable params
- Ratio: 16.8M / 7B = 0.24%

Memory calculation:
- Base model: 7B * 2 bytes (BF16) = 14GB
- Adapter weights: 16.8M * 2 bytes = 33.6MB
- Adapter gradients: 16.8M * 2 bytes = 33.6MB
- Adapter optimizer states: 16.8M * 8 bytes = 134MB
- Activations: based on batch_size, seq_len, hidden_size
- Total: ~16-18GB (NOT 90GB like full fine-tuning)

CRITICAL: Only count trainable params for gradients + optimizer states!
```

**Detection:**
- LoRA memory estimate similar to full fine-tuning
- Missing LoRA rank parameter input
- Calculator doesn't ask "which layers to target with LoRA"
- Estimate doesn't change dramatically between rank 8 and rank 64

**Phase mapping:** Phase 1 (Fine-tuning MVP) - Essential for LoRA support

**Sources:**
- [LoRA fine-tuning Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Understanding LoRA Adapters Rank and Alpha Parameters](https://datawizz.ai/blog/understanding-lora-adapters-rank-and-alpha-parameters)

---

### Pitfall 4: QLoRA Precision Mixing Miscalculation

**What goes wrong:** Calculator treats QLoRA as "4-bit LoRA" with uniform precision. Reality: QLoRA mixes three precisions: 4-bit NF4 base model (frozen), FP16/BF16 adapter weights (trainable), FP32 optimizer states (for adapters only).

**Why it happens:**
- Thinking QLoRA is just "quantized LoRA"
- Not understanding the three-precision architecture
- Missing that base model quantization doesn't affect adapter memory

**Consequences:**
- Incorrect memory calculation (typically overestimation)
- Missing that 4-bit base saves ~75% on frozen weights
- Not accounting for dequantization overhead during forward pass

**Prevention:**

```
QLoRA memory breakdown (Llama 7B example):
1. Base model (4-bit NF4, frozen):
   - 7B * 0.5 bytes * 1.1 (overhead) = 3.85GB

2. Paged optimizer buffers:
   - Additional 500MB-1GB for CPU-GPU paging

3. LoRA adapters (BF16, trainable):
   - Same as regular LoRA: ~17M params * 2 bytes = 34MB

4. Adapter gradients (BF16):
   - 17M params * 2 bytes = 34MB

5. Adapter optimizer states (FP32):
   - 17M params * 8 bytes = 136MB

6. Activations (mixed precision):
   - Forward pass: batch * seq_len * hidden * layers * 2-4x
   - With gradient checkpointing: 60% reduction

Total: ~6-8GB (vs 90GB full fine-tuning, 16-18GB LoRA)

Key insight: 4-bit base model provides 75% memory savings on frozen weights,
adapters stay in FP16/BF16 for training stability, optimizer states still FP32.
```

**Detection:**
- QLoRA memory = LoRA memory / 4 (too simple)
- Missing "paged optimizer" explanation
- No mention of NF4 quantization specifically
- Estimate doesn't account for dequantization overhead

**Phase mapping:** Phase 2 (Advanced Fine-tuning) - After basic LoRA works

**Sources:**
- [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/artidoro/qlora)
- [Quantized LoRA (QLoRA) Principles](https://apxml.com/courses/lora-peft-efficient-llm-training/chapter-4-advanced-lora-variants/qlora-principles)

---

### Pitfall 5: Gradient Accumulation Peak Memory Misconception

**What goes wrong:** Calculator shows memory reduction when increasing gradient accumulation steps. Reality: Gradient accumulation reduces PER-STEP batch size (activation memory), but does NOT reduce peak gradient or optimizer state memory. Common misconception that it's a general memory optimization.

**Why it happens:**
- Blog posts say "gradient accumulation saves memory"
- Not distinguishing activation memory vs gradient memory
- Confusing effective batch size with peak memory

**Consequences:**
- Showing reduced memory estimate with more accumulation steps (WRONG)
- Users expect 8x accumulation = 8x memory savings (not true)
- Missing that accumulation is for batch size simulation, not memory reduction

**Prevention:**

```
Memory impact of gradient accumulation:

Micro-batch size: 1, Accumulation steps: 8, Effective batch: 8

REDUCED (scales with micro-batch):
- Activations: batch=1 activations, not batch=8
- Forward pass intermediate values: batch=1 only
- Savings: ~85% on activation memory

NOT REDUCED (constant regardless of accumulation):
- Model weights: same size
- Gradients: accumulated in-place, same memory
- Optimizer states: same size
- Peak memory: still need to store full gradients

Correct formula:
Peak memory = weights + gradients + optimizer_states + (micro_batch_activations)
NOT: Peak memory = weights + gradients + optimizer_states + (effective_batch_activations)

Example (Llama 7B, mixed precision):
- Batch=8, accum=1: 14GB (weights) + 14GB (grad) + 56GB (opt) + 12GB (act) = 96GB
- Batch=1, accum=8: 14GB (weights) + 14GB (grad) + 56GB (opt) + 1.5GB (act) = 85.5GB
- Savings: 10.5GB (11%), NOT 8x (87%)
```

**Detection:**
- Memory estimate scales linearly down with accumulation steps
- Explanation says "gradient accumulation saves memory" without caveats
- Users expect 8x accumulation to enable training on 1/8 the VRAM (impossible)

**Phase mapping:** Phase 1 (Fine-tuning MVP) - Critical for accurate guidance

**Sources:**
- [Gradient Accumulation: Increase Batch Size Without Explicitly Increasing Batch Size](https://blog.dailydoseofds.com/p/gradient-accumulation-increase-batch)
- [Gradient Accumulation and Checkpointing](https://aman.ai/primers/ai/grad-accum-checkpoint/)
- [Gradient Accumulation: Overcome GPU Memory Limitations](https://docs.vultr.com/how-to-use-gradient-accumulation-to-overcome-gpu-memory-limitations)

---

### Pitfall 6: DeepSpeed ZeRO Stage Memory Profile Confusion

**What goes wrong:** Calculator shows linear memory reduction with ZeRO stages (Stage 1 = 50%, Stage 2 = 33%, Stage 3 = 25% per GPU). Reality: Memory savings are 2x / 4x / 8-10x respectively, with different communication overhead and precision requirements.

**Why it happens:**
- Misunderstanding what each stage partitions
- Thinking "4 GPUs = divide by 4" for all stages
- Not accounting for communication volume differences

**Consequences:**
- Massive overestimation or underestimation of multi-GPU memory
- Missing that ZeRO-3 is fundamentally different from ZeRO-1/2
- Not warning about communication overhead (15-30% throughput loss)

**Prevention:**

```
DeepSpeed ZeRO stage memory profiles (per GPU):

Stage 1 (Optimizer State Partitioning):
- Model weights: replicated on all GPUs
- Gradients: replicated on all GPUs
- Optimizer states: partitioned (divided by N)
- Memory savings: ~2x total
- Communication: optimizer all-gather (minimal)
- Use case: Easy win with minimal complexity

Stage 2 (Optimizer + Gradient Partitioning):
- Model weights: replicated on all GPUs
- Gradients: partitioned (divided by N)
- Optimizer states: partitioned (divided by N)
- Memory savings: ~4x total
- Communication: gradient reduce-scatter + optimizer all-gather
- Use case: Standard for multi-GPU training

Stage 3 (Optimizer + Gradient + Parameter Partitioning):
- Model weights: partitioned (divided by N)
- Gradients: partitioned (divided by N)
- Optimizer states: partitioned (divided by N)
- Memory savings: ~8-10x total (near-linear scaling)
- Communication: all-gather params on-the-fly during forward/backward
- Overhead: 15-30% throughput reduction from parameter communication
- Use case: Training models that don't fit on N GPUs otherwise

Example (Llama 70B, mixed precision, 4x A100 80GB):
- Stage 1: ~320GB total / 4 = 80GB per GPU (optimizer only partitioned)
- Stage 2: ~320GB total / 4 = 80GB per GPU (grad + opt partitioned)
- Stage 3: ~320GB total / 8 = 40GB per GPU (everything partitioned)

CRITICAL: Don't use simple "divide by N" - savings are 2x/4x/8x, not N-way split!
```

**Detection:**
- ZeRO stage 1/2/3 all show `total_memory / num_gpus`
- Missing communication overhead warning
- Stage 3 memory same as Stage 2 (should be ~2x better)
- No explanation of on-the-fly parameter gathering

**Phase mapping:** Phase 3 (Multi-GPU Fine-tuning) - After single-GPU works

**Sources:**
- [Zero Redundancy Optimizer - DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [Memory Requirements — DeepSpeed](https://deepspeed.readthedocs.io/en/latest/memory.html)
- [Scaling Large Language Models with DeepSpeed ZeRO](https://medium.com/@dpratishraj7991/scaling-large-language-models-with-deepspeed-zero-zero-and-zero-offload-a-complete-guide-70d393e311f4)

---

### Pitfall 7: Activation Checkpointing Memory-Compute Trade-off Misrepresented

**What goes wrong:** Calculator shows "enable gradient checkpointing = 60% memory reduction" without mentioning compute cost. Reality: Checkpointing trades 25-40% slower training for 50-70% memory reduction. Selective checkpointing is critical to avoid OOM.

**Why it happens:**
- Focusing on memory savings only
- Not understanding recomputation cost
- Thinking checkpointing is "free memory"

**Consequences:**
- Users surprised by training slowdown
- Applying checkpointing to every layer (can cause OOM on CPU)
- Not understanding when NOT to use checkpointing

**Prevention:**

```
Gradient checkpointing (activation checkpointing):

How it works:
1. Forward pass: only save activations at checkpoint boundaries
2. Backward pass: recompute activations on-the-fly from checkpoints
3. Trade: Compute intermediate activations twice vs storing them once

Memory savings:
- Full storage: ~10-15 activations per layer * num_layers
- With checkpointing: ~2-3 activations per layer (checkpointed only)
- Reduction: 60-70% of activation memory
- Example: 12GB activations → 4GB with checkpointing

Compute overhead:
- Recompute activations during backward pass
- Slowdown: 25-40% longer training time
- Flash Attention: reduces recomputation cost (already optimized)

Selective checkpointing best practices:
- Checkpoint attention layers (expensive to compute, large activations)
- Don't checkpoint cheap operations (LayerNorm, residual adds)
- Don't checkpoint output layer (accessed frequently)
- Typical: checkpoint every 2-4 transformer blocks

Common mistake: Checkpointing every operation
- Can cause CPU memory issues (need to store checkpoint metadata)
- Minimal memory savings beyond every 2-4 blocks
- Maximum compute overhead

Calculator should show:
✓ Memory reduction: 60-70% of activation memory
✓ Compute overhead: +25-40% training time
✓ Recommendation: Enable for large models/long sequences
```

**Detection:**
- Only shows memory savings, no compute cost mention
- "Enable checkpointing" checkbox without explanation
- No selective vs full checkpointing option
- Missing guidance on when to use it

**Phase mapping:** Phase 2 (Advanced Fine-tuning) - After basic training works

**Sources:**
- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [Gradient Accumulation and Checkpointing](https://aman.ai/primers/ai/grad-accum-checkpoint/)
- [Gradient Checkpoints — PyTorch Training Performance Guide](https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html)

---

## Moderate Pitfalls

### Pitfall 8: LoRA Rank/Alpha Interaction Not Explained

**What goes wrong:** Calculator treats LoRA rank and alpha as independent. Reality: Alpha/rank ratio determines LoRA scaling. Increasing rank without alpha causes performance degradation. Rank affects memory linearly.

**Prevention:**

```
LoRA hyperparameter interaction:

Rank (r):
- Number of dimensions in adapter matrices (A: d×r, B: r×d)
- Memory: 2 * r * d_model * num_target_layers
- Typical values: 8-64 (low), 128-256 (high)
- Higher rank = more parameters = more capacity = more memory

Alpha (scaling factor):
- Scaling applied to adapter output: adapter_output * (alpha / rank)
- Does NOT affect memory
- Typical values: 16-32 (often 2x rank)
- Ratio alpha/rank should be >= 1.0

Common mistakes:
✗ Increasing rank from 16 to 64 without changing alpha (alpha/rank drops 4x)
✗ Setting alpha < rank (network becomes unstable)
✗ Assuming alpha affects memory (it doesn't)

Recommendation:
- Set alpha = 2 * rank (Microsoft recommendation)
- Start with rank 16, alpha 32 for simple tasks
- Use rank 64-128, alpha 128-256 for complex adaptations
- Memory scales linearly with rank

Memory calculation:
Llama 7B, rank 16, 4 targets per layer, 32 layers:
- Adapters: 2 * 16 * 4096 * 4 * 32 = 16.8M params
Llama 7B, rank 64, 4 targets per layer, 32 layers:
- Adapters: 2 * 64 * 4096 * 4 * 32 = 67.1M params (4x more memory)
```

**Detection:**
- Rank input exists but alpha doesn't
- No explanation of rank/alpha relationship
- Memory estimate doesn't change with alpha value (correct!)
- No guidance on choosing rank values

**Phase mapping:** Phase 1 (Fine-tuning MVP) - Important for user guidance

**Sources:**
- [LoRA fine-tuning Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Understanding LoRA Adapters Rank and Alpha Parameters](https://datawizz.ai/blog/understanding-lora-adapters-rank-and-alpha-parameters)
- [What rank (r) and alpha to use in LoRA](https://medium.com/@fartypantsham/what-rank-r-and-alpha-to-use-in-lora-in-llm-1b4f025fd133)

---

### Pitfall 9: Mixed Precision Training Complications

**What goes wrong:** Calculator has "training precision" dropdown (FP16/BF16/FP32) that changes all memory calculations proportionally. Reality: Mixed precision keeps master weights in FP32, optimizer states in FP32, only forward/backward in BF16.

**Prevention:**

```
Mixed precision training architecture:

"BF16 training" actually means:
- Forward pass computations: BF16 (2 bytes per activation)
- Backward pass gradients: BF16 (2 bytes per gradient)
- Master weights: FP32 (4 bytes, copy of model weights)
- Optimizer states: FP32 (8 bytes per param for AdamW)
- Final model weights: BF16 (2 bytes)

Memory breakdown (mixed precision):
- Model weights (BF16): params * 2 bytes
- Master weights (FP32): params * 4 bytes
- Gradients (BF16): params * 2 bytes
- Optimizer states (FP32): params * 8 bytes
- Total: params * 16 bytes

Memory breakdown (full FP32):
- Model weights (FP32): params * 4 bytes
- Gradients (FP32): params * 4 bytes
- Optimizer states (FP32): params * 8 bytes
- Total: params * 16 bytes (SAME as mixed precision!)

Key insight: Mixed precision saves memory on ACTIVATIONS (batch-dependent),
not on model weights/gradients/optimizer states. Savings are 20-30%, not 50%.

BF16 vs FP16:
- BF16: wider range, better for training, no loss scaling needed
- FP16: narrower range, requires loss scaling, can underflow
- Both: same memory usage (2 bytes per value)
- Recommendation: Use BF16 on Ampere+ GPUs (A100, H100, 4090)
```

**Detection:**
- Switching from BF16 to FP32 doubles all memory (wrong!)
- No mention of "master weights"
- Missing loss scaling explanation for FP16
- Optimizer state memory changes with training precision

**Phase mapping:** Phase 2 (Advanced Fine-tuning) - After basic training works

**Sources:**
- [Mixed Precision Training in LLMs: FP16, BF16, FP8, and Beyond](https://medium.com/@dpratishraj7991/mixed-precision-training-in-llms-fp16-bf16-fp8-and-beyond-b4af13ca846f)
- [How can using FP16, BF16, or FP8 mixed precision speed up my model training?](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training)
- [Performance and Scalability: How To Fit a Bigger Model](https://huggingface.co/docs/transformers/v4.15.0/performance)

---

### Pitfall 10: Framework-Specific Overhead Differences

**What goes wrong:** Calculator has single "training overhead" value for all frameworks. Reality: Unsloth uses 70% less VRAM than standard PyTorch/Transformers. vLLM and TGI are inference-only frameworks (can't be used for training).

**Prevention:**

```
Framework memory overhead (fine-tuning):

Standard PyTorch + Transformers:
- Baseline overhead: 500MB-1GB
- No special optimizations
- Reference implementation

Unsloth (optimized):
- 70% less VRAM than standard
- Shares vLLM weight space (no duplication)
- FP8 support with shared buffers
- Recommendation: Use for LoRA/QLoRA on consumer GPUs
- Savings: Qwen2.5-32B: 30GB saved, Qwen2.5-14B: 14GB saved

HuggingFace Accelerate:
- Adds ~200-500MB overhead
- Enables multi-GPU, mixed precision, DeepSpeed integration
- Small overhead for significant flexibility

DeepSpeed:
- ZeRO-Offload overhead: 500MB-1GB for CPU-GPU paging
- Communication buffers: 100-500MB per GPU
- Overall: Similar to standard with better scaling

vLLM / TGI:
- INFERENCE ONLY - cannot be used for training
- PagedAttention is for serving, not training
- Common misconception: "vLLM's memory optimizations apply to training" (WRONG)

Calculator should:
- Default: Standard PyTorch overhead (1GB)
- Unsloth option: 70% reduction on trainable params + activations
- Warning: "vLLM and TGI are inference-only frameworks"
```

**Detection:**
- vLLM/TGI listed as training framework options
- No Unsloth option (major omission for consumer GPU users)
- Framework overhead constant regardless of choice
- Missing "inference vs training framework" distinction

**Phase mapping:** Phase 2 (Advanced Fine-tuning) - After core calculations work

**Sources:**
- [Memory Efficient RL | Unsloth Documentation](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)
- [Ollama vs vLLM vs Unsloth: A Detailed Comparison](https://medium.com/@neeldevenshah/ollama-vs-vllm-vs-unsloth-a-detailed-comparison-from-an-ai-engineers-perspective-c6aba9a479d1)
- [vLLM vs. TGI](https://modal.com/blog/vllm-vs-tgi-article)

---

### Pitfall 11: Batch Size Scaling Confusion (Per-Device vs Effective vs Total)

**What goes wrong:** Calculator has "batch size" input that's ambiguous. Reality: In distributed training, there's per-device batch size, gradient accumulation steps, and number of devices. Effective batch = per_device * accum_steps * num_devices.

**Prevention:**

```
Batch size terminology (distributed training):

1. Per-device batch size (micro-batch):
   - Batch size processed on single GPU in single forward/backward pass
   - Directly affects activation memory
   - Example: per_device_batch_size = 2

2. Gradient accumulation steps:
   - Number of micro-batches before optimizer step
   - Example: gradient_accumulation_steps = 4

3. Number of devices (GPUs):
   - Example: num_devices = 8

4. Effective batch size (total):
   - effective_batch = per_device * accum_steps * num_devices
   - Example: 2 * 4 * 8 = 64
   - This is what matters for training dynamics

Memory impact:
- Activation memory scales with: per_device_batch_size (micro-batch)
- Gradient memory: same regardless of accumulation
- Optimizer memory: same regardless of accumulation
- Total memory does NOT scale with effective batch size!

Common mistake:
User sets "batch size = 64", calculator shows memory for 64 samples on single GPU.
Reality: User meant effective batch 64 = micro-batch 2 * 4 accum * 8 GPUs.

Calculator should ask:
- Per-device batch size: [1/2/4/8]
- Gradient accumulation steps: [1/2/4/8/16]
- Number of GPUs: [1/2/4/8]
- Show: "Effective batch size: X"
```

**Detection:**
- Single "batch size" input without clarification
- Memory scales linearly with batch size (missing activation vs gradient distinction)
- Multi-GPU interface doesn't mention per-device batch size

**Phase mapping:** Phase 3 (Multi-GPU Fine-tuning) - Critical for distributed training

**Sources:**
- [Batch size vs gradient accumulation](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260)
- [Batch size vs Gradient accumulation – Axolotl](https://docs.axolotl.ai/docs/batch_vs_grad.html)
- [Effective Training Techniques — PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)

---

### Pitfall 12: DeepSpeed ZeRO CPU Offload Overhead Not Modeled

**What goes wrong:** Calculator shows "ZeRO-Offload" checkbox that reduces GPU memory proportionally. Reality: CPU offload has 15-30% throughput cost from PCIe transfers, requires page-locked CPU memory (not unlimited), and adds complexity.

**Prevention:**

```
ZeRO-Offload memory and performance model:

ZeRO-Offload (Stage 2 + optimizer offload):
- GPU: Model weights + Gradients
- CPU: Optimizer states (AdamW: 8 bytes/param)
- Memory savings: ~40-50% GPU, but requires CPU RAM
- Throughput: -15-30% from PCIe transfer latency
- PCIe bandwidth: ~25-32 GB/s (PCIe 4.0 x16)

ZeRO-3-Offload (Stage 3 + parameters + optimizer offload):
- GPU: Active layer weights + activations (minimal)
- CPU: Model weights + optimizer states (streamed on-demand)
- Memory savings: 80-90% GPU, but requires significant CPU RAM
- Throughput: -20-40% from constant parameter streaming
- Best for: Models that don't fit on GPU at all

CPU RAM requirements:
- ZeRO-Offload: optimizer states size (~8 bytes/param)
- ZeRO-3-Offload: model weights + optimizer states (~10-12 bytes/param)
- Example: Llama 70B ZeRO-3-Offload needs ~700GB CPU RAM

Trade-offs:
✓ Can train models that don't fit on GPU
✓ Aggregate PCIe bandwidth improves with more GPUs
✗ 15-40% throughput reduction
✗ Requires page-locked CPU memory (limited resource)
✗ Adds complexity (CPU memory management)

Calculator should show:
- GPU memory reduced (show amount)
- CPU memory required (show amount)
- Throughput impact: "15-30% slower training"
- Warning: "Requires high CPU RAM and may cause system instability"
```

**Detection:**
- Offload checkbox with no CPU memory requirement shown
- No throughput impact warning
- Missing PCIe bandwidth consideration
- "Free memory" implication (it's not free)

**Phase mapping:** Phase 3 (Multi-GPU Fine-tuning) - Advanced feature

**Sources:**
- [ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/)
- [DeepSpeed ZeRO-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)
- [Scaling Large Language Models with DeepSpeed ZeRO](https://medium.com/@dpratishraj7991/scaling-large-language-models-with-deepspeed-zero-zero-and-zero-offload-a-complete-guide-70d393e311f4)

---

## Minor Pitfalls

### Pitfall 13: Flash Attention Backward Pass Overhead Ignored

**What goes wrong:** Calculator shows "Flash Attention = 10-20x memory savings" without caveats. Reality: Forward pass saves memory, but backward pass still needs to write intermediate values to HBM, reducing effectiveness. Training gets 2-4x speedup, not 10-20x.

**Prevention:**

- Flash Attention memory savings: 10-20x for inference (forward only)
- Training savings: 4-8x (backward pass needs intermediate values)
- Speedup: 2-4x wall-clock time (still significant!)
- Recommendation: Always enable for training with seq_len > 2K

**Phase mapping:** Phase 2 (Advanced Fine-tuning) - Important for accurate estimates

**Sources:**
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Out of the box acceleration and memory savings](https://pytorch.org/blog/out-of-the-box-acceleration/)

---

### Pitfall 14: Calculator Cross-Validation Errors Ignored

**What goes wrong:** Standalone calculator with no validation against real measurements. Reality: DeepSpeed's official estimator can be off by 2.2x (estimated 18GB, reality 40GB). Cross-validation is essential.

**Prevention:**

```
Validation strategy:

1. Ground truth comparison:
   - Test against real training runs (HuggingFace Transformers baseline)
   - Measure: nvidia-smi during training, track peak memory
   - Models: Llama 7B/13B/70B, Mistral 7B, Mixtral 8x7B

2. Error budget:
   - Target: <15% error for common configurations
   - Acceptable: <25% for edge cases
   - Flag: >25% error means investigation needed

3. Calculator comparison:
   - Compare against 2+ other VRAM calculators
   - If deviation >30%, investigate missing parameters
   - Reference: Modal's fine-tuning calculator (10% error when configured correctly)

4. User feedback loop:
   - "Report actual vs estimated" feature
   - Collect real-world measurements
   - Update formulas based on data

5. Document limitations:
   - "This is an estimate, actual usage varies by implementation"
   - "Test on your hardware before committing to large training runs"
   - Link to profiling guides
```

**Detection:**
- No validation against real measurements mentioned
- "Accuracy" claims without supporting data
- No user feedback mechanism
- Single calculator with no cross-validation

**Phase mapping:** Phase 5 (Validation & Polish) - Before public release

**Sources:**
- [LLM VRAM Calculator Guide 2026: Expert Memory Estimation Tips](https://www.propelrc.com/llm-vram-calculator/)
- [Estimating vRAM – Hamel's Blog](https://hamel.dev/notes/llm/finetuning/estimating_vram.html)
- [How much VRAM do I need for LLM model fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning)

---

### Pitfall 15: Activation Memory Formula Per-Layer Variation

**What goes wrong:** Calculator uses `activations = batch * seq_len * hidden_size * num_layers * 4`. Reality: MLP layers expand to intermediate_size (often 4x hidden_size), attention has different memory profile, residual connections add overhead.

**Prevention:**

```
Per-layer activation breakdown:

Attention layer:
- Q, K, V projections: batch * seq_len * hidden_size each
- Attention scores (if materialized): batch * n_heads * seq_len²
- Output projection: batch * seq_len * hidden_size
- Total: ~4x hidden_size * batch * seq_len (without Flash Attention)

MLP layer:
- Up projection: batch * seq_len * intermediate_size (typically 4x hidden)
- Activation function: same size
- Down projection: batch * seq_len * hidden_size
- Total: ~8x hidden_size * batch * seq_len

Residual connections: 2x hidden_size * batch * seq_len per layer

Total per transformer block: ~10-12x hidden_size * batch * seq_len

With gradient checkpointing:
- Store only checkpoint boundaries (every 2-4 blocks)
- Reduction: 60-70% of activation memory

Formula:
Without checkpointing: 10 * batch * seq_len * hidden * num_layers
With checkpointing: 3 * batch * seq_len * hidden * num_layers
```

**Phase mapping:** Phase 1 (Fine-tuning MVP) - Important for accurate estimates

---

### Pitfall 16: LoRA Target Layers Not Parameterized

**What goes wrong:** Calculator assumes LoRA applies to all attention layers. Reality: Users can choose which layers to target (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). Memory scales with number of targets.

**Prevention:**

```
LoRA target layer selection:

Common configurations:
1. Attention only (most common):
   - Targets: q_proj, k_proj, v_proj, o_proj
   - Params per layer: 4 * 2 * rank * hidden_size
   - Use case: General fine-tuning

2. Attention + MLP:
   - Targets: q,k,v,o + gate_proj, up_proj, down_proj
   - Params per layer: 7 * 2 * rank * hidden_size (1.75x more)
   - Use case: Complex adaptations

3. Minimal (budget):
   - Targets: q_proj, v_proj only
   - Params per layer: 2 * 2 * rank * hidden_size (50% of full attention)
   - Use case: Limited VRAM, simple tasks

Memory calculation:
Llama 7B, rank 16, 32 layers:
- 4 targets: 2 * 16 * 4096 * 4 * 32 = 16.8M params
- 7 targets: 2 * 16 * 4096 * 7 * 32 = 29.4M params (75% more!)
- 2 targets: 2 * 16 * 4096 * 2 * 32 = 8.4M params (50% less)

Calculator should:
- Expose target layer selection (dropdown: "attention only" / "attention + MLP" / "custom")
- Show parameter count changes with targets
- Explain memory impact of more targets
```

**Phase mapping:** Phase 2 (Advanced Fine-tuning) - After basic LoRA works

---

## Phase-Specific Warnings

| Phase | Focus | Likely Pitfalls | Mitigation |
|-------|-------|-----------------|------------|
| Phase 1: Fine-tuning MVP | Full FT, LoRA, QLoRA basics | #1 (KV cache reuse), #2 (optimizer precision), #3 (LoRA adapter-only), #5 (gradient accumulation) | Separate training from inference logic immediately. Validate against real training runs. Test with Llama 7B LoRA on 24GB GPU. |
| Phase 2: Advanced Fine-tuning | Mixed precision, checkpointing, Flash Attention | #7 (checkpointing trade-offs), #9 (mixed precision complications), #13 (Flash Attention backward pass) | Add detailed explanations for each optimization. Show compute vs memory trade-offs explicitly. |
| Phase 3: Multi-GPU Fine-tuning | DeepSpeed ZeRO, distributed training | #6 (ZeRO stage confusion), #11 (batch size terminology), #12 (CPU offload overhead) | Use official DeepSpeed formulas. Show per-GPU memory explicitly. Warn about communication overhead. |
| Phase 4: Framework Presets | vLLM, TGI, Unsloth, DeepSpeed configs | #10 (framework differences), #14 (validation errors) | Distinguish inference vs training frameworks clearly. Add Unsloth preset. Validate presets against documentation. |
| Phase 5: Validation & Polish | Cross-validation, user testing | #14 (calculator errors), all integration bugs | Compare against 2+ other calculators. Collect real measurements. Test edge cases (MoE, GQA, quantized). |

---

## Integration-Specific Warnings

### Reusing Inference Code for Training (HIGH RISK)

**Common mistakes:**
1. KV cache calculation (training doesn't use KV cache the same way)
2. Batch processing (inference is sequential generation, training is parallel batches)
3. Framework overhead (inference frameworks != training frameworks)
4. Memory profiling (inference peak != training peak due to gradients)

**Prevention:**
- Create separate calculation paths for inference vs training
- Share only: model weight calculation, quantization overhead, basic architecture params
- Don't share: KV cache, batch scaling, framework selection

### Adding Training Toggle to Existing Calculator (MEDIUM RISK)

**Common mistakes:**
1. "Training mode" checkbox that multiplies everything by 2x (wrong!)
2. Showing KV cache in training mode (doesn't apply)
3. Using same batch size semantics (per-device vs effective batch)

**Prevention:**
- Training mode should be a separate page/section, not a toggle
- Hide inference-specific fields (generation length, KV cache)
- Show training-specific fields (gradient accumulation, optimizer, checkpointing)

---

## Validation Strategy

### Unit Testing Critical Calculations

```typescript
// Test: LoRA adapter parameter count
test('LoRA adapter params: Llama 7B, rank 16, 4 targets', () => {
  const params = 2 * 16 * 4096 * 4 * 32
  expect(params).toBe(16_777_216) // 16.8M
})

// Test: Optimizer state memory (AdamW)
test('AdamW optimizer states: 8 bytes per trainable param', () => {
  const trainableParams = 16_777_216
  const optimizerMemory = trainableParams * 8 / (1024**3)
  expect(optimizerMemory).toBeCloseTo(0.125, 2) // 125MB
})

// Test: Mixed precision total memory
test('Mixed precision: 16 bytes per trainable param', () => {
  const params = 1_000_000_000
  const weights = params * 2 // BF16
  const masterWeights = params * 4 // FP32
  const gradients = params * 2 // BF16
  const optimizer = params * 8 // FP32
  const total = (weights + masterWeights + gradients + optimizer) / (1024**3)
  expect(total).toBeCloseTo(14.9, 1) // ~15GB
})
```

### Ground Truth Benchmarks

Test against real training runs:

```
Llama 7B LoRA (rank 16, 4 targets, batch 1, seq 2048):
- Expected: ~16-18GB
- Measure: nvidia-smi peak during training
- Acceptable error: <15%

Llama 7B QLoRA (rank 16, 4-bit base, batch 1, seq 2048):
- Expected: ~8-10GB
- Measure: nvidia-smi peak during training
- Acceptable error: <20%

Llama 70B DeepSpeed ZeRO-3 (4x A100, batch 2 per device):
- Expected: ~40GB per GPU
- Measure: nvidia-smi on each GPU
- Acceptable error: <20%
```

### Cross-Validation Against Other Calculators

Compare estimates against:
1. Modal's fine-tuning calculator (10% error when configured)
2. HuggingFace Accelerate memory estimator
3. DeepSpeed configuration generator
4. Community spreadsheets (validate formulas)

If deviation >30%, investigate formula bugs.

---

## Success Criteria

Fine-tuning estimation is complete when:

- [ ] Full fine-tuning, LoRA, and QLoRA calculations validated (<15% error)
- [ ] Optimizer state memory correctly accounts for trainable params only
- [ ] Gradient accumulation correctly reduces activation memory only
- [ ] DeepSpeed ZeRO stages have correct 2x/4x/8x memory profiles
- [ ] Mixed precision shows 16 bytes/param (not 8 or 32)
- [ ] Activation checkpointing shows both memory reduction AND compute cost
- [ ] LoRA rank/alpha relationship explained
- [ ] Training calculations completely separate from inference (no code reuse for KV cache)
- [ ] Framework presets distinguish inference-only (vLLM/TGI) from training (Unsloth/DeepSpeed)
- [ ] Per-device vs effective batch size terminology clear
- [ ] Ground truth validation: <15% error on 5+ test cases
- [ ] Cross-validation: <30% deviation from 2+ other calculators

---

## Sources

All sources verified current as of 2026-02-10:

### Optimizer States & Training Memory
- [Memory Requirements (HBM, GPU RAM)](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/memory-requirements-hbm-gpu-ram)
- [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one)
- [Modern Optimizers: AdamW, Lion](https://medium.com/@spjosyula2005/modern-optimizers-adamw-lion-and-what-actually-works-at-scale-68ffc033713b)
- [DeepSpeed Memory Requirements](https://deepspeed.readthedocs.io/en/latest/memory.html)

### LoRA & QLoRA
- [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://github.com/artidoro/qlora)
- [LoRA fine-tuning Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Understanding LoRA Adapters Rank and Alpha Parameters](https://datawizz.ai/blog/understanding-lora-adapters-rank-and-alpha-parameters)
- [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

### Gradient Accumulation & Checkpointing
- [Gradient Accumulation: Increase Batch Size Without Explicitly Increasing Batch Size](https://blog.dailydoseofds.com/p/gradient-accumulation-increase-batch)
- [Gradient Accumulation and Checkpointing](https://aman.ai/primers/ai/grad-accum-checkpoint/)
- [Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [Gradient Checkpoints — PyTorch Training Performance Guide](https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html)

### DeepSpeed ZeRO
- [Zero Redundancy Optimizer - DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [Memory Requirements — DeepSpeed](https://deepspeed.readthedocs.io/en/latest/memory.html)
- [ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/)
- [DeepSpeed ZeRO-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)
- [Scaling Large Language Models with DeepSpeed ZeRO](https://medium.com/@dpratishraj7991/scaling-large-language-models-with-deepspeed-zero-zero-and-zero-offload-a-complete-guide-70d393e311f4)

### Mixed Precision Training
- [Mixed Precision Training in LLMs: FP16, BF16, FP8, and Beyond](https://medium.com/@dpratishraj7991/mixed-precision-training-in-llms-fp16-bf16-fp8-and-beyond-b4af13ca846f)
- [How can using FP16, BF16, or FP8 mixed precision speed up my model training?](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training)
- [Performance and Scalability: How To Fit a Bigger Model](https://huggingface.co/docs/transformers/v4.15.0/performance)

### Framework Comparisons
- [Memory Efficient RL | Unsloth Documentation](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)
- [Ollama vs vLLM vs Unsloth: A Detailed Comparison](https://medium.com/@neeldevenshah/ollama-vs-vllm-vs-unsloth-a-detailed-comparison-from-an-ai-engineers-perspective-c6aba9a479d1)
- [vLLM vs. TGI](https://modal.com/blog/vllm-vs-tgi-article)

### Flash Attention
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [Out of the box acceleration and memory savings](https://pytorch.org/blog/out-of-the-box-acceleration/)

### Validation & VRAM Estimation
- [How much VRAM do I need for LLM model fine-tuning?](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [LLM VRAM Calculator Guide 2026: Expert Memory Estimation Tips](https://www.propelrc.com/llm-vram-calculator/)
- [Estimating vRAM – Hamel's Blog](https://hamel.dev/notes/llm/finetuning/estimating_vram.html)
- [Ultimate VRAM Calculator Guide 2026](https://orbit2x.com/blog/ultimate-vram-calculator-guide-gpu-memory-ai-models)

### Batch Size & Distributed Training
- [Batch size vs gradient accumulation](https://discuss.huggingface.co/t/batch-size-vs-gradient-accumulation/5260)
- [Batch size vs Gradient accumulation – Axolotl](https://docs.axolotl.ai/docs/batch_vs_grad.html)
- [Effective Training Techniques — PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
