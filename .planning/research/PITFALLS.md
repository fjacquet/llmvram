# Domain Pitfalls: LLM VRAM Calculators

**Domain:** LLM VRAM estimation tools
**Researched:** 2026-02-09
**Confidence:** LOW-MEDIUM (based on training data only - tools unavailable for verification)

**Note:** This research was conducted without access to current documentation tools. All findings are based on training data (January 2025 cutoff) and should be verified against current sources.

## Critical Pitfalls

These mistakes cause estimation errors >20% and can lead to OOM failures in production.

### Pitfall 1: Quantization Overhead Ignored

**What goes wrong:** Calculator assumes 4-bit quantization means exactly 0.5 bytes per parameter (4 bits / 8 bits). Reality: GPTQ adds grouping metadata, zero-point storage, scale factors. AWQ adds activation scales. Actual memory is 0.55-0.65 bytes per parameter for 4-bit models.

**Why it happens:**

- Treating quantization as pure bit-packing
- Ignoring format-specific metadata
- Not accounting for alignment padding

**Consequences:**

- 10-30% underestimation for quantized models
- OOM on models that "should fit"
- User loses trust in calculator

**Prevention:**

```
Quantization memory = (parameters * bits_per_param / 8) * overhead_multiplier

Overhead multipliers:
- GPTQ 4-bit: 1.1-1.3x (groupsize dependent)
- AWQ 4-bit: 1.15-1.25x
- bitsandbytes NF4: 1.1-1.2x
- GGUF Q4_K_M: 1.15x
- GGUF Q4_0: 1.05x
```

**Detection:**

- User reports "model doesn't fit but calculator says it should"
- Estimates match theory but not real-world measurements
- Larger errors on smaller quantization (2-bit, 3-bit)

**Phase mapping:** Phase 1 (MVP) must get this right or entire calculator is unusable

---

### Pitfall 2: KV Cache Quadratic Scaling Assumed

**What goes wrong:** Calculator uses formula `KV_memory = 2 * n_layers * d_model * seq_len`, assuming linear scaling. Reality: With attention mechanisms, effective memory depends on implementation. GQA (Grouped Query Attention) and MQA (Multi-Query Attention) drastically reduce KV cache.

**Why it happens:**

- Using formulas from original Transformer (2017)
- Not accounting for modern attention optimizations
- Ignoring architecture-specific KV reduction

**Consequences:**

- Overestimation for GQA/MQA models (Mistral, Llama 3, GPT-4)
- 2-8x overestimation for MQA models
- Users think they need more VRAM than required

**Prevention:**

```
KV_memory = 2 * n_layers * d_model * seq_len * (n_kv_heads / n_heads)

Standard attention: n_kv_heads = n_heads (ratio = 1.0)
GQA: n_kv_heads < n_heads (ratio = 0.125 to 0.5)
MQA: n_kv_heads = 1 (ratio = 1/n_heads, often ~0.03)

Example:
- Llama 3 70B: 8 KV heads, 64 query heads → 8/64 = 0.125x
- Mistral 7B: 8 KV heads, 32 query heads → 8/32 = 0.25x
```

**Detection:**

- Estimates are 2-4x higher than vLLM/TGI actual usage
- Same context length gives vastly different estimates for different models
- Missing n_kv_heads parameter in model config

**Phase mapping:** Phase 1 (MVP) - Core calculation logic

---

### Pitfall 3: MoE Active Parameters Miscalculation

**What goes wrong:** Calculator loads entire model weight into VRAM calculation. Reality: MoE models (Mixtral, DeepSeek V3) only activate a subset of experts per token. But ALL expert weights must be in VRAM, just not all activated simultaneously for compute.

**Why it happens:**

- Confusing "active parameters" with "loaded parameters"
- Marketing materials say "13B active, 47B total" → assuming 13B memory
- Not understanding sparse vs dense memory requirements

**Consequences:**

- Massive underestimation (up to 4x)
- Mixtral 8x7B calculated as 13B instead of ~47B
- Complete OOM on models that "should fit easily"

**Prevention:**

```
MoE memory calculation:

Model weights = total_parameters (NOT active_parameters) * bytes_per_param
- Mixtral 8x7B: 46.7B parameters loaded, ~13B active per token
- DeepSeek V3: All expert weights in VRAM, subset activated

KV cache = same as dense model (based on d_model, not active params)

Activation memory = based on active_parameters (lower than dense)
```

**Detection:**

- "13B MoE" calculator result similar to dense 13B
- Missing expert count / experts-per-token in config
- Estimates way below Mixtral/Qwen MoE actual measurements

**Phase mapping:** Phase 2 (Architecture Support) - Must be separate from dense model logic

---

### Pitfall 4: Multi-GPU Memory Split Naive

**What goes wrong:** Calculator divides memory by GPU count: `memory_per_gpu = total_memory / n_gpus`. Reality: Tensor parallelism has replication overhead (embeddings, layernorms), pipeline parallelism has activation stashing, and there's communication buffer overhead.

**Why it happens:**

- Assuming perfect sharding
- Ignoring replicated components
- Not accounting for interconnect buffers

**Consequences:**

- 10-20% underestimation for multi-GPU setups
- OOM on last GPU in pipeline
- Users frustrated by "2x GPU should be 2x model size"

**Prevention:**

```
Tensor Parallelism overhead:
- Embeddings: replicated on all GPUs (typically 2-5% of model)
- LayerNorms: replicated
- Effective split: 85-90% of model weight, not 100%

Pipeline Parallelism overhead:
- Activation stashing: 10-15% additional per stage
- Micro-batch buffers: depends on batch size

Per-GPU overhead:
- NCCL buffers: 100-500MB per GPU
- Framework overhead: 500MB-1GB PyTorch/CUDA
```

**Detection:**

- Multi-GPU estimates exactly total/n_gpus
- Missing TP/PP strategy selection
- Same estimate for 2x24GB and 1x48GB (should differ)

**Phase mapping:** Phase 3 (Multi-GPU) - Needs separate logic from single-GPU

---

### Pitfall 5: Fine-tuning Optimizer State Multiplier Wrong

**What goes wrong:** Calculator uses fixed "AdamW = 8 bytes per parameter" rule. Reality: Optimizer memory depends on trainable parameter count, not total parameters. With LoRA/QLoRA, only adapter weights have optimizer states.

**Why it happens:**

- Using full fine-tuning formulas for PEFT
- Not distinguishing trainable vs frozen parameters
- Ignoring gradient checkpointing impact

**Consequences:**

- 10-100x overestimation for LoRA/QLoRA
- Users think fine-tuning is impossible
- Massive underestimation for full fine-tuning (missing gradients)

**Prevention:**

```
Full Fine-tuning (all parameters trainable):
- Model weights: params * bytes_per_param
- Gradients: params * 4 bytes (fp32)
- Optimizer states (AdamW): params * 8 bytes (2x fp32 moments)
- Total: ~13x model weight (for fp32 training)

LoRA (only adapter trainable):
- Base model: params * bytes_per_param (quantized OK)
- LoRA adapters: r * d_model * num_layers * 2 (A and B matrices)
- Gradients: only for adapters (~1% of full)
- Optimizer states: only for adapters
- Total: ~1.2-1.5x base model (for rank 16-64)

QLoRA:
- Base model in 4-bit: params * 0.5
- Rest same as LoRA
- Total: often 40-50% of full fine-tuning memory
```

**Detection:**

- LoRA estimate same as full fine-tuning
- Missing rank parameter for LoRA
- No distinction between trainable/frozen parameters

**Phase mapping:** Phase 1 (MVP) if including fine-tuning, otherwise Phase 4

---

### Pitfall 6: Context Length Scaling Assumption

**What goes wrong:** Calculator scales KV cache linearly: doubling context = doubling memory. Reality: Attention computation memory scales quadratically with sequence length in standard implementations, though KV cache is linear. Flash Attention changes this.

**Why it happens:**

- Focusing only on KV cache (which is linear)
- Ignoring attention matrix materialization
- Not accounting for Flash Attention optimization

**Consequences:**

- Underestimation at long contexts (32K+)
- Missing memory spike during prefill phase
- Wrong estimates for training vs inference

**Prevention:**

```
Inference with Flash Attention:
- KV cache: linear with seq_len ✓
- Attention: O(1) memory (Flash Attention)
- Total: approximately linear

Inference without Flash Attention:
- KV cache: linear
- Attention matrix: O(seq_len²) in VRAM during compute
- Total: dominated by quadratic term at 16K+ context

Training:
- Always quadratic without Flash Attention
- Linear with Flash Attention v2+
- Gradient checkpointing trades compute for memory
```

**Detection:**

- Same memory estimate for 4K and 128K context (should be 32x)
- No Flash Attention toggle
- Missing prefill vs decode distinction

**Phase mapping:** Phase 2 (Architecture Support) - Context strategies

---

### Pitfall 7: Batch Size Memory Scaling Wrong

**What goes wrong:** Calculator assumes memory scales linearly with batch size. Reality: Model weights are constant, KV cache scales linearly with batch, but activations scale with batch AND sequence length. Padding inefficiency amplifies this.

**Why it happens:**

- Formula: `total = model_weights + batch_size * per_sample_memory`
- Treating all components as batch-dependent
- Ignoring padding waste in batched inference

**Consequences:**

- Underestimation for large batches
- Missing "sweet spot" batch size guidance
- No warning about padding overhead

**Prevention:**

```
Memory breakdown by batch scaling:

Constant (batch-independent):
- Model weights: independent of batch
- Embedding tables: independent of batch

Linear with batch:
- KV cache: batch_size * seq_len * kv_memory_per_token
- Output logits: batch_size * vocab_size * 4 bytes

Linear with batch AND seq_len:
- Activations: batch_size * seq_len * d_model * n_layers * multiplier
- Multiplier depends on architecture (2-4x typical)

Padding waste:
- If batch has sequences [100, 2000, 500, 300] → all padded to 2000
- Effective batch memory = batch_size * max_seq_len
- Can waste 50-80% memory with variable lengths
```

**Detection:**

- Batch=2 estimate is exactly 2x batch=1
- No padding efficiency consideration
- Missing dynamic batching explanation

**Phase mapping:** Phase 2 (Architecture Support) or Phase 4 (Optimization)

---

## Moderate Pitfalls

### Pitfall 8: GGUF Format Variants Not Distinguished

**What goes wrong:** Calculator has single "GGUF" quantization option. Reality: GGUF has 20+ quantization variants (Q4_0, Q4_K_M, Q5_K_S, Q6_K, IQ2_XXS) with 2-50% memory differences.

**Prevention:**

- Expose GGUF subtype selection
- Q4_0: 4.5 bits/param, Q4_K_M: 4.8 bits/param, Q5_K_M: 5.6 bits/param
- Link to GGUF quantization guide

**Phase mapping:** Phase 2 (Architecture Support) - Quantization deep-dive

---

### Pitfall 9: Framework Overhead Ignored

**What goes wrong:** Calculator estimates pure model memory. Reality: PyTorch has 500MB-1GB baseline overhead, CUDA context 200-500MB, vLLM paged attention 1-2GB buffer.

**Prevention:**

```
Framework overhead:
- PyTorch: 500MB-1GB
- CUDA context: 200-500MB per GPU
- vLLM paged KV cache: 10-20% additional
- TensorRT-LLM: 300-800MB

Add to every calculation: base_overhead + framework_overhead
```

**Detection:**

- Calculator shows 23.5GB for model that needs 24GB GPU
- No "available memory" vs "total memory" distinction

**Phase mapping:** Phase 1 (MVP) - Add constant overhead

---

### Pitfall 10: Activation Memory Formula Oversimplified

**What goes wrong:** Calculator uses `activations = batch * seq_len * hidden_size * 4`. Reality: Different layers have different activation sizes (MLP expands to intermediate_size, often 4x hidden_size).

**Prevention:**

```
Per-layer activation memory:
- Attention: batch * seq_len * hidden_size
- MLP: batch * seq_len * intermediate_size (typically 4x hidden)
- Residual connections: 2x hidden per layer
- Total per layer: ~10-12x hidden_size * batch * seq_len

With gradient checkpointing:
- Recompute activations instead of storing
- Trade 30-40% more compute for 60-70% less memory
```

**Phase mapping:** Phase 1 (MVP) for fine-tuning mode

---

### Pitfall 11: Model Parallelism Communication Ignored

**What goes wrong:** Multi-GPU calculator assumes GPUs work independently. Reality: Tensor parallelism requires all-reduce after each layer (~100GB/s bandwidth needed for 70B model), pipeline parallelism has bubble overhead.

**Prevention:**

- Add interconnect requirement notes (NVLink vs PCIe)
- Warn when model size / n_gpus / bandwidth > latency budget
- Show TP vs PP efficiency trade-offs

**Phase mapping:** Phase 3 (Multi-GPU) - Advanced section

---

### Pitfall 12: Precision Conversion Overhead Missing

**What goes wrong:** Calculator allows "load FP16 model, train in BF16" without warning. Reality: Conversion has temporary memory spike (need both versions in VRAM briefly), and mixed precision has FP32 master weights.

**Prevention:**

- Warn on precision mismatch
- Mixed precision: add FP32 master weights (4x optimizer memory)
- Show conversion memory spike

**Phase mapping:** Phase 4 (Fine-tuning Advanced)

---

## Minor Pitfalls

### Pitfall 13: Vocabulary Size Impact Ignored

**What goes wrong:** Calculator doesn't expose vocab_size parameter. Reality: Embedding and LM head memory = vocab_size *hidden_size* bytes_per_param. Models with 250K vocab (DeepSeek) use 10-15% more memory than 32K vocab models.

**Prevention:** Include vocab_size in architecture config, calculate embedding memory separately

**Phase mapping:** Phase 2 (Architecture Support)

---

### Pitfall 14: Rotary Embeddings Not Accounted

**What goes wrong:** RoPE (Rotary Position Embeddings) precomputed tables cached in VRAM. At 128K context, can be 500MB-2GB.

**Prevention:** Add RoPE cache = seq_len *hidden_size* 4 bytes (fp32)

**Phase mapping:** Phase 2 (Architecture Support) - Optional

---

### Pitfall 15: Shared Tensors in Multi-GPU

**What goes wrong:** Calculator shows 2x GPU = 2x memory available. Reality: Some frameworks share tensors across GPUs (e.g., DeepSpeed ZeRO stage 3 shards optimizer states but still needs full forward pass per GPU).

**Prevention:** Add ZeRO stage selector with accurate memory formulas

**Phase mapping:** Phase 3 (Multi-GPU) - Advanced

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1: MVP calculations | Quantization overhead (#1), MoE parameters (#3) | Use 1.15x multiplier for quantized, separate MoE logic immediately |
| Phase 2: Architecture variants | KV cache scaling (#2), Context length (#6) | Expose n_kv_heads, add Flash Attention toggle |
| Phase 3: Multi-GPU | Naive memory split (#4), Communication overhead (#11) | Add TP/PP overhead constants, warn on interconnect |
| Phase 4: Fine-tuning | Optimizer state (#5), Activation memory (#10) | Distinguish LoRA/Full FT, add gradient checkpointing |
| Phase 5: Advanced features | Batch size scaling (#7), Framework overhead (#9) | Profile real workloads, validate against vLLM |

---

## Validation Strategy

To prevent these pitfalls:

1. **Ground truth comparison:** For each model family (Llama, Mistral, Mixtral), measure actual VRAM usage with vLLM/TGI and compare to calculator
2. **Error budget:** Aim for <10% error on common configurations, <20% on edge cases
3. **Test matrix:**
   - Dense models: Llama 7B, 13B, 70B
   - MoE models: Mixtral 8x7B, DeepSeek V3
   - Quantizations: FP16, GPTQ 4-bit, AWQ 4-bit, GGUF Q4_K_M
   - Contexts: 2K, 8K, 32K, 128K
   - Batch sizes: 1, 4, 16, 64
4. **User testing:** Show estimates to experienced ML engineers, collect "this seems wrong" feedback
5. **Reference implementations:** Link to vLLM memory model, HF accelerate docs, DeepSpeed ZeRO paper

---

## Sources

**Confidence note:** This document was created from training data (January 2025 cutoff) without access to verification tools. All findings should be validated against:

- vLLM documentation (memory management, PagedAttention)
- HuggingFace transformers documentation (model architectures)
- DeepSpeed and Megatron-LM papers (multi-GPU strategies)
- GGML/llama.cpp quantization documentation
- Model cards on HuggingFace (architecture specifics for Llama 3, Mistral, Mixtral, DeepSeek)

Recommended validation sources:

- <https://github.com/vllm-project/vllm> (actual memory profiling)
- <https://huggingface.co/docs/transformers/main/en/model_memory_anatomy>
- <https://arxiv.org/abs/2205.05198> (FlashAttention paper)
- <https://arxiv.org/abs/1910.02054> (Megatron-LM, model parallelism)
- <https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md>
