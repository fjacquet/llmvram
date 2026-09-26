# Engine Audit: Known-Good Data, Serving Memory, Decode Model

**Date:** 2026-09-26
**Status:** Implemented (PRs #54, #55, #56)

## Context

A full audit of the engines asked one question: can the numbers be trusted for
datacenter sizing (Kimi K3 / Qwen3.8 on GB300 NVL72)? Single-GPU basics held.
Multi-GPU memory, decode throughput, KV sizes for exotic attention and the
Blackwell FLOPS figures did not. This record keeps the decisions and their
sources; the CHANGELOG keeps the before/after numbers.

## Decisions

### D1. Reference values come from authorities, not formulas (PR #54)

Hand-derived KV sizes were wrong three times (Gemma 4 `global_head_dim`, Ling
`layer_group_size`, DeepSeek V4 compressed attention). So:

- **vLLM is the authority** for serving memory: what its `KVCacheSpec` allocates
  per layer (Full, MLA, SlidingWindow, ChunkedLocal, Mamba).
- **A second independent source must agree** before a value lands: HF
  transformers cache shapes on the meta device, or a published figure.
- Disagreement or a single source → value unchanged, flagged in the PR
  (Mistral Large 3, Nemotron Ultra).
- Data shape matches reality: optional `kv_sliding_elements_per_token` +
  `kv_sliding_window` (both or neither) for windowed layers.
- Vendor FLOPS pages list "with sparsity" figures; the database stores dense
  (B200 2250, GB300 2500). A test bounds every GPU below 2600.
- New weight formats FP8 (1 B) and MXFP4 (0.53125 B).

### D2. Multi-GPU memory follows vLLM (PR #55)

| Rule | Source |
|---|---|
| Embeddings and LM head shard across TP; only layer norms replicate | `VocabParallelEmbedding`, `ParallelLMHead` |
| KV splits `min(tp, kv_heads)` ways | `ModelConfig.get_num_kv_heads`: `max(1, kv_heads // tp)` |
| MLA KV is duplicated on every TP rank | `get_num_kv_heads`: `if use_mla: return 1`; LMCache connector: "Tensor parallel does not change the KV caches for MLA models" |
| One framework context + NCCL buffers per GPU | per-process CUDA context; NCCL ring/tree fixed connections |
| Interconnect efficiency is throughput, never memory | — |

New optional `use_mla` model field, set where the HF config carries
`kv_lora_rank` (14 models).

### D3. Decode = bytes per step / bandwidth + communication (PR #56)

- A step reads the weights once plus every sequence's KV at the full context.
- The compute ceiling is aggregate and includes attention FLOPs
  (`4 × layers × context × hidden` per token).
- Each GPU reads its share; tensor parallelism adds two all-reduces per layer;
  pipeline stages overlap by `B / (B + stages − 1)` (a decode token cannot be
  micro-batched).
- `INTERCONNECT_SPECS.allreduceLatencyUs`: NVLink 11 µs (NCCL ring, arXiv
  2607.16100; MSCCL 9.5 µs, arXiv 2504.09014), Infinity Fabric 20 µs (single
  source, arXiv 2508.11298), PCIe 25 µs (**unverified estimate**). NCCL ring is
  the conservative choice over low-latency kernels (~2.4 µs).
- The KV engine counts sliding-window layers at `min(window, context)`.

## Known limits (deferred)

- One bytes-per-parameter figure for every tensor: quantized formats under-count
  5-44% on small and hybrid models (model × quantization spike, 491 checkpoints).
- No constant-size linear-attention / Mamba state.
- No expert parallelism or DP-attention yet, so MLA models under TP carry the
  full duplicated KV. Addressed by the datacenter sizing PRD (4a).
- `fetch-models.ts` still derives KV generically; training / ZeRO untouched.
