# Design: Accurate KV-Cache Math for Exotic-Attention Models

**Date:** 2026-07-03
**Status:** Approved (design); data verified from live HuggingFace config.json
**Scope:** One optional schema field + one engine branch + curated data values for 16
models. No UI changes, no multi-GPU changes.

## Problem

`calculateKVCacheVRAM` (src/engines/kv-cache.ts) uses the standard GQA formula:

```
KV_bytes = 2 × layers × hidden_size × seq × batch × precision × (num_kv_heads / num_attention_heads)
```

This assumes (a) every layer is a full-attention layer, (b) head_dim = hidden_size /
num_attention_heads, and (c) the cache stores K and V per KV head. All three assumptions
fail for the exotic-attention models in the database, with wildly inconsistent errors
(verified 2026-07-03 against live config.json):

| Model | Current formula error |
|---|---|
| DeepSeek R1, Kimi K2.x (MLA, dummy kv=heads) | **~25× overestimate** |
| GLM 5.2 (MLA + DSA) | ~21× overestimate |
| Kimi Linear (20/27 layers linear) | ~31× overestimate |
| Nemotron 3 Nano/Super/Ultra (mamba hybrids) | 6–11× overestimate |
| DeepSeek V4 (MLA, config kv=1) | **~5× UNDERestimate** |
| MiniMax M2.1/2.5/2.7 (head_dim=128 ≠ hidden/heads=64) | 2× underestimate |
| MiniMax M3 (head_dim=128 ≠ 96) | 1.33× underestimate |

## Solution: one generic override field

Every attention scheme reduces to "how many elements does the model cache per token."
Add exactly one optional field and one engine branch.

### 1. Schema (`src/utils/schemas.ts`)

```ts
// Exotic-attention override: total cached elements per token across ALL layers
// (MLA latent dims, hybrid attention-layers-only, explicit head_dim). When present,
// the KV-cache engine uses it directly instead of the GQA formula.
kv_cache_elements_per_token: z.number().int().positive().optional(),
```

`num_kv_heads` is unchanged (display + fallback). The Zod-inferred `Model` type picks
the field up automatically.

### 2. Engine (`src/engines/kv-cache.ts`)

Early branch in `calculateKVCacheVRAM`, before the GQA computation:

```ts
// Exotic-attention override (MLA, hybrid, explicit head_dim): the model declares
// its exact per-token cache size; precision quantization applies identically.
if (model.kv_cache_elements_per_token) {
  return new Decimal(model.kv_cache_elements_per_token)
    .mul(sequenceLength)
    .mul(batchSize)
    .mul(precisionBytes)
    .div(BYTES_PER_GB)
}
```

The existing GQA path is untouched — the 36 standard models compute exactly as before.
Multi-GPU sharding and inference totals consume the returned GB value; zero changes.
Constant-state memory (mamba SSM state, linear-attention recurrent state) does not
scale with sequence length and is excluded (documented in the docstring; it is a small
constant relative to weights).

### 3. Data (`src/data/models.json`) — 16 models

Values computed from live config.json fields on 2026-07-03. Derivations:
- **MLA:** `layers × (kv_lora_rank + qk_rope_head_dim)` — all are `× (512 + 64) = × 576`.
- **Hybrid:** `full_attention_layer_count × 2 × num_kv_heads × head_dim`.
- **Explicit head_dim GQA:** `layers × 2 × num_kv_heads × head_dim`.

| id | Value | Derivation | Confidence |
|---|---|---|---|
| deepseek-r1 | 35136 | 61 × 576 (kv_lora_rank=512, rope=64) | config-verified |
| moonshotai-kimi-k2-thinking | 35136 | 61 × 576 | config-verified |
| moonshotai-kimi-k2-instruct | 35136 | 61 × 576 | config-verified |
| moonshotai-kimi-k2.5 | 35136 | 61 × 576 (in text_config) | config-verified |
| zai-org-glm-5.2 | 44928 | 78 × 576 (MLA base; shared DSA indexer overhead small, not counted) | config-verified (base) |
| moonshotai-kimi-linear-48b-a3b | 4032 | 7 full-attn layers (`full_attn_layers` array) × 576; 20 KDA layers constant-state | config-verified |
| deepseek-v4-flash | 24768 | 43 × (512 + 64) (head_dim=512 ≙ renamed latent; CSA/HCA compresses further — this is the **upper bound**) | paper-inferred upper bound |
| deepseek-v4-pro | 35136 | 61 × 576 (same caveat) | paper-inferred upper bound |
| minimax-m3 | 61440 | 60 × 2 × 4 × 128 (explicit head_dim=128; sparsity affects reads, not storage) | config-verified |
| minimax-m2.1 | 126976 | 62 × 2 × 8 × 128 (all layers uniform full attention) | config-verified |
| minimax-m2.5 | 126976 | 62 × 2 × 8 × 128 | config-verified |
| minimax-m2.7 | 126976 | 62 × 2 × 8 × 128 | config-verified |
| nvidia-nemotron-3-nano-4b | 8192 | 4 attention layers (of 42: 21 mamba + 17 MLP + 4 attn) × 2 × 8 × 128 | config-verified |
| nvidia-nemotron-3-nano-30b-a3b | 3072 | 6 attention layers (of 52) × 2 × 2 × 128 | config-verified |
| nvidia-nemotron-3-super-120b-a12b | 4096 | 8 attention layers (of 88) × 2 × 2 × 128 | config-verified |
| nvidia-nemotron-3-ultra-550b-a55b | 6144 | 12 attention layers (of 108) × 2 × 2 × 128 | config-verified |

**DeepSeek V4 note:** the config lacks `kv_lora_rank`; `head_dim=512` is the renamed MLA
latent dim, and per-layer `compress_ratios` (CSA/HCA) shrink the real cache further
(paper claims ~10% of V3.2's at 1M ctx). We ship the uncompressed-MLA upper bound —
already ~5× more accurate than today, and conservative in the right direction for a
capacity planner.

**Not included (standard GQA, formula already correct):** GLM 4.7/4.7 Flash (true GQA
kv=8), DeepSeek R1 Distills (Llama/Qwen-based), all dense models, all standard MoE
(Mixtral-style, Qwen3.6, Gemma 4, GPT-OSS, Llama 4, Mistral, Devstral, Ministral,
Magistral). Note: Nemotron 3 Nano 4B was initially assumed dense-attention but its live
config shows `nemotron_h` hybrid — it IS included above (verified 2026-07-03).

### 4. Data correction (found during verification)

`nvidia-nemotron-3-ultra-550b-a55b` has `num_hidden_layers: 128` in models.json — wrong.
The live config's `layers_block_type` has **108** entries (48 mamba + 48 moe + 12
attention). Correct to **108** in the same change.

### 5. Fetch script note

`scripts/fetch-models.ts` gets a one-line comment: `kv_cache_elements_per_token` is
hand-curated (MLA/hybrid fields are not readable by the script); the script never
writes it.

## Testing

`src/engines/kv-cache.test.ts` (additions):
- Override path exact value: a DeepSeek-R1-shaped model (35136 elements) at fp16,
  seq 4096, batch 1 → `35136 × 4096 × 2 / 1024³ ≈ 0.268 GB` (vs ~6.67 GB via the old
  formula — the headline correction).
- Precision scaling on the override path: int8 result = exactly half of fp16.
- Fallback: a model without the field computes identically to before (existing tests
  remain untouched and passing).

`src/utils/models.test.ts` (addition):
- Invariant: every id in the 16-model list above has `kv_cache_elements_per_token` set
  to the exact table value; no other model has the field.

Full suite + typecheck + lint must pass.

## Error handling

None beyond Zod: positive int or absent; absent = current behavior. No user input
touches this field.

## Out of scope

- UI changes (approved decision: numbers just get correct).
- Modeling constant-state memory for mamba/linear layers.
- DeepSeek V4 CSA/HCA compression modeling (revisit if the paper's formula becomes
  extractable; the upper bound stands until then).
- Multi-GPU/offloading engine changes (consume GB totals; unaffected).
