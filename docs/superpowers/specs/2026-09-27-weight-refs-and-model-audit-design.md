# Measured Weight Sizes and a Model Auditor

**Date:** 2026-09-27
**Status:** Draft, awaiting review

## Context

A spike (2026-09-26) compared the engine's weight size with 491 real Hugging Face
checkpoints across the 54 models. GGUF Q3_K-Q8_0 and BF16 agree within a few
percent. Quantized formats people deploy do not: the engine under-counts NVFP4,
INT4, FP8 on small models and GGUF Q2_K by 5-44%, the dangerous direction ("fits"
when it does not). The cause: one bytes-per-parameter figure for every tensor,
while real checkpoints keep some tensors in 16-bit.

What stays 16-bit depends on the **recipe** (who quantized it), not only on the
model:

| Model | Checkpoint | 16-bit | Rest |
|---|---|---|---|
| Gemma 4 31B | nvidia/Gemma-4-31B-IT-NVFP4 | 10.5B | FP4 |
| Gemma 4 31B | RedHatAI/gemma-4-31B-it-FP8-block | 2.0B | FP8 |
| Qwen3.8 27B | unsloth/Qwen3.8-27B-NVFP4 | 1.8B | 10.6B FP8 + FP4 |
| Kimi K3 | moonshotai/Kimi-K3 (native MXFP4) | 57.2B | FP4 experts |
| Kimi K3 | nvidia/Kimi-K3-NVFP4 | 21.0B | 36.2B FP8 + FP4 |

So a single per-model "unquantized parameters" figure cannot be right for two
releases of the same model. The fix follows the pattern used for KV sizes:
**measured known-good values with provenance, a formula only as fallback**.

Separately, `npm run refresh:models` (`scripts/fetch-models.ts`) fails validation,
estimates parameters with a rough formula, ignores `text_config`, and cannot read
gated repos. It becomes the tool that measures and audits these values.

## Goals

1. Weight memory matches the real checkpoint for every format that has one.
2. Formats without a checkpoint use constants derived from the format definition.
3. A repeatable tool audits curated model data against Hugging Face and measures
   new reference sizes; curation stays manual.
4. Users can see whether a weight figure is measured or estimated.

## Non-Goals

- A "native" format choice for mixed checkpoints (DeepSeek V4 FP8 + FP4): its
  native checkpoint is measured under `fp8`, and the limit is documented.
- Tensor-level recipe modelling.
- Changes to KV, sliding-window, MLA or linear-state fields.
- PDF/PPTX export of the new source line.
- The KV-split rule duplicated between `performance.ts` and `kv-tier.ts`.

## Section 1: Data and engine

### Data

New optional model field:

```ts
weight_refs?: Partial<Record<QuantizationFormat, { repo: string; gib: number }>>
```

- `gib` = total bytes of the checkpoint's weight files / 1024³ (the engine's GB).
  Measured, never computed.
- An entry exists only where a published checkpoint exists (typically 3-6 per model).
- `bf16` points at the original repo and doubles as a check on
  `num_parameters_billion`.
- Reference repo per format, in order:
  1. the model's native release (Kimi K3 MXFP4, DeepSeek FP8, Kimi K2 INT4);
  2. the vendor recipe: `nvidia/*` (ModelOpt) for NVFP4, `RedHatAI/*` for FP8 and
     INT4 (w4a16);
  3. GGUF: `unsloth/*-GGUF`, else `bartowski/*-GGUF`;
  4. AWQ / GPTQ: the most-downloaded repo for the model.

### Engine

One rule, used by every weight consumer:

```ts
effectiveBytesPerParameter(model, format): Decimal =
  model.weight_refs?.[format]
    ? gib × 1024³ / (num_parameters_billion × 1e9)
    : BYTES_PER_PARAMETER[format]
```

- `calculateModelWeightVRAM` takes the `Model` (not a bare parameter count) and
  uses it.
- The decode path (`performance.ts`, MoE active / batched parameters) multiplies
  its parameter count by the same effective bytes-per-parameter, so throughput
  also follows the real checkpoint.
- Multi-GPU, sessions, KV tier and offloading consume the weight figure and are
  unchanged.

### Fallback constants

Constants describe the quantized tensors only (16-bit parts are what the refs
capture). Each carries its source in `constants.ts`:

| Format | Now | New | Basis |
|---|---|---|---|
| `int4` | 0.5 | 0.5625 | 4 bits + 16-bit scale per group of 32, symmetric (Kimi K2 `quantization_config`: `group_size: 32`) |
| `awq`, `gptq` | 0.6 | 0.52 | 4 bits + 16-bit scale + 4-bit zero per group of 128 ≈ 4.16 bits |
| `gguf-q2_k` | 0.328 | 0.366 | median of 25 published Q2_K files in the 2026-09 spike ≈ 2.93 bpp (the Q2_K mix uses higher-bit types for some tensors) |
| `nvfp4` | 0.5625 | unchanged | E2M1 + FP8 scale per 16 = 4.5 bits (already correct) |

### UI

One line under the weight figure in the results panel:
"Measured from `<repo>`" (linked to the Hugging Face repo) or
"Estimated: no reference checkpoint for `<format>`".

## Section 2: Auditor and drafter (`npm run refresh:models`)

### Modules

- `scripts/hf.ts`: all network access. Token from `HF_TOKEN`, else
  `~/.cache/huggingface/token`; a gated repo without access is reported "skipped
  (gated)". Reads `config.json` through `text_config` when present; exact
  parameter counts and dtype mix from the safetensors API; file sizes from the
  repo tree.
- `scripts/model-audit.ts`: pure functions (unit-tested, no network): compare a
  curated entry with measured config/safetensors values; select a repo's weight
  files; pick the reference repo per format.
- `scripts/fetch-models.ts`: thin CLI over the two.

### Modes

- **Default (audit):** for every curated model, compare `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_kv_heads`, `num_experts`,
  `num_experts_per_token`, `context_length` and `num_parameters_billion`
  (tolerance 1%), and every `weight_refs` entry against the measured checkpoint
  (tolerance 1%). Also flag models whose config shows MLA (`kv_lora_rank`),
  sliding windows (`layer_types` / `sliding_window`) or linear attention
  (`linear_attn_config`, `layer_types`, `hybrid_override_pattern`) but lack the
  matching curated field — reported, never derived. Prints a drift report; exit 0,
  or 1 with `--strict`.
- **`--measure <model-id>`:** find reference checkpoints per the Section 1 rules
  and print ready-to-paste `weight_refs` JSON.
- **`--draft`:** for roster ids not in `models.json`, write draft entries to
  `src/data/models-fetched.json` with measured fields and candidate `weight_refs`.
- The script never writes `models.json`.

### Weight-file selection

- Safetensors: sum `*.safetensors`, excluding `original/`, `metal/` and
  `consolidated*` duplicates.
- GGUF: files whose name matches the exact quant tag
  (`(^|[-_./])<TAG>(\.gguf$|-\d{5}-of-|/)`), including split shards; exclude
  `mmproj*` and draft models.
- More than one candidate file set, or none: report the ambiguity; never guess.

### Initial data

`--measure` over all 54 models produces the first `weight_refs`, reviewed and
committed like the KV values.

## Section 3: Testing and delivery

### Tests

- Engine: `effectiveBytesPerParameter` returns the pinned size exactly when a ref
  exists and the constant otherwise; decode reads use the same value; each new
  constant matches its cited figure.
- Data integrity (`models.test.ts`): every ref has a repo and a positive `gib`;
  the implied bytes-per-parameter lies in a band for its format (nvfp4 0.5-1.0,
  mxfp4 0.5-1.0, int4 0.5-1.0, awq/gptq 0.5-1.0, fp8 0.95-1.3, bf16/fp16 1.9-2.1,
  GGUF within ±40% of its constant); `bf16` refs agree with
  `num_parameters_billion` within 2%.
- Corpus anchors from the spike, with their real file sizes: Gemma 4 31B NVFP4,
  Kimi K3 native MXFP4, Llama 3.1 8B FP8, one GGUF Q2_K, one AWQ — engine within
  1% using the ref.
- Auditor: pure functions against fixture configs and repo trees: `text_config`,
  GGUF shard sets, the Q2_K vs Q2_K_L trap, `original/` duplicates, a gated repo.
  No network in CI.

### Delivery

1. **PR 1, auditor + engine:** schema field, `effectiveBytesPerParameter`, new
   constants, rebuilt `refresh:models`, UI source line, tests. With no refs yet,
   results move only through the corrected constants.
2. **PR 2, data:** `weight_refs` for all 54 models via `--measure`, reviewed, plus
   the corpus anchors. This is where per-model corrections land.

### Docs

CHANGELOG, `CLAUDE.md` (key pattern for weight refs and the auditor;
`refresh:models` wording), `ARCHITECTURE.md`, README, guide.
