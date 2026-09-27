# 0001. Weight sizes are measured per format

**Status:** Accepted, 2026-09-27 (v1.12.0)
**Spec:** `docs/superpowers/specs/2026-09-27-weight-refs-and-model-audit-design.md`

## Context

A spike over 491 Hugging Face checkpoints showed the engine under-counting
NVFP4, INT4, small-model FP8 and GGUF Q2_K by 5-44%, the dangerous direction
("fits" when it does not). One bytes-per-parameter constant cannot be right:
which tensors stay 16-bit depends on who quantized the checkpoint, so two
releases of the same model in the same format differ.

## Decision

- Each model carries `weight_refs[format] = { repo, gib }`, the measured file
  size of a reference checkpoint (native release, then NVIDIA NVFP4,
  RedHatAI FP8/INT4, unsloth then bartowski GGUF, most-downloaded AWQ/GPTQ).
- The engine derives bytes per parameter from the ref; `BYTES_PER_PARAMETER`
  is only the fallback. fp16 and bf16 share refs (same size).
- `npm run refresh:models` audits curated data against Hugging Face and
  measures refs; it never writes `models.json`. Curation stays manual.

## Consequences

- Weight memory matches the real checkpoint for every measured format; the
  results say "measured from <repo>" or "estimated".
- Adding a model means measuring its refs; the auditor flags drift.
- A ref gives one average for the whole file: see ADR 0002 for MoE.

## Sources

Hugging Face repo trees and safetensors summaries; the live audit matched all
483 refs within 1% at release.
