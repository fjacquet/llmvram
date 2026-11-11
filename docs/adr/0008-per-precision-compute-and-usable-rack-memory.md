# 0008. Per-precision compute and usable memory for rack GPUs

**Status:** Accepted, 2026-11-10
**Spec:** `docs/superpowers/specs/2026-11-10-rack-data-precision-compute-design.md`

## Context

Presales sizing of GB200 NVL72, GB300 NVL72 and Vera Rubin racks needs per-GPU
numbers that hold up in front of a customer. Five things were wrong:

1. Compute always used FP16 TFLOPS (`fp16_tflops ?? fp32_tflops`), for decode's
   compute ceiling and for prefill. vLLM runs FP8 W8A8 on FP8 tensor cores
   (compute capability >= 8.9) and NVFP4 / MXFP4 as W4A4 on SM100+, so FP8 and
   FP4 checkpoints got a prefill time up to 2x (FP8) to 9x (Rubin FP4) too long.
2. GB200 NVL72, the most widely deployed rack, was missing.
3. Vera Rubin was missing (NVIDIA renamed "Vera Rubin NVL144" to NVL72: 72 GPU
   packages, not 144 dies).
4. GB300 NVL72 stored the physical 288 GB, not the usable figure. NVIDIA's rack
   page gives 20 TB of GPU memory (about 278 GB per GPU), and SemiAnalysis
   InferenceX measures 278. Same mistake as pitfall #8 (B200 is 180, not 192).
5. Rubin breaks three encoded rules: dense FP16 = dense FP8 / 2 (Rubin is 4 PF
   against 17.5 PF), the FP16 < 2600 guard test (Rubin is 4000), and the
   `INTERCONNECT_SPECS` efficiency slope (about 1.007 at 3000 GB/s).

## Decision

- **D1** `vram_gb` is what vLLM can allocate, never physical HBM: GB300 NVL72
  278, GB200 NVL72 186 (NVIDIA's 13.4 TB / 72 is already usable). Ids are kept.
- **D2** Rubin rows carry "(preliminary)" in their name until volume shipments.
  No new schema field.
- **D3** Three new rows: `nvidia-gb200-nvl72`, `nvidia-vera-rubin-nvl72`,
  `nvidia-hgx-rubin-nvl8`, each with a `spec_url` and a source comment per
  non-obvious figure (`scripts/fetch-gpus.ts`).
- **D4** Optional dense `fp8_tflops` and `fp4_tflops` fields. A field's
  presence means vLLM runs that precision natively on the card, not that the
  vendor publishes a figure. The one exception is the GB300 Desktop Superchip
  (`nvidia-gb300-desktop-252gb`, tier 'consumer'), which carries fp8/fp4.
  Workstation, consumer RTX, GB10 and Apple rows stay on FP16.
- **D5** One table (`src/engines/compute.ts`) maps each weight format to the
  tensor core vLLM computes it on: fp8 to FP8, nvfp4 to FP4 on NVIDIA only,
  mxfp4 to FP4, everything else to FP16. Without the field the format falls back
  to FP16 (Marlin W8A16 / W4A16), never to FP8. Attention always runs at FP16.
  Prefill and decode's compute ceiling share the function.
- **D6** `nvlink-6`: 3000 GB/s bidirectional per GPU (216 TB/s / 72), TP
  efficiency capped at 0.98, all-reduce latency 11 us as NVLink.
- **D7** Grace / Vera host link for the KV tier: GB200 joins GB300; Vera Rubin
  NVL72 gets its own value; HGX Rubin NVL8 has none (x86 host).
- **D8** CLAUDE.md pitfall #6 rewritten; the FP16 guard keeps 2600 with an
  allow-list (the two Rubin rows) and adds fp16 <= fp8 <= fp4.
- **D9** Prefill efficiency per precision, from a prefill-isolating source, or
  the fallback below.

### D9 outcome: B (fallback)

No source isolates FP4 prefill against a matched baseline, so FP4 formats on
FP4-native cards compute at the card's FP8 rate, and FP8 keeps `PREFILL_MFU`
0.45. **Consequence: FP8 and FP4 prefill are identical on a given card.**

Searched and rejected:

- TensorRT-LLM performance overview: output tokens/s only, B200 only in FP4.
- SemiAnalysis InferenceX precision comparisons: throughput and interactivity
  at 1k/1k, no TTFT.
- sglang #41919: NVFP4 TTFT on B200 with no FP8 or BF16 run to compare.
- TensorRT-LLM #5929: FP4 on B200 and FP8 on H200, different GPUs, dominated by
  batch-1 fixed overhead.
- Hugging Face `nvidia/Qwen3.6-27B-NVFP4` discussion 8: NVFP4 prefill at about
  0.81x FP8 on an RTX PRO 6000 (no FP8/FP4 peak in our data, different vLLM
  versions). Directionally it says FP4 does not reach 2x FP8 in prefill.
- Dell PowerEdge H200 vs B200: an FP8 single-request TTFT (Llama 3.3 70B,
  16k prompt, concurrency 1, TP8, 462 ms) gives an efficiency of about 0.18.
  Rejected: no matched BF16 run, and the gap is end-to-end fixed overhead, not
  tensor-core efficiency. Used, it would make FP8 prefill slower than BF16.

### Rulings made during implementation

- **HGX Rubin NVL8: one rule.** Every figure is the HGX page's system total / 8,
  compute included: 250 GB, 16250 GB/s, 3000 GB/s NVLink, FP16 / FP8 / FP4 4000 /
  16250 / 33750. The page's per-GPU figures (288 GB, 22 TB/s, 3.6 TB/s) don't
  divide back from its own totals; the lower figure wins.
- **Grace host link = min(link share, host-memory share).** GB200 and GB300
  NVL72: the C2C link gives 225 per GPU, LPDDR5X 14 TB/s / 36 Grace / 2 gives
  194, so 194. Vera Rubin NVL72: 450 (link), below 600 (memory).
- **Rubin usable memory** has no measured figure: GB300's reserve ratio
  (278 / 288) applied to 288 gives 278.
- **FP4 at the FP8 rate** (Outcome B above).
- **vLLM is the reference engine** for the kernel table (user decision,
  2026-11-10). Another engine gets its own table once it publishes its
  quantization support.

## Consequences

- fp8, nvfp4 and mxfp4 prefill and compute-bound decode get faster on Hopper,
  Blackwell and AMD MI300/MI350; 16-bit formats are unchanged. Example: Llama
  3.1 70B fp8 on 8x H100 SXM, 8192-token prompt, prefill 0.377 s to 0.202 s.
- A shared link that fit GB300 NVL72 at 288 GB may now read "does not fit".
  The verdict flips; nothing throws (pinned in `inference.integration.test.ts`).
- The prefill bottleneck label compares time, not FLOPs, because the two terms
  now run at different rates.
- When a prefill-isolating FP4 measurement appears, the FP4 entry of
  `PREFILL_MFU_BY_PRECISION` (today a 0.45 placeholder; the fp16 and fp8 entries are read by `performance.ts`, only fp4 is unread) gets
  the sourced value and `linearComputeRate` returns FP4.

## Sources

NVIDIA rack pages (nvidia.com/en-us/data-center/gb200-nvl72/, .../gb300-nvl72/,
.../vera-rubin-nvl72/) and the HGX page (.../hgx/); SemiAnalysis InferenceX
(inferencex.semianalysis.com, chips and compare-precision pages); ServeTheHome
on the HGX Rubin revision; vLLM docs: FP8 W8A8 (compute capability >= 8.9),
`CompressedTensorsW4A4Mxfp4`, `TorchNvFp4LinearKernel` (sm_10x),
`MarlinNvFp4LinearKernel` (W4A16). Per-figure sources are in
`docs/gpu-specs-reference.md`.
