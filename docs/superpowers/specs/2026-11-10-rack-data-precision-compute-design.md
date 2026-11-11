# Rack data and per-precision compute (piece 1 of the rack-sizing work)

**Date:** 2026-11-10
**Status:** Approved by the user 2026-11-10
**Series:** piece 1 of 3. Piece 2 is rack sizing that solves in reverse (users or
tok/s in, racks out, with the replica split of a 72-GPU domain). Piece 3 is the
model roster refresh. Each piece gets its own design and PR.

## Why

The user is a presales engineer. Neoclouds and large enterprises ask them to size
GB200 NVL72, GB300 NVL72 and Vera Rubin racks. The customer may hold them to
any of three numbers: tokens per second, users served, or racks to buy. The user
does not yet know how to size these racks, so the tool's numbers must be
defensible in front of a customer. Piece 2 builds rack sizing on top of the
per-GPU numbers, so they have to be right first.

## What is wrong today (verified in code and vendor pages, 2026-11-10)

1. **Compute always uses FP16 TFLOPS.** `performance.ts:253`
   (`decodeGpuTFLOPS = gpu.fp16_tflops ?? gpu.fp32_tflops`) feeds both the decode
   compute ceiling and prefill (time to first token). vLLM runs FP8 W8A8 on FP8
   tensor cores (compute capability >= 8.9), and NVFP4 / MXFP4 as true W4A4 on
   Blackwell (SM100+). The tool therefore overstates prefill time for FP8 and FP4
   checkpoints by 2x (FP8) up to 6x (GB300 FP4) and 9x (Rubin FP4).
2. **GB200 NVL72 is missing.** It is the most widely deployed rack.
3. **Vera Rubin is missing.** NVIDIA renamed "Vera Rubin NVL144" to **Vera Rubin
   NVL72**: it now counts 72 GPU packages, not 144 dies.
4. **GB300 NVL72 memory is the physical figure (288 GB), not the usable figure.**
   NVIDIA's rack page gives 20 TB of GPU memory, about 278 GB per GPU. That page
   writes 20.7 TB when it means 288 x 72 (Vera Rubin page), and "37 TB fast
   memory" = 20 + 17 (LPDDR5X), not 20.7 + 17. SemiAnalysis InferenceX measures
   278 GB usable per GB300 GPU. This repeats pitfall #8 (B200 is 180, not 192).
5. **Rubin breaks three rules the repo encodes:**
   - pitfall #6 "dense FP16 = dense FP8 / 2": Rubin is 4 PF FP16 against 17.5 PF FP8;
   - `src/utils/gpus.test.ts` guards every GPU's FP16 below 2600: Rubin dense is 4000;
   - the `INTERCONNECT_SPECS` slope (+0.05 efficiency per bandwidth doubling)
     gives about 1.007 at 3000 GB/s.

## Decisions

### D1. Usable memory, never physical (decided by the user: "use the safe version")

`vram_gb` is what vLLM can allocate. GB300 NVL72 becomes about 278 GB
(InferenceX 278, NVIDIA's 20 TB / 72). GB200 NVL72 is 186 GB (NVIDIA's 13.4 TB /
72 is already the usable figure). Ids are never renamed: `nvidia-gb300-nvl72`
keeps its id, and old shared links now show the safer number.

For Rubin no usable figure exists yet. _Our call:_ apply GB300's reserve ratio
(278 / 288) and get about 278 GB, until a measured figure exists.

### D2. Preliminary tag for Rubin

Rubin specs already moved once (HBM4 22 TB/s announced, 19.2 TB/s on the current
NVL72 page; NVLink 6 3.6 TB/s announced, 3 TB/s per GPU on the NVL72 page). Rubin
entries carry "(preliminary)" in their `name` until volume shipments. No new
schema field.

### D3. New GPU entries (in `scripts/fetch-gpus.ts`, then `npm run refresh:gpus`)

| id (new)                  | name                                               | vram_gb   | HBM GB/s | dense FP16 / FP8 / FP4 TFLOPS | interconnect | max_gpus_per_node |
| ------------------------- | -------------------------------------------------- | --------- | -------- | ----------------------------- | ------------ | ----------------- |
| `nvidia-gb200-nvl72`      | NVIDIA GB200 NVL72 (72-GPU rack) 186GB             | 186       | 8000     | 2500 / 5000 / 10000           | nvlink-5     | 72                |
| `nvidia-vera-rubin-nvl72` | NVIDIA Vera Rubin NVL72 (72-GPU rack, preliminary) | ~278 (D1) | 19200    | 4000 / 17500 / 35000          | nvlink-6     | 72                |
| `nvidia-hgx-rubin-nvl8`   | NVIDIA HGX Rubin NVL8 (8-GPU, preliminary)         | 250       | 16250    | 4000 / 16250 / 33750          | nvlink-6     | 8                 |

Sources: NVIDIA rack pages (gb200-nvl72, gb300-nvl72, vera-rubin-nvl72) and the
HGX page. Rubin FP4 is the dense "NVFP4 training" figure (35 PF). The 50 PF
"NVFP4 inference" figure is footnoted as sparse on NVIDIA's page, so it is not
used.

_Our call on HGX Rubin NVL8:_ NVIDIA's HGX page lists 22 TB/s and 3.6 TB/s per
GPU, but its own system totals (2 TB, 130 TB/s) don't divide back to those
figures. One rule for every figure, the page's system totals / 8: 2 TB to 250
GB, 130 TB/s to 16.25 TB/s, 24 TB/s NVLink to 3 TB/s (the lower figures), until
NVIDIA's pages agree. The rule covers compute too (controller ruling, 2026-11-10):
130 PF FP8 to 16250 and 270 PF NVFP4 training to 33750, not the NVL72 per-GPU
17500 / 35000. Memory follows the HGX B300 row's
rule (page system total / 8): "2 TB" / 8 = 250 GB, below D1's ~278. D1 asks for
the safe figure, so 250 it is.

Every entry follows the existing pattern: a `spec_url`, and a comment on each
non-obvious figure naming the source and whether it is dense.

### D4. Dense FP8 and FP4 compute fields

Add optional `fp8_tflops` and `fp4_tflops` (dense) to `GPUSchema`, the types and
the custom-GPU path (custom GPUs keep FP16 only; the new fields stay absent).
**A field's presence means "vLLM runs this precision natively on this card"**,
not "the vendor publishes a figure". _Our call (safe version):_ fill a field only
where that is true, with dense values and a source comment per card:

- `fp8_tflops`: Hopper (H100, H200), SM100 Blackwell (B200, HGX B300, GB200,
  GB300 NVL72 and Desktop), Rubin, AMD MI300-series and MI350-series.
- `fp4_tflops`: SM100 Blackwell and Rubin (NVIDIA: NVFP4 and MXFP4), AMD
  MI350-series (MXFP4 only, see D5).
- Left absent, so they stay on FP16: workstation and consumer cards (L40S, RTX
  6000 Ada, RTX Pro 6000, RTX 5090/4090/3090), GB10, Apple. Their stored FP16
  basis is mixed (the RTX Pro 6000 and RTX 6000 Ada rows look like the vector
  rate, 2 x FP32), so adding a tensor FP8 figure would fake a 4x speed-up. Fixing
  those rows is out of scope here.

Known values from NVIDIA pages (dense = sparse / 2 where the footnote says so):

- HGX B200: FP8 4500, FP4 9000 (72 PF FP8 and 72 PF FP4 dense per 8 GPUs)
- HGX B300: FP8 4500, FP4 13500 (FP4 108 PF dense per 8 GPUs)
- GB300 NVL72: FP8 5000, FP4 15000 (rack 720 PF FP8 sparse, 1080 PF FP4 dense)
- GB200 NVL72: FP8 5000, FP4 10000

The builder sources the rest (H100/H200, L40S, RTX parts, AMD) with one vendor
page plus one second source per figure, as in the 2026-09 audit.

### D5. Which tensor core runs each weight format

This table is **vLLM's** kernel choice; vLLM stays the reference engine (decided
by the user, 2026-11-10). Another engine (e.g. ZML/LLMD) would get its own table
once it publishes its quantization support. The linear (GEMM) FLOPs run at the
precision vLLM actually computes in. Attention
FLOPs always run at FP16 (_our call:_ conservative, because KV and attention
kernels stay 16-bit by default).

| Weight format                                            | Compute precision                                    | Fallback when the GPU lacks that figure                                            |
| -------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| fp8                                                      | FP8                                                  | FP16 (vLLM: W8A16 Marlin below compute capability 8.9)                             |
| mxfp4                                                    | FP4                                                  | FP16 (vLLM: W4A16 Marlin without native FP4). _Not_ FP8: Marlin computes in 16-bit |
| nvfp4                                                    | FP4 on NVIDIA cards only (AMD runs MXFP4, not NVFP4) | FP16                                                                               |
| nvfp6                                                    | FP16 (_our call:_ no confirmed native vLLM FP6 path) | FP32                                                                               |
| fp16, bf16, fp32, int8, int4, nf4, gptq, awq, all gguf-* | FP16 (today's `fp16 ?? fp32`, unchanged)             | FP32                                                                               |

Sources to cite in the code: vLLM docs via Context7, namely `CompressedTensorsW4A4Mxfp4`
("true W4A4 on SM100+ with FlashInfer, otherwise W4A16 weight-only via Marlin"),
`MarlinNvFp4LinearKernel` and `TorchNvFp4LinearKernel` (NVFP4 needs sm_10x), and
the FP8 W8A8 page (compute capability >= 8.9).

_Our call:_ int8 stays on FP16. The roster has no W8A8-INT8 checkpoints, and
vLLM's int8 path for this tool's formats is weight-only.

One selection function (e.g. `computeTFLOPS(gpu, format)`) is shared by the decode
compute ceiling and prefill, keeping the existing "one source, so the two
rooflines can't desynchronise" comment true.

### D6. NVLink 6

Add `nvlink-6` to the interconnect enum, `INTERCONNECT_SPECS` and
`INTERCONNECT_LABELS`: 3000 GB/s bidirectional per GPU (216 TB/s / 72, the same
convention as nvlink-5's 1800 = 130 TB/s / 72), `recommendedMaxTPDegree` 8,
all-reduce latency 11 us (same as NVLink, no Rubin measurement exists).
`tpScalingEfficiency`: cap the derived slope at 0.98 (_our call:_ the slope is
derived, not measured, and must stay below 1).

### D7. Grace / Vera host link for the KV tier

`graceLinkGBps` gains:

- `nvidia-gb200-nvl72`: 225 (same topology as GB300: 2 GPUs per Grace, 900 GB/s
  bidirectional C2C);
- `nvidia-vera-rubin-nvl72`: 450 (NVIDIA: 1.8 TB/s C2C per superchip of 2 GPUs,
  one direction halved, then shared by 2).

HGX Rubin NVL8 gets no host-grace tier (x86 host, like HGX B300). Config rule R3
must accept the new ids wherever it keys on Grace GPUs.

### D8. Rules and docs updated

- Rewrite CLAUDE.md pitfall #6: FP16 is dense; FP8 and FP4 are separate dense
  fields; the FP16 = FP8 / 2 ratio holds through Blackwell but not Rubin.
- Replace the FP16 < 2600 guard with: every GPU has `fp16 <= fp8 <= fp4` where
  present, and every datacenter card above 2600 FP16 is on an explicit allow-list
  with its source (Rubin).
- New ADR 0008: per-precision compute and usable memory for rack GPUs.
- CHANGELOG (breaking numbers: GB300 NVL72 memory, prefill for FP8/FP4 formats)
  and README GPU count (memory rule: docs always sync with data).

### D9. Prefill efficiency per precision (decided by the user: option B)

`PREFILL_MFU` (0.45) was set against FP16. Low-precision GEMMs usually reach a
lower fraction of peak, so 0.45 on FP4 peak may promise an optimistic time to
first token.

- The builder looks for measured prefill throughput at FP8 and FP4 on Hopper or
  Blackwell (MLPerf Inference, SemiAnalysis InferenceX, vendor-published vLLM or
  TensorRT-LLM benchmarks). From those it derives a per-precision efficiency
  (`PREFILL_MFU` FP16, FP8, FP4), each with a source comment, as for every
  other constant.
- **Fallback if no usable FP4 source is found:** FP4 formats compute prefill at
  the card's FP8 rate (`fp8_tflops`, 0.45). Same for FP8: without a source it
  keeps 0.45 on FP8 peak, because vLLM's FP8 W8A8 path is mature.
- Decode's compute ceiling keeps today's rule (peak TFLOPS, no MFU), but picks
  the same precision through the same selection function (D5), including the
  FP4-at-FP8 fallback.

## Goals

1. Every per-GPU figure for GB200, GB300 and Rubin traces to a vendor page plus a
   second source, dense, usable.
2. Time to first token and the decode compute ceiling use the tensor core the
   format actually runs on.
3. Results for formats that run on FP16 (fp16, bf16, AWQ, GPTQ, GGUF, int4, int8,
   nf4) are unchanged, except for GB300 NVL72 memory.

## Non-goals

- Rack sizing, replica split, reverse solver (piece 2).
- Model roster (piece 3).
- Rubin CPX and Rubin Ultra NVL576 / Kyber (2027 products).
- Power, cooling, cost (PRD non-goal).
- UI changes. ADR 0005: no new inputs. The GPU selector's TFLOPS line stays FP16.
- Training engines (they don't use FLOPS).

## Anti-goals

- A number the customer can't trace to a vendor page.
- Peak figures sold as usable: sparse TFLOPS, physical HBM.
- Changing an id, which breaks shared links.

## Success criteria

- `npm run typecheck`, `npm run lint`, `npx vitest run`, `npm run build` green.
- Tests: `computeTFLOPS` per row of the D5 table, including fallbacks (H100 +
  nvfp4 uses FP16, H100 + fp8 uses FP8); a prefill test showing an FP8 model's
  linear prefill term on B200 matches the ratio of the sourced efficiencies x
  peaks (D9) against its BF16 term; GB300 NVL72 `vram_gb`
  pinned with its source; NVLink 6 efficiency < 1.
- `gpus.json` regenerated, never hand-edited; GPU count in README and memory.

## Left to the builder

- Exact usable GB for GB300 NVL72 (278 or 279) after checking one more source
  (an nvidia-smi or vLLM log), staying at or below 278 if sources disagree.
- Where `computeTFLOPS` lives (performance.ts or a small new module) and how the
  format reaches it (it's already in the inference params).
- FP8 / FP4 figures for the cards not listed in D4, sourced as described.
