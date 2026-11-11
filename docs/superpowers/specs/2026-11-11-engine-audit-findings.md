# Engine audit before rack sizing (piece 2): findings

**Date:** 2026-11-11
**Code audited:** `feat/rack-precision-compute` at 7d4a2e1 (piece 1 plus main #73-#77)
**Scope:** read-only. This document lists findings only; every fix is a decision for the user.
**Why now:** piece 2 divides a customer's users or tok/s target by per-replica sessions, tok/s
and first-token time. Any error in those per-replica numbers becomes the same error in the rack
count.

## Headline

**The engine is a best-case roofline, not a vLLM estimate.** Decode assumes 100% of HBM
bandwidth, and prefill assumes 45% of peak FLOPS with no cited source. Against published
measurements:
- decode comes out about 1.2x optimistic against TensorRT-LLM's best case, and about 3x
  optimistic against a vLLM deployment;
- prefill comes out about 2.4x optimistic.

A rack count built on these numbers would come out **1.2x to 3x too small**. That is the one
finding piece 2 cannot build around without a decision.

## C. Calibration (absolute numbers)

The engine was run on each configuration with the matching model, format, parallelism, batch and
context. The ratio is engine / measured.

| #   | Measured case                                                                       | Measured                        | Engine                                                         | Ratio                                                                                                         | Source quality                                                                            |
| --- | ----------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| C-a | vLLM 0.6.6, Llama 3.3 70B FP8, TP4, concurrency 64, ShareGPT (~222 in / 186 out)    | ITL 22.1 ms                     | 7.8 ms                                                         | 2.8x optimistic                                                                                               | Weak: GPU not stated (480 W/GPU suggests H100), old vLLM, prefill mixed into decode steps |
| C-b | Same run at concurrency 128                                                         | ITL 28.2 ms, 3,401 output tok/s | 8.5 ms, 15,130 tok/s                                           | 3.3x / 4.4x                                                                                                   | Same as C-a                                                                               |
| C-c | TensorRT-LLM perf overview, Llama 3.3 70B FP8, H100, TP2, 1000/1000, unbounded rate | 4,181 output tok/s              | 5,094 tok/s at batch 128 (the largest that fits at 2k context) | 1.2x optimistic, **if** the table is the total for 2 GPUs. If it is per GPU, the engine is 1.6x _pessimistic_ | Medium: the page says "Total Output Throughput" without stating per GPU or total          |
| C-d | Dell, Llama 3.3 70B FP8, B200, TP8, 16k prompt, concurrency 1                       | TTFT 462 ms                     | prefill 192 ms                                                 | 2.4x optimistic                                                                                               | Medium: single point, no matched BF16 run                                                 |

Causes in the code:
- **C1, decode at 100% HBM bandwidth.** `performance.ts:240-241`: `memorySeconds = bytes / memory_bandwidth_gbps`, with no achieved-bandwidth factor. vLLM also spends time on its scheduler, sampling, kernel launches and the attention kernel's own inefficiency, none of which the engine charges.
- **C2, prefill MFU 0.45 without a source.** `constants.ts:490-500` cites "the literature reports 40-60%" with no reference. C-d implies about 0.18 at batch 1, TP8, 16k.
- **C3, continuous batching is not modelled.** In a live server, other requests' prefills land in the same steps, so measured ITL includes prefill work. The engine treats decode and prefill as separate phases, which is optimistic for both under load.

Effect on rack count: linear. A 3x tok/s error is a 3x error in racks for a throughput target.
For a session-count target the error is smaller, because sessions come from memory, which is
modelled accurately (see M below).

**Needs a decision (yours):** how to make piece 2 safe. Options I can see:
1. An achieved-bandwidth (MBU) and an MFU factor per engine, sourced from measured runs.
2. Publish ranges: roofline as the best case, and the roofline divided by a measured factor as
   the expected case.
3. Calibrate against your own lab runs (vLLM `benchmark_serving.py` sweeps on one or two
   configurations).

Option 3 gives the only numbers you can defend in front of a customer.

## M. Memory and data

| #   | Finding                                                                                                                                                                                                                                                                             | Evidence                                                                                                            | Effect                                                                               |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| M1  | **Mistral Large 3 675B KV is overstated 24.9x.** The model is MLA (`kv_lora_rank` 512, `qk_rope_head_dim` 64, 61 layers) but has no `kv_cache_elements_per_token`, so the generic GQA formula runs with 128 KV heads.                                                               | Engine 874,496 elements/token against 35,136 from `params.json`. That is 213.5 GiB vs ~8.6 GiB per session at 128k. | Sessions understated about 25x. A wrong "does not fit" at long context.              |
| M2  | **Nemotron Ultra 253B KV is overstated 2.5x.** The model is NAS-pruned: only 64 of its 162 layers have attention (8 KV heads, head_dim 128). The formula counts all 162.                                                                                                            | Engine 331,776 elements/token against 131,072 from `block_configs`.                                                 | Sessions understated 2.5x.                                                           |
| M3  | The other 8 models without a KV override are plain GQA (the Llama 3.x family, Devstral 2, Mistral Medium 3.5), where the generic formula is exact.                                                                                                                                  | `num_kv_heads` present.                                                                                             | None.                                                                                |
| M4  | **The fit check ignores vLLM's memory utilisation.** "Does it fit" is `total > vram_gb` (no 0.9), while `maxConcurrentSessions` applies `GPU_MEMORY_UTILIZATION` 0.9 (`concurrency.ts:72`).                                                                                         | Seen while building the piece-1 boundary test.                                                                      | A configuration can show "fits" with 0 sessions. On a 278 GB GPU the gap is 27.8 GB. |
| M5  | **The unit basis of usable memory is unverified.** InferenceX's "278 GB" gives no unit. By analogy, NVIDIA's GB200 rack total (13.4 TB / 72 = 186) matches nvidia-smi on GB200 (~185 GiB), so 278 is probably GiB-like, which is what the engine assumes (`BYTES_PER_GB` = 1024^3). | Not confirmed for GB300.                                                                                            | If 278 were decimal GB, capacity is overstated 7%. Low risk.                         |
| M6  | Pipeline-parallel inference adds a 12% "activation stashing" overhead (`multi-gpu.ts:169-171`, `multi-node.ts:118`). Stashing is a training concept: inference keeps no activations for a backward pass.                                                                            | Unsourced constant.                                                                                                 | Small: activations are a minor share of inference memory.                            |

## R. Replica layouts piece 2 will generate (Kimi K3 MXFP4, GB300 NVL72, 128k, 64 users)

| Layout | per-GPU GB           | sessions / replica | replicas / rack | sessions / rack |
| ------ | -------------------- | ------------------ | --------------- | --------------- |
| TP 8   | 400.7 (does not fit) | 19                 | 9               | 171             |
| TP 72  | 237.6                | 67                 | 1               | 67              |
| EP 8   | 301.9 (does not fit) | 0                  | 9               | 0               |
| EP 16  | 203.1                | 273                | 4               | 1,092           |
| EP 36  | 148.3                | 1,085              | 2               | 2,170           |
| EP 72  | 126.3                | 2,547              | 1               | 2,547           |
- The behaviour matches the design. Under TP, MLA KV is duplicated (217 GB on every GPU whatever the degree). Under EP, KV divides by N and the replicated base dominates small groups.
- **Checked, not a bug:** TP 8 "does not fit" at the 64 requested users (400.7 GB) yet reports 19 sessions. That is right: 183 GB fixed + 19 x 3.4 GB KV is about 248 GB, within 278 x 0.9. Piece 2 should show it as "fits up to 19", not "does not fit".
- **R1:** `recommendedMaxTPDegree` is 8 for NVLink 5 and 6 on a 72-GPU domain, so every EP or TP group above 8 raises a soft warning. Piece 2 needs a domain-aware rule.

## S. Status of the 2026-09-26 audit list

| Item                                                                                                       | Status                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Decode ignores KV reads                                                                                    | Fixed (`performance.ts`, step 1)                                                                                                          |
| Multi-GPU decode × N × efficiency, no communication cost                                                   | Fixed (all-reduce latency per layer; decode never multiplied by efficiency)                                                               |
| Compute roofline multiplied by batch                                                                       | Fixed (aggregate ceiling)                                                                                                                 |
| KV ÷ numGPUs with no replication floor                                                                     | Fixed (`kvCacheTPShards`, MLA duplication)                                                                                                |
| Framework overhead charged per GPU cluster-wide; 3% replicated embeddings; communication counted as memory | Fixed in PR #55 per the audit record (not re-read line by line). `TP_COMMUNICATION_OVERHEAD` survives as a dead constant (`constants.ts`) |
| Generic KV formula (29/54 models without an override)                                                      | Mostly fixed: 44/54 have overrides now. Two of the remaining ten are wrong (M1, M2)                                                       |
| ZeRO-1/2 replicate master weights and divide activations by N                                              | **Not fixed** (`deepspeed.ts:78-101`). Training only, outside the rack-sizing scope                                                       |
| Kimi K3 linear-attention state                                                                             | Fixed (232 MB per session)                                                                                                                |
| Blackwell FP16 sparse figures                                                                              | Fixed (piece 1 adds FP8/FP4)                                                                                                              |

## U. Constants without a measured source

| Constant                                                        | Value          | Basis                                                                      |
| --------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------- |
| `tpScalingEfficiency` (NVLink 3/4/5/6, Infinity Fabric, PCIe)   | 0.65-0.98      | Derived (+0.05 per bandwidth doubling), not measured. Affects prefill only |
| `allreduceLatencyUs` PCIe                                       | 25 us          | Estimate, no measurement                                                   |
| `allreduceLatencyUs` Infinity Fabric                            | 20 us          | Single source (RCCL point-to-point)                                        |
| `allreduceLatencyUs` NVLink 6                                   | 11 us          | Copied from NVLink 5, no Rubin measurement                                 |
| `PREFILL_MFU`                                                   | 0.45           | "Literature 40-60%", no citation (C2)                                      |
| Decode achieved bandwidth                                       | 1.0 (implicit) | None (C1)                                                                  |
| `PP_COMMUNICATION_OVERHEAD` / `PP_ACTIVATION_STASHING_OVERHEAD` | 5% / 12%       | None (M6)                                                                  |
| `NCCL_BUFFER_PER_GPU_GB`                                        | 0.25           | Band figure from the 100-500 MB range, documented as such                  |

## What piece 2 needs settled first

1. **Calibration policy (C):** which of the three options. This is the gating decision.
2. **Fix M1/M2** before any sessions-per-rack answer ships. These are mechanical.
3. **M4 (fit vs 0.9):** one rule for "fits", the same rule the session count uses.

## Sources

- vLLM 0.6.6 Llama 3.3 70B FP8 run: https://gist.github.com/lhl/bf81a9c7dfc4244c974335e1605dcf22
- TensorRT-LLM perf overview: https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
- Dell B200 FP8 TTFT point: see `docs/adr/0008-per-precision-compute-and-usable-rack-memory.md` (D9 research)
- Mistral Large 3 `params.json`: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512
- Nemotron Ultra `config.json` `block_configs`: https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
- InferenceX GB300 NVL72 (278 GB usable): https://inferencex.semianalysis.com/chips/gb300-nvl72
