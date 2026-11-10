# GPU Specifications Reference

Authoritative spec sources for GPUs in `src/data/gpus.json`. This document resolves conflicting numbers found across third-party sources and records the most reliable data points with provenance.

---

## NVIDIA GB10 (DGX Spark)

### Product Overview

| Detail | Value |
|--------|-------|
| Official Name | NVIDIA DGX Spark |
| Chip | GB10 Grace Blackwell Superchip |
| Co-designed with | MediaTek |
| Architecture | Blackwell (compute capability 12.1) |
| Process Node | TSMC 3nm |
| Form Factor | 150 × 150 × 50.5 mm desktop mini |
| MSRP | $4,699 (as of Feb 2026; raised from ~$3,000 due to LPDDR5x supply constraints) |
| Availability | Shipping since mid-2025 |

### CPU

| Spec | Value |
|------|-------|
| Cores | 20 ARM v9.2 (10× Cortex-X925 + 10× Cortex-A725) |
| L3 Cache | 32 MB |

### GPU

| Spec | Value |
|------|-------|
| CUDA Cores | 6,144 |
| Tensor Cores | 5th Generation (FP4 support) |
| Multiprocessors | 48 |

### Memory

| Spec | Value |
|------|-------|
| Type | Unified LPDDR5X (coherent CPU–GPU) |
| Capacity | 128 GB |
| Usable GPU Memory | 119.7 GiB |
| Bandwidth | 273 GB/s |
| Interface | 256-bit |
| GPU L2 Cache | 50 MB |

> **Note**: This is unified memory — there is no separate VRAM. Both CPU and GPU share the full 128 GB pool. For VRAM calculator purposes, the effective GPU-accessible memory is 128 GB.

### Tensor Core Performance (Dense)

Benchmarked by NVIDIA employee (alan.dang) on the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993).

| Precision | TFLOPS |
|-----------|--------|
| FP4 (MXF4 / NVF4) | 427.3 |
| FP4 → FP16 | 213.8 |
| FP4 → FP32 | 213.7 |
| FP6 → FP16 | 213.9 |
| FP6 → FP32 | 213.7 |
| FP8 → FP16 | 213.8 |
| FP8 → FP32 | 213.7 |
| INT8 | 215.1 |
| **FP16** | **213.0** |
| **BF16** | **212.9** |
| TF32 | 53.3 |

### Sparse Performance

| Precision | TFLOPS |
|-----------|--------|
| FP4 (sparse) | ~1,000 (1 PFLOP) |

### Non-Tensor Core Performance

| Precision | TFLOPS |
|-----------|--------|
| FP32 | ~31 |

### Power

| Spec | Value |
|------|-------|
| Chip TDP | 140W |
| System PSU | 240W |

### Connectivity

| Spec | Value |
|------|-------|
| Networking | Dual QSFP Ethernet, 200 Gb/s aggregate |
| CPU–GPU Interconnect | NVLink C2C (on-chip) |
| Storage | 4 TB NVMe SSD |

### Discrepancy Resolution

- **`fp16_tflops: 213` in gpus.json is correct** — confirmed by NVIDIA employee benchmarks.
- Some third-party sources (including AI-generated summaries) incorrectly cite ~124 TFLOPS for FP16. This appears to derive from a naive 2× FP32 estimate (2 × 31 = 62) or confusion with CUDA-core-only performance.
- Official NVIDIA documentation only publishes "1 PFLOP FP4 sparse" — detailed per-precision numbers come exclusively from the developer forum benchmarks.

### Sources

- [NVIDIA DGX Spark Product Page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [DGX Spark User Guide — Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [Detailed Compute Performance Metrics — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/detailed-compute-performance-metrics-for-dgx-spark/351993)
- [NVIDIA DGX Spark In-Depth Review — LMSYS Org](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [Tom's Hardware — Going deep on GB10 Superchip](https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review/2)
- [MediaTek — GB10 Superchip Co-Designed](https://www.mediatek.com/press-room/newly-launched-nvidia-dgx-spark-features-gb10-superchip-co-designed-by-mediatek)
- [ServeTheHome — GB10 SoC at Hot Chips 2025](https://www.servethehome.com/nvidia-outlines-gb10-soc-architecture-at-hot-chips-2025/)
- [WCCFTech — NVIDIA Dissects GB10 Superchip](https://wccftech.com/nvidia-gb10-superchip-soc-3nm-20-arm-v9-2-cpu-cores-nvfp4-blackwell-gpu-lpddr5x-9400-memory-140w-tdp/)

---

## Apple M5 Max

### Product Overview

| Detail | Value |
|--------|-------|
| Official Name | Apple M5 Max |
| Announced | March 3, 2026 |
| Available | March 11, 2026 |
| Architecture | Fusion (two 3nm dies bonded into single SoC) |
| Found in | MacBook Pro 14″ and 16″ (2026) |

### CPU

| Spec | Value |
|------|-------|
| Cores | 18-core (6 super cores + 12 performance cores) |

### GPU

| Spec | 40-core variant | 32-core variant |
|------|-----------------|-----------------|
| GPU Cores | 40 | 32 |
| Neural Accelerator | 1 per GPU core | 1 per GPU core |

### Neural Engine

| Spec | Value |
|------|-------|
| Cores | 16-core |

### Memory

| Spec | 40-core variant | 32-core variant |
|------|-----------------|-----------------|
| Type | Unified | Unified |
| Capacity | Up to 128 GB | Up to 128 GB |
| Bandwidth | **614 GB/s** | **460 GB/s** |

### Compute Performance

| Spec | Value |
|------|-------|
| FP16 TFLOPS | ~35 (40-core GPU) |

### Media Engine

| Spec | Value |
|------|-------|
| Video Encode | 2× engines |
| ProRes Encode/Decode | 2× engines |

### Connectivity

| Spec | Value |
|------|-------|
| Thunderbolt | Thunderbolt 5 (on-chip controllers) |
| Wi-Fi | Wi-Fi 7 |
| Bluetooth | Bluetooth 6 |

### AI / LLM Highlights

- Up to 4× faster LLM prompt processing compared to M4 Pro/M4 Max
- Up to 8× faster AI image generation compared to M1 Pro/M1 Max
- 128 GB unified memory with 614 GB/s bandwidth enables running large LLMs locally

### Notes for gpus.json

- `vram_gb: 128` — top-bin configuration (128 GB unified memory)
- `memory_bandwidth_gbps: 614` — 40-core GPU variant; 32-core gets 460 GB/s
- `fp16_tflops: 35` — 40-core GPU variant
- `tier: apple-silicon` — consistent with other Apple entries
- `interconnect: unified` — Apple's unified memory architecture

### Sources

- [Apple debuts M5 Pro and M5 Max — Apple Newsroom](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [Apple introduces MacBook Pro with M5 Pro and M5 Max — Apple Newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/)
- [Apple M5 — Wikipedia](https://en.wikipedia.org/wiki/Apple_M5)
- [Apple M5 Max Processor — NotebookCheck](https://www.notebookcheck.net/Apple-M5-Max-Processor-Benchmarks-and-Specs.1244918.0.html)
- [MacBook Pro M5 Pro & Max 2026 guide — Macworld](https://www.macworld.com/article/2942089/macbook-pro-m5-pro-max-release-specs-price.html)
- [MacBook Pro 16-inch M5 Max Review — Macworld](https://www.macworld.com/article/3081408/m5-max-16-inch-macbook-pro-review.html)
- [MacBook Pro M5 Max Review — Tom's Hardware](https://www.tomshardware.com/laptops/macbooks/apple-macbook-pro-14-inch-m5-max-2026-review)
- [M5 Pro and M5 Max MacBook Pro — MacRumors](https://www.macrumors.com/2026/03/03/apple-debuts-m5-pro-and-m5-max-chips/)
- [M5 Pro and M5 Max specs and performance — AppleInsider](https://appleinsider.com/articles/26/03/03/how-m5-pro-and-m5-max-push-macbook-pro-into-high-bandwidth-ai-era)

---

## Rack-scale NVIDIA, Rubin, and dense FP8 / FP4 (verified 2026-11-10)

Every figure below is per GPU and **dense**. `vram_gb` is the usable figure,
never the physical stack size. Each value has a vendor source and a second
source. Where the two disagree, the lower value wins ("safe version").

How each source was read: the NVIDIA pages, every InferenceX page, the Flopper.io
pages for H100 PCIe, H200 NVL and MI355X/MI350X, the H100 datasheet PDF, the Meta
post and the ServeTheHome article were opened directly (fetched). The AMD
product pages and datasheets timed out, so their figures come from search-result
excerpts. So do CoreWeave, CUDO, Gigabyte, pi3g/Supermicro and Tom's Hardware.
Every InferenceX page says it is an unofficial copy that may be out of date.

### NVIDIA vendor figures and their sparsity footnotes

| Page | Raw figures (as published) | Footnote |
|------|----------------------------|----------|
| [GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) | rack: 20 TB HBM3e, up to 576 TB/s; 17 TB LPDDR5X, 14 TB/s; FP4 1440 \| 1080 PF; FP8 720 PF; FP16 360 PF | "All Tensor Core specifications are with sparsity unless otherwise noted." FP4 is given as sparse \| dense. |
| [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) | rack: 13.4 TB HBM3E, 576 TB/s; NVLink 130 TB/s; FP4 1440 \| 720 PF; FP8 720 PF; FP16 360 PF; FP32 5760 TF; 17 TB LPDDR5X, 14 TB/s; per superchip up to 480 GB LPDDR5X, up to 512 GB/s | "Specification in sparse. Dense is one-half sparse spec shown." |
| [Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) | per GPU: 288 GB HBM4, 19.2 TB/s, NVLink 6 3 TB/s; NVFP4 inference 50 PF; NVFP4 training 35 PF; FP8 17.5 PF; FP16 4 PF; FP32 130 TF. Superchip (2 GPU + 1 Vera): C2C 1.8 TB/s, up to 1.5 TB LPDDR5X | Inference is footnoted "Sparse specification"; training, FP8 and FP16 are footnoted "Dense specification". No TDP, no LPDDR5X bandwidth. |
| [HGX](https://www.nvidia.com/en-us/data-center/hgx/), Rubin NVL8 | system: 2 TB HBM4, 130 TB/s; NVLink Switch 24 TB/s; NVFP4 inference 400 PF; NVFP4 training 270 PF; FP8 130 PF; FP16 32 PF; FP32 1040 TF. Per GPU on the same page: 288 GB, 22 TB/s, 3.6 TB/s | Inference is footnote 1, "Sparse specification". Training, FP8 and FP16 are footnote 2, "Dense specification". |
| [HGX](https://www.nvidia.com/en-us/data-center/hgx/), B300 / B200 | 2.1 TB / 1.4 TB; FP4 144 \| 108 PF / 144 \| 72 PF; FP8 72 PF; FP16 36 PF | "Specification in Sparse \| Dense" for FP4. Elsewhere: "Specification in Sparse. Dense is ½ sparse spec shown." |
| [DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/) | FP4 "20 \| 15³" PF; FP8 10 PF; FP16 5 PF; 252 GB HBM3e, 7.1 TB/s; 496 GB LPDDR5X, 396 GB/s | Footnote 2 says "with sparsity unless otherwise noted". Footnote 3 says "Without sparsity". The "153" in the page text is 15 followed by footnote marker 3. |

### Values stored in `gpus.json`

| id | vram | GB/s | FP16 | FP8 | FP4 | Second source |
|----|------|------|------|-----|-----|---------------|
| `nvidia-gb300-nvl72` | 278 | 8000 | 2500 | 5000 | 15000 | [InferenceX GB300](https://inferencex.semianalysis.com/chips/gb300-nvl72): 278 GB usable, 2500 / 5000 / 15000. CoreWeave (via Northflank and whtop listings) gives 279 GB. The lower value wins. |
| `nvidia-gb200-nvl72` | 186 | 8000 | 2500 | 5000 | 10000 | [InferenceX GB200](https://inferencex.semianalysis.com/chips/gb200-nvl72): 186 GB, 2500 / 5000 / 10000, 1200 W. [Together AI](https://www.together.ai/gpu/nvidia-gb200-nvl72) and CoreWeave also list 186 GB. |
| `nvidia-vera-rubin-nvl72` | 278 | 19200 | 4000 | 17500 | 35000 | [Meta Prometheus post](https://engineering.fb.com/2026/10/19/data-center-engineering/inside-prometheus-meta-gigawatt-scale-supercluster/attachment/vera-rubin-prometheus/) gives 19.2 TB/s, 1.26 EF FP8 and 2.52 EF FP4 training per rack, and 3 TB/s NVLink. [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-details-rubin-architectural-optimizations-for-inference-improvements-target-better-performance-and-efficiency-from-the-gpu-to-the-rack) gives 4 PF FP16 and 130 TF FP32. Memory applies GB300's reserve ratio (278 / 288) to 288 GB. |
| `nvidia-hgx-rubin-nvl8` | 250 | 16250 | 4000 | 16250 | 33750 | [ServeTheHome, 2026-10-31](https://www.servethehome.com/?p=106378) reports the revised specs: 2.3 to 2.0 TB, 176 to 130 TB/s, FP8 140 to 130 PF, FP4 training 280 to 270 PF, and 250 to 252 GB usable per GPU. Every figure is the system total divided by 8. |
| `nvidia-b200-192gb` | 180 | 8000 | 2250 | 4500 | 9000 | [InferenceX B200](https://inferencex.semianalysis.com/chips/b200) |
| `nvidia-gb300-288gb` (HGX B300) | 262.5 | 8000 | 2250 | 4500 | 13500 | [InferenceX B300](https://inferencex.semianalysis.com/chips/b300). It says 268 GB usable, but the row keeps the HGX page's 2.1 TB / 8 = 262.5, the lower figure. |
| `nvidia-gb300-desktop-252gb` | 252 | 7100 | 2500 | 5000 | 15000 | pi3g comparison and the Supermicro Super AI Station listing give 20 PF FP4 sparse, 15 PF dense. |
| `nvidia-h100-80gb-pcie` | 80 | 2000 | 756 | 1513 | — | The H100 datasheet gives 3,026 TF FP8 with sparsity. Flopper.io gives 1,513 dense. |
| `nvidia-h100-80gb-sxm` | 80 | 3350 | 989 | 1979 | — | The [H100 page](https://www.nvidia.com/en-us/data-center/h100/) and the datasheet give 3,958 TF with sparsity. [InferenceX H100](https://inferencex.semianalysis.com/chips/h100) gives 1,979 dense. |
| `nvidia-h200-141gb` | 141 | 4800 | 989 | 1979 | — | The [H200 page](https://www.nvidia.com/en-us/data-center/h200/) gives 3,958 TF with sparsity. Flopper.io gives 1,979 dense. |
| `nvidia-h200-nvl-141gb` | 141 | 4800 | 835 | 1670 | — | The H200 page gives 3,341 TF with sparsity; half is 1,670.5, rounded down. Flopper.io gives the same 3,341 as "2:4 sparse only". |
| `amd-mi300x` | 192 | 5300 | 1307 | 2614 | — | AMD MI300X page: 2,614.9 TF (rounded down), or 5,229.8 with sparsity. [InferenceX MI325X](https://inferencex.semianalysis.com/chips/mi325x) states the same 2,615 as MI300X. |
| `amd-mi325x` | 256 | 6000 | 1307 | 2614 | — | AMD MI325X datasheet: 2,614.9 TF dense (rounded down). InferenceX: 2,615. |
| `amd-mi350x` | 288 | 8000 | 2300 | 4600 | 9200 | AMD MI350X page (excerpt): MXFP8 4.6 PF and MXFP4 9.2 PF. Flopper.io lists 4,614 / 9,227.5 dense. The stored values are rounded down to match the stored 2300 FP16. |
| `amd-mi355x` | 288 | 8000 | 2516 | 5033 | 10066 | AMD MI355X page (excerpt): FP8 5 PF and MXFP4 10.1 PF. Flopper.io and [InferenceX MI355X](https://inferencex.semianalysis.com/chips/mi355x) give 5,033 and 10,066, both labelled dense. |

The workstation and consumer rows (L40S, RTX 6000 Ada, RTX PRO 6000, RTX
5090/4090/3090), GB10 and Apple get no FP8 or FP4 figure (spec D4). Their
stored FP16 basis is mixed, so a tensor-core FP8 figure would overstate their
speed-up.

### Grace / Vera host link (`graceLinkGBps`)

Each value is min(C2C link share, host LPDDR5X share), one direction, per GPU.

| GPU | Link share | Memory share | Stored |
|-----|-----------|--------------|--------|
| GB300 / GB200 NVL72 | 900 GB/s bidirectional per Grace, 450 one way, shared by 2 GPUs = 225 | 14 TB/s across 36 Grace = 389 per Grace, 194 per GPU (NVIDIA rack pages; Gigabyte GB200 and GB300 NVL72 rack specs) | **194** |
| Vera Rubin NVL72 | 1.8 TB/s per superchip, 900 one way, shared by 2 GPUs = 450 | 1.2 TB/s per Vera CPU ([Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/nvidia-spills-the-beans-on-vera-cpu-spec-benchmarks-revealed-olympus-architecture-detailed-and-more/2)), 600 per GPU | **450** |
| GB300 Desktop | 900 GB/s, 1 GPU | 396 GB/s (DGX Station page) | **396** |

The sources conflict on NVL72 memory bandwidth. Together AI and CUDO quote
18.4 TB/s, which equals 36 × NVIDIA's "up to 512 GB/s" per superchip and gives
256 per GPU. At that figure the 225 link would be the limit. The 14 TB/s figure
is the lower one, so it wins. The Gigabyte pages returned HTTP 403 to automated
fetches, so their 14 TB/s figure comes from search-result excerpts.

### Other facts recorded, not used

- The Vera Rubin NVL72 rack lists 20.7 TB HBM4, which is 287.5 GB per GPU (about
  physical). The stored 278 is D1's estimate, not a published usable figure.
- Meta puts the Vera Rubin NVL72 rack at "over 170 kW". That is not a per-GPU TDP.
- ServeTheHome derives 16.8 TB/s per HGX Rubin GPU. The stored value is 130 / 8 = 16.25,
  the lower of the two.
- InferenceX gives the HGX B300 268 GB usable. The stored value is 262.5 (2.1 TB / 8),
  the lower of the two.

### Still open

- An aggregator quotes Lenovo's H200 NVL product guide at 1,570 TF dense FP8. That
  is 6% below the stored 1670, and it is unverified.
- No nvidia-smi or vLLM log for GB300 shows usable MiB. The stored 278 rests on
  NVIDIA's 20 TB / 72 and InferenceX.
- Rubin has no published TDP, LPDDR5X bandwidth or bus width. Those fields are
  absent from the Rubin rows, and `bus_width` is 0.
- GB200's TDP of 1200 W appears only on InferenceX, not on NVIDIA's page, so the
  field is left absent.
