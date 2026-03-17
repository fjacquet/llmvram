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
