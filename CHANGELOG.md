# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-09

### Added

- Inference VRAM calculator with 22 quantization formats (FP32, FP16, BF16, INT8, INT4, NF4, GPTQ, AWQ, GGUF Q2-Q8, NVFP4/6)
- KV cache estimation with GQA/MQA support and independent quantization
- MoE architecture handling (total params for VRAM, active params for compute)
- Performance estimation (tokens/sec, TTFT) using roofline model
- Multi-GPU support (1-8 GPUs) with tensor and pipeline parallelism
- Interconnect awareness (NVLink, PCIe) with validation warnings
- CPU/RAM and NVMe offloading with performance impact estimation
- Memory breakdown visualization (Recharts donut chart + table)
- Fit/no-fit indicator with actionable recommendations
- URL hash sharing with LZ-String compression
- Side-by-side comparison (up to 3 configs) with diff highlighting
- Curated GPU database (19 GPUs: NVIDIA datacenter/consumer, AMD MI300X, Apple Silicon)
- Curated model database (37 models with accurate architecture parameters)
- Custom model and GPU input for arbitrary hardware
- Data refresh scripts for HuggingFace model configs
- Web Workers for non-blocking calculations with sync fallback
- Responsive layout with dark mode
- Zod schema validation for all data
- Decimal.js precision arithmetic in all engines
