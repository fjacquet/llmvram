# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-05

### Added

- 19 new models in curated database (56 total): DeepSeek R1, Gemma 3 (1B/4B/12B/27B), GPT OSS 20B/120B, Llama 4 Maverick 17B 128E, Llama 4 Scout 17B 16E, Llama 3.2 11B/90B Vision, Llama 3.3 70B, Magistral Small 2507, Ministral 3 14B Reasoning, Mistral Large 3 675B, Phi 3.5 Mini Instruct
- `context_length`, `license`, and `hf_url` metadata fields on all models in `models.json`
- `spec_url` metadata field on all GPUs in `gpus.json`
- ModelSelector: inline MoE badge, context length, and parameter count in dropdown; context length, license, and HuggingFace card link shown below selector when a model is selected
- GPUSelector: inline memory bandwidth in dropdown; bandwidth, memory type, FP16 TFLOPS, TDP, and spec sheet link shown below selector when a GPU is selected

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
