# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- NVIDIA RTX 6000 Ada (48 GB GDDR6, 960 GB/s, 91.1 TFLOPS FP32, 300 W) to GPU database (closes #11, #12)

## [1.4.0] - 2026-03-28

### Added

- **PWA / offline support**: app is now installable as a Progressive Web App and loads fully from cache without a network connection
  - Service worker (Workbox, `registerType: autoUpdate`) precaches all build assets on first visit; subsequent loads serve from cache
  - Web App Manifest (`manifest.webmanifest`): standalone display mode, indigo theme, icons at 64/192/512px + maskable variant
  - `start_url` / `scope` derived automatically from Vite `base` — works as `/` locally and `/llmvram/` on GitHub Pages with no extra config
  - Apple PWA meta tags (`apple-touch-icon`, `apple-mobile-web-app-capable`, `status-bar-style`) for iOS home screen install
  - Dual `theme-color` meta tags (light: `#6366f1`, dark: `#4f46e5`) for browser chrome theming

### Fixed

- Pinned `serialize-javascript` to `^7.0.5` via npm `overrides` to resolve two high-severity build-time vulnerabilities in the `vite-plugin-pwa → workbox-build → @rollup/plugin-terser` chain (GHSA-5c6j-r48x-rmvq, GHSA-qj8w-gfj5-8c6v)

## [1.3.0] - 2026-03-28

### Added

- **Concurrent users** parameter (1–256) for KV cache sizing: models the full load of simultaneous active sessions whose context must fit in VRAM at the same time
- Per-user speed (tok/s) and per-user TTFT (ms) metrics shown in the performance panel when `concurrentUsers > 1`
- **Interconnect variant selector**: GPUs with multiple interconnect options (e.g. NVLink-4 vs PCIe-5) now show a radio group to select the active variant, updating tensor parallelism efficiency accordingly
- **PPTX export**: export the full analysis (VRAM breakdown, performance, multi-GPU, and configuration summary) as a PowerPoint presentation with charts
- **OS dark mode sync**: `isDarkMode` now defaults to `prefers-color-scheme` on first visit and tracks live OS theme changes via a `matchMedia` listener — no page reload needed

### Fixed

- **PDF export** now works correctly with Tailwind v4: replaced `html2pdf.js` (bundles `html2canvas` v1.4.1 which cannot parse `oklch()`) with `html2canvas-pro` + `jspdf`; captures the full `#calculator-section` (InputPanel + ResultsPanel) using `windowWidth: 800` to force single-column layout in the PDF without affecting the live UI; dark mode is temporarily stripped before capture for correct contrast

## [1.2.3] - 2026-03-22

### Added

- MiniMax M2.7 model (228B MoE, 256 experts, 204.8K context) — 64 models total

## [1.2.2] - 2026-03-20

### Added

- 6 NVIDIA Nemotron models: Nemotron 3 Nano 4B, Nemotron 3 Nano 30B A3B, Nemotron 3 Super 120B A12B, Nemotron Llama 3.1 Nano 8B, Nemotron Llama 3.3 Super 49B, Nemotron Llama Ultra 253B (63 models total)

## [1.2.1] - 2026-03-19

### Added

- Mistral Small 4 119B model to curated database (57 models total)
- Favicon and logo assets

### Fixed

- 5 broken GPU spec sheet URLs (RTX PRO 6000, M3 Ultra, M3 Max, M2 Max, M1 Max)
- `flatted` 3.3.3 → 3.4.2 to resolve high-severity DoS vulnerability (GHSA-25h7-pfq9-p65f)

### Changed

- Upgraded Zod v3 → v4.3.6

### Documentation

- Added GPU specs reference and VRAM calculation pitfalls to `docs/`

## [1.2.0] - 2026-03-05

### Added

- Apple M5 Max (128 GB, 614 GB/s) to GPU database (now 20 GPUs)
- Interconnect-aware tensor parallelism scaling efficiency: NVLink-5 (97%), NVLink-4 (92%), PCIe-5 (78%), PCIe-4 (65%) — replaces flat 12% TP overhead
- `scalingEfficiency` and `interconnectBandwidthGBps` fields on `MultiGPUVRAMBreakdown` result
- Multi-GPU throughput scaling: tokens/sec now multiplied by `numGPUs × scalingEfficiency` to reflect real communication cost
- ShardingStrategySelector badge shows TP efficiency % alongside bandwidth (e.g. "NVLINK-4: 900 GB/s · 92% TP efficiency")
- MultiGPUBreakdownChart displays interconnect summary row with bandwidth, efficiency, and effective tokens/sec
- FitIndicator cluster view for multi-GPU: shows "Using X GB of N × Y GB = Z GB" instead of single-GPU capacity

### Changed

- `calculateMultiGPUVRAM` now requires a `GPU` argument to resolve interconnect type
- `calculateTensorParallelVRAM` derives comm overhead from interconnect bandwidth instead of flat constant
- `estimatePerformance` accepts optional `multiGPUResult` and scales throughput accordingly
- Worker and hook calculation order: VRAM → offloading → multi-GPU → performance (ensures multi-GPU result is available for performance scaling)
- Recommendations use actual `scalingEfficiency` from breakdown instead of hardcoded 0.85

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
