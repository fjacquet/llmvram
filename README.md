# LLM VRAM Calculator

Browser-based tool for estimating VRAM requirements and performance when running large language models on various GPU configurations. Pure client-side with no backend dependency.

## Features

- **VRAM Estimation**: Model weights, KV cache, activations, and framework overhead
- **22 Quantization Formats**: FP32, FP16, BF16, INT8, INT4, NF4, GPTQ, AWQ, GGUF Q2-Q8, NVFP4/6
- **KV Cache Quantization**: Independent from weight quantization (FP16, FP8, INT8, INT4)
- **Multi-GPU Support**: Tensor and pipeline parallelism with NCCL overhead and interconnect validation
- **Offloading**: CPU/RAM and NVMe offloading simulation with performance impact
- **Performance Estimation**: Tokens/sec, time-to-first-token, bottleneck analysis (roofline model)
- **Configuration Comparison**: Save and diff up to 3 configurations side-by-side
- **URL Sharing**: LZ-String compressed state in URL hash for shareable links
- **MoE Architecture**: Correct handling of total vs active parameters
- **Dark Mode**: System-aware with manual toggle

## Supported Hardware

| Vendor | GPUs |
|--------|------|
| **NVIDIA Datacenter** | H100 PCIe/SXM, H200, B200, A100 PCIe/SXM, L40S, RTX PRO 6000 |
| **NVIDIA Consumer** | RTX 5090, RTX 4090, RTX 3090, DGX Spark (GB10) |
| **AMD** | MI300X |
| **Apple Silicon** | M1/M2/M3 Ultra, M1/M2/M3/M4/M5 Max |

Plus custom GPU input for any hardware.

## Model Database

64 curated models including LLaMA 2/3/4, Mistral, Mixtral, Qwen, DeepSeek, Gemma, Phi, Command-R, Falcon, Yi, GLM, Kimi K2/K2.5, GPT OSS, Nemotron, MiniMax, and more. Plus custom model input.

## Quick Start

```bash
npm install
npm run dev
```

Open <http://localhost:5173> in your browser.

## Development

```bash
npm run build          # TypeScript check + Vite production build
npm run typecheck      # TypeScript only
npm run lint           # Biome linting
npm run lint:fix       # Biome auto-fix
npm run format         # Biome format
npm run test           # Vitest watch mode
npm run test:coverage  # Coverage report (75% thresholds on engines/utils)
npm run refresh:all    # Refresh model/GPU data from HuggingFace
```

Or use `make help` for all available targets.

## Technology Stack

- React 19 + Vite 7
- TypeScript (strict mode, `noUncheckedIndexedAccess`)
- Zustand (state management with URL hash persistence)
- Tailwind CSS v4 (dark mode)
- Recharts (visualization)
- Decimal.js (precision arithmetic)
- Zod (schema validation, single source of truth for types)
- Web Workers (non-blocking calculations)
- Biome (linting + formatting)
- Vitest (testing)

## Project Structure

```
src/
├── engines/        # Pure calculation logic (VRAM, performance, multi-GPU, offloading)
├── components/
│   ├── inputs/     # Model/GPU selectors, quantization, parameters
│   ├── outputs/    # Charts, tables, fit indicator, recommendations
│   ├── comparison/ # Side-by-side config comparison
│   ├── common/     # Dark mode toggle, shared components
│   └── layout/     # App layout, input/results panels
├── data/           # Static JSON databases (gpus.json, models.json)
├── store/          # Zustand stores (UI state, comparison, URL serialization)
├── hooks/          # useInferenceCalculation, useDarkMode, useURLSync
├── utils/          # Zod schemas, validation helpers
├── types/          # TypeScript type re-exports from Zod
├── workers/        # Web Worker for calculation offloading
└── test/           # Test setup
scripts/            # Data refresh scripts (HuggingFace fetch)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

## License

MIT
