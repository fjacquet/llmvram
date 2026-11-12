# LLM VRAM Calculator

[![Release](https://img.shields.io/github/v/release/fjacquet/llmvram?sort=semver)](https://github.com/fjacquet/llmvram/releases/latest)
[![CI](https://github.com/fjacquet/llmvram/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/fjacquet/llmvram/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Open the calculator: <https://fjacquet.github.io/llmvram/>** (runs in your browser, nothing is sent to a server)

How much GPU memory does a large language model need, how fast will it answer, and how many
servers or racks does it take to serve your users? This calculator answers those three
questions for 59 current open-weight models on datacenter, workstation and Apple hardware.

## Three modes

- **Inference**: pick a model, a quantization and a machine; see whether it fits, the memory
  breakdown, decode speed and time to first token, and how many sessions fit at your context.
- **Sizing**: give a model, a context and a number of simultaneous users; get the number of
  NVL72 racks, the replica split and the expected speed per user.
- **Training**: memory for full fine-tuning, LoRA and QLoRA, with DeepSpeed ZeRO sharding.

Save up to three results side by side in **Comparison**, share any setup as a link, and export
a PDF report or a PowerPoint deck.

## How to read the figures

- **Estimates, not benchmarks.** The memory figures follow what vLLM allocates. Speeds show an
  _expected_ figure, calibrated on published vLLM runs, beside the _theoretical ceiling_ of
  the hardware.
- **vLLM is the reference engine.** TensorRT-LLM, SGLang or llama.cpp will differ; read their
  figures as indicative ([ADR 0011](docs/adr/0011-vllm-reference-engine.md)).
- **Memory is per GPU.** An HGX or OAM baseboard counts as one 8-GPU machine, and results show
  both the per-GPU share and the machine total. Inference fits at 90% of memory, vLLM's
  default.
- **Model origin** is the publisher's headquarters country. It says nothing about where a
  model was trained, and it is not a measure of bias.

## Supported Hardware

| Vendor                | GPUs                                                                                                                                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA Datacenter** | H100 PCIe/SXM, H200 SXM/NVL, B200, HGX B300 (8-GPU) / GB300 NVL72 (72-GPU rack), GB200 NVL72, Vera Rubin NVL72 (preliminary), HGX Rubin NVL8 (preliminary), A100 PCIe/SXM, L40S, RTX PRO 6000, RTX 6000 Ada |
| **NVIDIA Consumer**   | RTX 5090, RTX 4090, RTX 3090, DGX Station (GB300 Desktop Superchip), DGX Spark (GB10)                                                                                                                       |
| **AMD**               | MI355X, MI350X, MI325X, MI300X                                                                                                                                                                              |
| **Apple Silicon**     | M1/M2/M3 Ultra, M1/M2/M3/M4/M5 Max                                                                                                                                                                          |

Plus custom GPU input for any hardware.

> **Note on the GB300 Desktop Superchip** (NVIDIA DGX Station, Dell Pro Max with GB300): the
> machine exposes 748 GB of coherent memory — 252 GB HBM3e at 7.1 TB/s plus 496 GB LPDDR5X at
> 396 GB/s. The calculator models a single memory pool, so this entry encodes only
> the 252 GB HBM3e tier. A model that spills past 252 GB will still load on the real hardware,
> but it runs at LPDDR5X speed, not the throughput shown here.

## Model Database

59 curated models including Gemma 4, Qwen3.8 / Qwen3.6, LLaMA 3.1 / 3.3 / Llama 4, DeepSeek V4 / V4.1 / R1, GLM 5.3 / 5.2 / 4.7, Kimi K3.1 Pro / K3 / K2.x / Linear, Ling 3.1 / 3.0, Mistral Large 4 / 3 / Medium 3.5, MiniMax M2.x / M3, Nemotron 3 / 3.5, LFM2.5, GPT OSS, and more. Plus custom model input.

## Quick Start

```bash
npm install
npm run dev
```

Open <http://localhost:5173> in your browser.

Plus custom GPU input for any hardware.

## Models

59 curated models, from Gemma, Qwen, Llama, DeepSeek, GLM, Kimi, Ling, Mistral, MiniMax,
Nemotron, LFM and GPT OSS, with weight sizes measured from published checkpoints per
quantization format. Custom models can be entered by hand.

## Learn more

- **In-app Guide**: the Guide tab explains every input and result.
- [Documentation index](docs/README.md): design decisions, calibration data, hardware sources
  and calculation pitfalls.
- [CHANGELOG](CHANGELOG.md): what changed in each release.
- [Developer guide](docs/development.md): build, test and contribute.

## License

MIT
