import { useState } from 'react'

const SECTIONS = [
  { id: 'quick-start', label: 'Quick Start' },
  { id: 'mode-toggle', label: 'Mode Toggle' },
  { id: 'model-config', label: 'Model Configuration' },
  { id: 'gpu-selection', label: 'GPU Selection' },
  { id: 'hardware-config', label: 'Hardware Configuration' },
  { id: 'offloading', label: 'Offloading' },
  { id: 'parameters', label: 'Parameters' },
  { id: 'training-config', label: 'Training Configuration' },
  { id: 'results', label: 'Results Panel' },
  { id: 'comparison', label: 'Comparison View' },
  { id: 'sharing', label: 'URL Sharing' },
  { id: 'glossary', label: 'Glossary' },
]

function SectionHeading({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <h2 id={id} className="text-xl font-bold text-gray-900 dark:text-white mt-8 mb-3 scroll-mt-6">
      {children}
    </h2>
  )
}

function SubHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mt-5 mb-2">{children}</h3>
  )
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 leading-relaxed">{children}</p>
}

function GlossaryTerm({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="py-2 border-b border-gray-100 dark:border-gray-800 last:border-b-0">
      <dt className="text-sm font-semibold text-gray-900 dark:text-white">{term}</dt>
      <dd className="text-sm text-gray-600 dark:text-gray-400 mt-0.5">{children}</dd>
    </div>
  )
}

export function GuidePage() {
  const [tocOpen, setTocOpen] = useState(false)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr] gap-6">
      {/* Table of Contents — sidebar on desktop, toggle on mobile */}
      <aside>
        {/* Mobile TOC toggle */}
        <button
          type="button"
          onClick={() => setTocOpen(!tocOpen)}
          className="lg:hidden w-full text-left px-4 py-2 mb-4 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
        >
          {tocOpen ? 'Hide' : 'Show'} Table of Contents
        </button>

        <nav
          className={`${tocOpen ? 'block' : 'hidden'} lg:block lg:sticky lg:top-6 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-4`}
        >
          <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
            Contents
          </h3>
          <ul className="space-y-1">
            {SECTIONS.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  className="block text-sm text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 py-0.5 transition-colors"
                >
                  {s.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </aside>

      {/* Main content */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 lg:p-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">User Guide</h1>
        <P>
          This guide covers every control and output in the LLM VRAM Calculator. Click the{' '}
          <span className="inline-flex items-center text-gray-500">(i)</span> icons next to any
          control for a quick summary.
        </P>

        {/* Quick Start */}
        <SectionHeading id="quick-start">Quick Start</SectionHeading>
        <ol className="list-decimal list-inside text-sm text-gray-700 dark:text-gray-300 space-y-2 mb-4">
          <li>
            <strong>Pick a model</strong> — Search or scroll the curated list of 37 models, or
            define a custom model with your own parameter count and architecture details.
          </li>
          <li>
            <strong>Pick a GPU</strong> — Choose from 20 GPUs across NVIDIA, AMD, and Apple Silicon,
            or enter custom hardware specs.
          </li>
          <li>
            <strong>Read the results</strong> — The right panel instantly shows whether the model
            fits, a VRAM breakdown, and performance estimates. If it doesn&apos;t fit, follow the
            recommendations.
          </li>
        </ol>

        {/* Mode Toggle */}
        <SectionHeading id="mode-toggle">Mode Toggle</SectionHeading>
        <P>
          The toggle at the top of the input panel switches between <strong>Inference</strong> and{' '}
          <strong>Fine-tuning</strong> mode.
        </P>
        <P>
          <strong>Inference mode</strong> estimates VRAM for running a model: weights + KV cache +
          activations + framework overhead. <strong>Fine-tuning mode</strong> adds optimizer states,
          gradients, and activation memory required for training.
        </P>

        {/* Model Configuration */}
        <SectionHeading id="model-config">Model Configuration</SectionHeading>
        <SubHeading>Model Selector</SubHeading>
        <P>
          A searchable dropdown with 37 curated models (LLaMA, Mistral, Qwen, DeepSeek, etc.). Each
          entry shows the parameter count in billions and a &quot;MoE&quot; badge for
          Mixture-of-Experts models. Select &quot;Custom model...&quot; at the bottom to specify
          your own architecture.
        </P>
        <P>
          <strong>Custom model fields:</strong> Name (required), Parameter count in billions
          (required), Hidden size, Number of layers, Number of attention heads. The calculator uses
          sensible defaults when optional fields are omitted.
        </P>

        <SubHeading>Quantization Picker</SubHeading>
        <P>
          Choose the precision format for model weights. Lower precision means less VRAM but may
          reduce output quality. Formats are grouped into five categories:
        </P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Float</strong> — FP32 (4 bytes/param), FP16 (2 bytes), BF16 (2 bytes)
          </li>
          <li>
            <strong>NVIDIA FP</strong> — NVFP6 (~0.75 bytes), NVFP4 (~0.5 bytes)
          </li>
          <li>
            <strong>Integer</strong> — INT8 (1 byte), INT4 (0.5 bytes), NF4 (0.5 bytes)
          </li>
          <li>
            <strong>GPTQ / AWQ</strong> — 4-bit with calibration overhead (~10-20% extra)
          </li>
          <li>
            <strong>GGUF</strong> — llama.cpp formats from Q8_0 down to Q2_K, optimized for CPU+GPU
            inference
          </li>
        </ul>

        {/* GPU Selection */}
        <SectionHeading id="gpu-selection">GPU Selection</SectionHeading>
        <P>
          A searchable dropdown grouped by manufacturer (NVIDIA, AMD, Apple). Each GPU shows its
          VRAM capacity and tier badge (DC = Datacenter, Consumer, Apple).
        </P>
        <P>
          <strong>Key specs that affect calculations:</strong> VRAM determines fit/no-fit. Memory
          bandwidth determines inference speed (tokens/sec). FP16 TFLOPS determines compute
          throughput. Interconnect type (NVLink, PCIe) affects multi-GPU efficiency.
        </P>
        <P>
          Select &quot;Custom GPU...&quot; to specify any hardware with name, VRAM (required),
          bandwidth (optional), and FP16 TFLOPS (optional).
        </P>

        {/* Hardware Configuration */}
        <SectionHeading id="hardware-config">Hardware Configuration</SectionHeading>
        <P>This section appears after selecting a GPU.</P>

        <SubHeading>Number of GPUs</SubHeading>
        <P>
          Slider from 1 to 8 GPUs. Multiple GPUs allow running models that exceed a single
          GPU&apos;s VRAM. The calculator accounts for communication overhead and memory
          replication.
        </P>

        <SubHeading>Sharding Strategy</SubHeading>
        <P>Visible when using 2+ GPUs. Two options:</P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Tensor Parallel</strong> — Splits each layer horizontally across GPUs. Requires
            fast interconnect (NVLink). Best for single-node setups.
          </li>
          <li>
            <strong>Pipeline Parallel</strong> — Assigns complete layers to different GPUs. Works
            over slower PCIe but introduces pipeline bubbles. Each GPU stores the full KV cache.
          </li>
        </ul>
        <P>
          A colored badge shows the detected interconnect and its bandwidth. Green = NVLink
          (excellent), Yellow = PCIe (adequate), Red = none (multi-GPU may not work). A warning
          appears if the tensor parallel degree exceeds the recommended maximum for the
          interconnect.
        </P>

        {/* Offloading */}
        <SectionHeading id="offloading">Offloading</SectionHeading>
        <P>
          When VRAM is insufficient, offload parts of the model to system memory or storage. Enable
          the toggle to reveal offloading options.
        </P>

        <SubHeading>Offload Target</SubHeading>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>CPU/RAM</strong> — Offload via PCIe to system memory. 2-15x slower depending on
            offload percentage.
          </li>
          <li>
            <strong>NVMe SSD</strong> — Offload to storage. Slower than CPU/RAM but useful when
            system memory is also limited.
          </li>
        </ul>

        <SubHeading>Offload Mode</SubHeading>
        <P>
          Choose &quot;By Percentage&quot; (slider 0-100%) or &quot;By Number of Layers&quot;
          (slider 0 to total model layers). Both methods control how much of the model weights are
          moved off GPU.
        </P>

        <SubHeading>KV Cache Offload</SubHeading>
        <P>
          Checkbox to offload the entire KV cache to CPU/RAM. This is separate from weight
          offloading and adds per-token latency during generation.
        </P>

        {/* Parameters */}
        <SectionHeading id="parameters">Parameters</SectionHeading>

        <SubHeading>Sequence Length</SubHeading>
        <P>
          The maximum context window in tokens (512 to 128K). Uses a logarithmic slider for easy
          navigation across the wide range. Preset buttons for common values: 512, 2K, 4K, 8K, 32K,
          128K. Longer sequences dramatically increase KV cache memory.
        </P>

        <SubHeading>Batch Size</SubHeading>
        <P>
          Number of sequences processed simultaneously (1 to 64). Higher batch sizes improve
          throughput but multiply KV cache and activation memory. Presets: 1, 4, 8, 16, 32, 64.
        </P>

        <SubHeading>KV Cache Precision</SubHeading>
        <P>
          Separate from weight quantization. Options: FP16 (default), FP8, INT8, INT4 (most
          aggressive). Lower precision KV cache saves significant memory for long sequences with
          minimal quality impact.
        </P>

        {/* Training Configuration */}
        <SectionHeading id="training-config">Training Configuration</SectionHeading>
        <P>
          This section appears when Fine-tuning mode is active. It controls all training-specific
          VRAM factors.
        </P>

        <SubHeading>Framework Preset</SubHeading>
        <P>Pre-configured optimization profiles:</P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>None</strong> — Manual configuration
          </li>
          <li>
            <strong>DeepSpeed ZeRO-1</strong> — Partitions optimizer states across GPUs (2x memory
            savings)
          </li>
          <li>
            <strong>DeepSpeed ZeRO-2</strong> — Partitions optimizer states + gradients (4x savings)
          </li>
          <li>
            <strong>DeepSpeed ZeRO-3</strong> — Partitions everything including parameters (8-10x
            savings)
          </li>
          <li>
            <strong>Unsloth</strong> — Optimized single-GPU training with 8-bit optimizer, gradient
            checkpointing, and Flash Attention
          </li>
          <li>
            <strong>vLLM / TGI</strong> — Inference-only frameworks (automatically switches to
            inference mode)
          </li>
        </ul>

        <SubHeading>Training Method</SubHeading>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Full Fine-tuning</strong> — Updates all model parameters. Highest VRAM (weights
            + optimizer states for all params + gradients + activations).
          </li>
          <li>
            <strong>LoRA</strong> — Trains small adapter layers (~1-4% of total parameters).
            Optimizer states apply only to adapter params.
          </li>
          <li>
            <strong>QLoRA</strong> — Combines 4-bit NF4 base model + FP16 adapters + FP32 optimizer.
            Lowest VRAM for fine-tuning.
          </li>
        </ul>

        <SubHeading>Optimizer</SubHeading>
        <P>
          Determines per-parameter memory overhead: AdamW (8 bytes/param), SGD+Momentum (4
          bytes/param), 8-bit AdamW (2 bytes/param), Adafactor (4 bytes/param).
        </P>

        <SubHeading>Training Precision</SubHeading>
        <P>
          FP32 (full precision), FP16 or BF16 (mixed precision with FP32 master weights). BF16 is
          recommended for modern GPUs — it halves activation memory while maintaining training
          stability.
        </P>

        <SubHeading>Memory Optimizations</SubHeading>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Gradient Accumulation</strong> — Accumulate gradients over 1-128 micro-batches
            before updating weights. The effective batch size display shows: micro-batch x
            accumulation x GPUs.
          </li>
          <li>
            <strong>Gradient Checkpointing</strong> — Recompute activations in backward pass instead
            of storing them. Saves ~60% activation memory at the cost of 20-25% more compute.
          </li>
          <li>
            <strong>Flash Attention</strong> — Reduces attention memory from O(n^2) to O(n). Benefit
            scales with sequence length.
          </li>
          <li>
            <strong>CPU Offload Optimizer</strong> — (DeepSpeed only) Moves optimizer states to CPU
            RAM. Reduces GPU VRAM but slows training by 15-30%.
          </li>
        </ul>

        {/* Results Panel */}
        <SectionHeading id="results">Results Panel</SectionHeading>

        <SubHeading>Fit Indicator</SubHeading>
        <P>A color-coded status bar showing GPU utilization:</P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong className="text-green-600 dark:text-green-400">Green (0-80%)</strong> — Fits
            Comfortably. Enough headroom for spikes and OS overhead.
          </li>
          <li>
            <strong className="text-yellow-600 dark:text-yellow-400">Yellow (80-95%)</strong> —
            Tight Fit. May work but close to the limit. Consider reducing batch size or sequence
            length.
          </li>
          <li>
            <strong className="text-red-600 dark:text-red-400">Red (&gt;95%)</strong> — Does Not
            Fit. The model exceeds available VRAM. Follow the recommendations below.
          </li>
        </ul>

        <SubHeading>VRAM Breakdown</SubHeading>
        <P>
          A donut chart and table showing the four memory components: Model Weights (usually the
          largest), KV Cache (scales with sequence length and batch size), Activations (temporary
          computation buffers), and Framework Overhead (PyTorch + CUDA context, 500MB-1.5GB).
        </P>

        <SubHeading>Multi-GPU Breakdown</SubHeading>
        <P>
          When using multiple GPUs, shows how VRAM is distributed across devices with per-GPU bars,
          including replication overhead for embeddings and layer norms.
        </P>

        <SubHeading>Training Breakdown</SubHeading>
        <P>
          In training mode, shows: Base Model Weights, Optimizer States, Gradients, Activations,
          Framework Overhead, and (for LoRA/QLoRA) Adapter Weights.
        </P>

        <SubHeading>Performance Estimate</SubHeading>
        <P>Three metrics based on a roofline model:</P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Decode Speed</strong> — Tokens per second during generation.
          </li>
          <li>
            <strong>Time to First Token (TTFT)</strong> — Latency in milliseconds for the first
            output token (prompt processing).
          </li>
          <li>
            <strong>Bottleneck</strong> — Whether the workload is memory-bandwidth bound (yellow),
            compute bound (blue), or balanced (green).
          </li>
        </ul>

        <SubHeading>Recommendations</SubHeading>
        <P>
          When a model doesn&apos;t fit, the calculator suggests actionable steps ranked by
          effectiveness: lower quantization, reduce context length, use KV cache quantization,
          enable offloading, use multiple GPUs, or upgrade GPU.
        </P>

        {/* Comparison View */}
        <SectionHeading id="comparison">Comparison View</SectionHeading>
        <P>
          Save up to 3 configurations using the &quot;Save to Compare&quot; button in the results
          panel. Switch to the Comparison tab to see them side by side with diff highlighting
          showing what differs between configs. Each column shows the full configuration and
          results. Click &quot;Clear All&quot; to reset.
        </P>

        {/* URL Sharing */}
        <SectionHeading id="sharing">URL Sharing</SectionHeading>
        <P>
          Every configuration change is automatically encoded into the URL hash using LZ-String
          compression. Click the link icon in the header to copy the current URL to clipboard. Share
          it with colleagues — they&apos;ll see the exact same configuration when they open the
          link.
        </P>

        {/* Glossary */}
        <SectionHeading id="glossary">Glossary</SectionHeading>
        <dl className="divide-y divide-gray-100 dark:divide-gray-800">
          <GlossaryTerm term="Activation Memory">
            Temporary buffers for intermediate computation results during forward/backward passes.
            Scales with batch size, sequence length, and hidden size.
          </GlossaryTerm>
          <GlossaryTerm term="AWQ (Activation-aware Weight Quantization)">
            4-bit quantization method that preserves important weights based on activation
            magnitudes. Adds ~15-25% overhead versus raw 4-bit.
          </GlossaryTerm>
          <GlossaryTerm term="Batch Size">
            Number of input sequences processed simultaneously. Larger batches improve GPU
            utilization but multiply memory requirements.
          </GlossaryTerm>
          <GlossaryTerm term="BF16 (BFloat16)">
            16-bit floating-point format with the same exponent range as FP32 but reduced mantissa.
            Preferred for training on modern GPUs (Ampere+) because it avoids overflow issues.
          </GlossaryTerm>
          <GlossaryTerm term="Bottleneck">
            Whether inference is limited by memory bandwidth (most LLM workloads), compute (large
            batch sizes), or balanced between the two.
          </GlossaryTerm>
          <GlossaryTerm term="CPU Offload">
            Moving data (model weights, optimizer states, or KV cache) from GPU VRAM to system RAM
            via PCIe. Trades latency for capacity.
          </GlossaryTerm>
          <GlossaryTerm term="DeepSpeed ZeRO">
            Microsoft&apos;s Zero Redundancy Optimizer. Stage 1 partitions optimizer states (2x
            savings), Stage 2 adds gradients (4x), Stage 3 adds parameters (8-10x). Requires
            multiple GPUs.
          </GlossaryTerm>
          <GlossaryTerm term="Effective Batch Size">
            The true batch size considering parallelism: micro-batch size x gradient accumulation
            steps x number of GPUs.
          </GlossaryTerm>
          <GlossaryTerm term="Flash Attention">
            An efficient attention algorithm that reduces memory from O(n^2) to O(n) in sequence
            length and runs 2-4x faster. Standard on modern frameworks.
          </GlossaryTerm>
          <GlossaryTerm term="FP16 / FP32">
            16-bit and 32-bit IEEE floating-point formats. FP32 uses 4 bytes/parameter, FP16 uses 2
            bytes. Mixed precision training uses FP16 for compute with FP32 master weights.
          </GlossaryTerm>
          <GlossaryTerm term="Framework Overhead">
            Baseline GPU memory consumed by the deep learning framework (PyTorch, CUDA context,
            memory allocator). Typically 500MB-1.5GB regardless of model size.
          </GlossaryTerm>
          <GlossaryTerm term="GGUF">
            File format used by llama.cpp for quantized models. Offers formats from Q2_K (2-bit) to
            Q8_0 (8-bit) with different quality/size trade-offs.
          </GlossaryTerm>
          <GlossaryTerm term="GPTQ">
            Post-training quantization method using calibration data to minimize error. 4-bit with
            ~10-30% overhead from group quantization tables.
          </GlossaryTerm>
          <GlossaryTerm term="GQA / MQA">
            Grouped Query Attention / Multi-Query Attention. Techniques that reduce KV cache by
            sharing key-value heads across query heads (e.g., LLaMA 3 70B uses 8 KV heads for 64
            query heads = 0.125x KV reduction).
          </GlossaryTerm>
          <GlossaryTerm term="Gradient Accumulation">
            Technique to simulate larger batch sizes by accumulating gradients over multiple
            micro-batches before performing a weight update. No extra VRAM cost.
          </GlossaryTerm>
          <GlossaryTerm term="Gradient Checkpointing">
            Trades compute for memory by discarding activations during the forward pass and
            recomputing them during the backward pass. Saves ~60% activation memory.
          </GlossaryTerm>
          <GlossaryTerm term="Hidden Size">
            Dimension of the model&apos;s internal representations. Affects KV cache size and
            activation memory. Common values: 4096 (7B), 5120 (13B), 8192 (70B).
          </GlossaryTerm>
          <GlossaryTerm term="INT4 / INT8">
            4-bit and 8-bit integer quantization formats. INT8 uses 1 byte/param, INT4 uses 0.5
            bytes/param.
          </GlossaryTerm>
          <GlossaryTerm term="Interconnect">
            Communication link between GPUs. NVLink (600-900 GB/s) enables efficient tensor
            parallelism. PCIe 4.0/5.0 (32-64 GB/s) is adequate for pipeline parallelism only.
          </GlossaryTerm>
          <GlossaryTerm term="KV Cache">
            Key-Value cache stores attention states for previously processed tokens. Grows linearly
            with sequence length and batch size. Often the second-largest memory consumer after
            weights.
          </GlossaryTerm>
          <GlossaryTerm term="Layers">
            Number of transformer blocks in the model. Each layer contains attention and
            feed-forward sub-layers. More layers = more parameters and more memory.
          </GlossaryTerm>
          <GlossaryTerm term="LoRA (Low-Rank Adaptation)">
            Fine-tuning method that trains small rank-decomposed adapter matrices (~1-4% of model
            parameters) while freezing the base model. Optimizer states apply only to adapters.
          </GlossaryTerm>
          <GlossaryTerm term="Memory Bandwidth">
            Rate at which data can be read from GPU memory (GB/s). The primary bottleneck for LLM
            inference, since each generated token must read all model weights.
          </GlossaryTerm>
          <GlossaryTerm term="MoE (Mixture of Experts)">
            Architecture where each token is routed to a subset of &quot;expert&quot; sub-networks.
            Total parameter count (all experts) determines VRAM since all weights must be loaded.
            Active parameters per token are fewer.
          </GlossaryTerm>
          <GlossaryTerm term="NF4 (4-bit NormalFloat)">
            Quantization format used by QLoRA that maps to a normal distribution. 0.5 bytes/param
            with better quality preservation than uniform INT4.
          </GlossaryTerm>
          <GlossaryTerm term="NVMe Offload">
            Offloading model data to NVMe SSD storage. Slower than CPU/RAM offloading but
            effectively unlimited capacity. Useful when both VRAM and system RAM are constrained.
          </GlossaryTerm>
          <GlossaryTerm term="NVFP4 / NVFP6">
            NVIDIA-specific floating-point formats: NVFP4 (~0.5 bytes/param) and NVFP6 (~0.75
            bytes/param). Available on Hopper/Blackwell GPUs.
          </GlossaryTerm>
          <GlossaryTerm term="Optimizer States">
            Additional memory for training optimizer variables. AdamW stores first and second moment
            estimates (8 bytes/param in FP32). These are always maintained in FP32 for numerical
            stability.
          </GlossaryTerm>
          <GlossaryTerm term="Pipeline Parallel">
            Multi-GPU strategy that assigns complete model layers to different GPUs. Model is split
            into sequential stages. Works over PCIe but introduces pipeline bubbles (idle time).
          </GlossaryTerm>
          <GlossaryTerm term="QLoRA">
            Combines 4-bit NF4 base model weights, FP16 LoRA adapter weights, and FP32 optimizer
            states. Enables fine-tuning large models on consumer GPUs.
          </GlossaryTerm>
          <GlossaryTerm term="Quantization">
            Reducing the numerical precision of model weights to decrease memory usage. Common
            formats range from FP32 (4 bytes) down to 2-bit (0.25 bytes) with varying quality
            trade-offs.
          </GlossaryTerm>
          <GlossaryTerm term="Roofline Model">
            Performance analysis framework that identifies whether a workload is limited by compute
            (TFLOPS) or memory bandwidth (GB/s). Used here to estimate tokens/sec and TTFT.
          </GlossaryTerm>
          <GlossaryTerm term="Sequence Length">
            Maximum number of tokens in the context window. KV cache grows linearly with sequence
            length. Doubling the sequence length approximately doubles KV cache memory.
          </GlossaryTerm>
          <GlossaryTerm term="Sharding">
            Splitting a model across multiple devices. Includes tensor parallelism (split within
            layers) and pipeline parallelism (split across layers).
          </GlossaryTerm>
          <GlossaryTerm term="Tensor Parallel">
            Multi-GPU strategy that splits individual layers (weight matrices) horizontally across
            GPUs. Requires high-bandwidth interconnect (NVLink) due to frequent all-reduce
            operations.
          </GlossaryTerm>
          <GlossaryTerm term="TFLOPS">
            Tera floating-point operations per second. Measures GPU compute throughput. FP16 TFLOPS
            is most relevant for LLM inference.
          </GlossaryTerm>
          <GlossaryTerm term="Tokens/sec">
            Decode speed — the rate at which new tokens are generated during inference. Primarily
            limited by memory bandwidth for single-batch inference.
          </GlossaryTerm>
          <GlossaryTerm term="TTFT (Time to First Token)">
            Latency from prompt submission to the first generated token. Involves processing the
            entire prompt (prefill phase), which is compute-bound for large prompts.
          </GlossaryTerm>
          <GlossaryTerm term="VRAM">
            Video Random Access Memory — the high-bandwidth memory on a GPU. All model weights, KV
            cache, activations, and framework overhead must fit in VRAM (or be offloaded) for the
            model to run.
          </GlossaryTerm>
        </dl>
      </div>
    </div>
  )
}
