import { useState } from 'react'

const SECTIONS = [
  { id: 'quick-start', label: 'Quick Start' },
  { id: 'whats-new', label: "What's new in 2.0" },
  { id: 'mode-toggle', label: 'Mode Toggle' },
  { id: 'model-config', label: 'Model Configuration' },
  { id: 'gpu-selection', label: 'GPU Selection' },
  { id: 'hardware-config', label: 'Hardware Configuration' },
  { id: 'offloading', label: 'Offloading' },
  { id: 'parameters', label: 'Parameters' },
  { id: 'training-config', label: 'Training Configuration' },
  { id: 'results', label: 'Results Panel' },
  { id: 'why-changed', label: 'Why a setting changed' },
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
            <strong>Pick a model</strong> — Search or scroll the curated list of 54 models, or
            define a custom model with your own parameter count and architecture details.
          </li>
          <li>
            <strong>Pick a GPU</strong> — Choose from 29 GPUs across NVIDIA, AMD, and Apple Silicon,
            or enter custom hardware specs.
          </li>
          <li>
            <strong>Read the results</strong> — The right panel instantly shows whether the model
            fits, a VRAM breakdown, and performance estimates. If it doesn&apos;t fit, follow the
            recommendations.
          </li>
        </ol>

        {/* What's new in 2.0 */}
        <SectionHeading id="whats-new">What&apos;s new in 2.0</SectionHeading>
        <P>
          <strong>Layout.</strong> The essential inputs stay visible: mode, model, format, GPU, GPUs
          per replica, servers, context and concurrent users. Everything else sits under{' '}
          <strong>Advanced</strong>, which opens by itself and shows &quot;N settings changed&quot;
          whenever one of its values differs from the default. Results lead with a verdict (does it
          fit, decode speed, first-token delay, sessions); charts and tables are under{' '}
          <strong>Details</strong>. The PDF export opens every section before it captures.
        </P>
        <P>
          <strong>Rules.</strong> A combination that cannot run is no longer offered, and a change
          that makes another setting impossible corrects that setting and says why in one notice
          (&quot;Adjusted for …&quot;). A shared link that encodes an impossible combination opens
          corrected, with a &quot;Shared link adjusted&quot; notice.
        </P>
        <P>
          <strong>GPU data, per card.</strong> H100 PCIe and A100 PCIe pair cards over a 2-card
          NVLink bridge (600 GB/s); larger groups cross PCIe 5 / PCIe 4, and H100 PCIe FP16 is the
          dense 756 TFLOPS. A100 SXM uses NVLink 3 (600 GB/s). The H200 entry is the SXM product
          (HGX 4 or 8); the new H200 NVL entry is the PCIe card with a 4-way bridge at 900 GB/s. DGX
          Spark (GB10) is one GPU per unit: cluster Sparks as servers over 200 GbE. DGX Station is a
          single GPU. L40S and RTX 6000 Ada are PCIe 4. Apple Silicon and GB10 are marked unified
          memory.
        </P>
        <P>
          <strong>Multi-server.</strong> Prefill and decode across servers are now computed from the
          bytes crossing the network, not from an efficiency curve. Small clusters get much faster
          first tokens; a single request over many HGX servers gets slower (it walks the servers one
          after another).
        </P>
        <P>
          <strong>Existing links.</strong> They open with the corrected values and a notice. Numbers
          change for the cards above and for most multi-server setups. An old H200 link opens as
          H200 SXM.
        </P>

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
          A searchable dropdown with 54 curated models (LLaMA, Mistral, Qwen, DeepSeek, etc.). Each
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
        <P>
          Where a published checkpoint exists for the chosen format, the weight size is measured
          from it (the results name the repository); otherwise it is estimated from the
          format&apos;s bytes per parameter.
        </P>

        {/* GPU Selection */}
        <SectionHeading id="gpu-selection">GPU Selection</SectionHeading>
        <P>
          A searchable dropdown grouped by manufacturer (NVIDIA, AMD, Apple). Each GPU shows its
          VRAM capacity and tier badge (DC = Datacenter, Consumer, Apple).
        </P>
        <P>
          <strong>Key specs that affect calculations:</strong> VRAM determines fit/no-fit. Memory
          bandwidth determines decode speed. FP16 TFLOPS (dense) determines prefill. The scale-up
          link sets multi-GPU cost: NVSwitch baseboards connect 4 or 8 GPUs (HGX), an NVL72 rack up
          to 72; NVLink bridges pair 2 PCIe cards (H100/A100 PCIe) or up to 4 (H200 NVL), and a
          larger group crosses PCIe. Unified-memory parts (Apple Silicon, DGX Spark) have no
          separate host memory.
        </P>
        <P>
          Select &quot;Custom GPU...&quot; to specify any hardware with name, VRAM (required),
          bandwidth (optional), and FP16 TFLOPS (optional).
        </P>

        {/* Hardware Configuration */}
        <SectionHeading id="hardware-config">Hardware Configuration</SectionHeading>
        <P>This section appears after selecting a GPU.</P>

        <SubHeading>GPUs per replica (in one server)</SubHeading>
        <P>
          The parallel degree of one model replica inside one server, capped by what the part forms
          in one node. Under tensor parallelism only degrees vLLM accepts are offered (they divide
          the attention heads; most models allow 1, 2, 4 and 8). With several servers the summary
          reads &quot;8 GPUs per server × 2 servers per replica&quot;. An 8-GPU server running four
          2-GPU replicas is sized as one 2-GPU replica. In fine-tuning mode (with a DeepSpeed ZeRO
          preset) the same control is labelled &quot;Number of GPUs&quot;.
        </P>

        <SubHeading>Sharding Strategy</SubHeading>
        <P>
          Visible whenever the GPU forms multi-GPU servers, so pipeline parallel can be chosen
          before picking a count tensor parallel cannot run. Options:
        </P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Tensor Parallel</strong> — Splits each layer horizontally across GPUs. Requires
            fast interconnect (NVLink). Best for single-node setups.
          </li>
          <li>
            <strong>Pipeline Parallel</strong> — Assigns complete layers to different GPUs. Works
            over slower PCIe. Each GPU stores only its own layers&apos; KV cache, but decode gains
            nothing at batch 1: stages run one after another until the batch fills the pipeline.
          </li>
          <li>
            <strong>Expert Parallel + DP attention</strong> — MoE models only. Splits the experts
            across GPUs and replicates attention, so each GPU serves its own sessions and MLA KV
            cache is not duplicated. Each MoE layer pays an all-to-all over the interconnect. Where
            a measured base/expert weight split is available, replicated-base memory and MoE decode
            price the base and the experts at their own measured rates instead of one blended
            average.
          </li>
        </ul>
        <P>
          A colored badge shows the detected interconnect and its bandwidth. A bridged card shows
          &quot;NVLink bridge&quot; only while the group fits the bridge; beyond it the badge and
          the maths both use PCIe. Green = NVLink (excellent), Yellow = PCIe (adequate), Red = none
          (multi-GPU may not work). A warning appears if the tensor parallel degree exceeds the
          recommended maximum for the interconnect.
        </P>

        {/* Offloading */}
        <SectionHeading id="offloading">Offloading</SectionHeading>
        <P>
          When VRAM is insufficient, offload parts of the model to system memory or storage. Enable
          the toggle to reveal offloading options. Every decode step reads the offloaded weights
          and/or KV cache over the host link (PCIe, NVMe, or Grace NVLink-C2C on a Grace-host GPU)
          instead of GPU memory; the results show the resulting slowdown as a computed ratio (e.g.
          &quot;≈ 31× slower decode than all-in-GPU&quot;), not a fixed range.
        </P>

        <SubHeading>Offload Target</SubHeading>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>CPU/RAM</strong> — Offload via PCIe (or Grace NVLink-C2C) to system memory.
          </li>
          <li>
            <strong>NVMe SSD</strong> — Offload to storage. Slower than CPU/RAM but useful when
            system memory is also limited.
          </li>
        </ul>
        <P>
          On unified-memory parts (Apple Silicon, DGX Spark) only NVMe is offered: RAM is already
          the GPU&apos;s memory, so there is nothing to offload into.
        </P>

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

        <SubHeading>Host Capacity</SubHeading>
        <P>
          &quot;Host capacity per server (GB)&quot; checks the offloaded memory against what the
          host(s) can actually hold, defaulting to a per-tier/target estimate (e.g. 2048 GB cpu-ram
          / 30720 GB nvme on datacenter GPUs); exceeding it across all servers is flagged as not
          fitting, even when the on-device share fits the GPU.
        </P>

        {/* Parameters */}
        <SectionHeading id="parameters">Parameters</SectionHeading>

        <SubHeading>Sequence Length</SubHeading>
        <P>
          The maximum context window in tokens, up to 1M (and beyond, for a model whose native
          context is larger). Uses a logarithmic slider for easy navigation across the wide range.
          Preset buttons for common values: 512, 2K, 4K, 8K, 32K, 128K, 256K, 512K, 1M. A marker on
          the slider shows the selected model&apos;s native context; going past it doesn&apos;t
          clamp the value, but the calculator warns that it requires RoPE scaling. Longer sequences
          dramatically increase KV cache memory.
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
        <P>GB here means GiB (1024³ bytes), as nvidia-smi reports.</P>

        <SubHeading>Verdict and details</SubHeading>
        <P>
          The verdict block answers first: whether the configuration fits, decode speed, the
          first-token figure and how many sessions fit. Warnings stay visible under it. Charts, the
          memory table, the multi-GPU split, the weight source, the KV tier summary and per-user
          metrics are under Details. Export PDF opens every collapsed section for the capture.
        </P>
        <SubHeading>First token with a batch (burst)</SubHeading>
        <P>
          At batch size 1 the figure is the time to the first token. At batch size B above 1 it is
          labelled &quot;Prefill per request (amortized over batch B)&quot;: the prefill of a burst
          of B prompts divided by B. In a real burst the first request answers sooner and the last
          one waits about B times longer. Across servers the prompt also crosses the network once
          per server boundary.
        </P>

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
          including replicated layer norms and NCCL communication buffers.
        </P>

        <SubHeading>Training Breakdown</SubHeading>
        <P>
          In training mode, shows: Base Model Weights, Optimizer States, Gradients, Activations,
          Framework Overhead, and (for LoRA/QLoRA) Adapter Weights.
        </P>

        <SubHeading>Performance Estimate</SubHeading>
        <P>Four metrics based on a roofline model:</P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>Decode Speed</strong> — Tokens per second during generation. Each step reads the
            weights once plus every session&apos;s KV cache, so long contexts and large batches slow
            it down; tensor parallelism adds two all-reduces per layer.
          </li>
          <li>
            <strong>Max concurrent sessions</strong> — How many sessions fit at the chosen context:
            the memory left on each GPU after weights and overhead (at 90%, vLLM&apos;s default)
            divided by one session&apos;s KV cache. Shown in red when the configured concurrent
            users exceed it.
          </li>
          <li>
            <strong>KV storage tier</strong> — Idle sessions park their KV cache off the GPU (host
            memory, NVMe or network storage) and reload it when they resume. Sessions held =
            sessions that fit ÷ active share. Resume vs recompute shows which is faster: short
            prompts recompute faster. Tier traffic turns red when it exceeds the tier bandwidth.
            Cluster-scale storage such as Dell Lightning FS (16K+ GPUs) is sized in the storage
            calculator, not here.
          </li>
          <li>
            <strong>Time to First Token (TTFT)</strong> — Latency for the first output token.
            Dominated by prompt processing (prefill), so it grows with prompt length: past a
            crossover the quadratic attention term overtakes the linear weight term and TTFT rises
            faster than the prompt does. The crossover is model-dependent — roughly active
            parameters divided by (layers x hidden size), which is about 107K tokens for Llama 3 70B
            but only about 37K for Qwen3.6 35B A3B.
          </li>
          <li>
            <strong>Bottleneck</strong> — Whether the workload is memory-bandwidth bound (yellow),
            compute bound (blue), or balanced (green).
          </li>
          <li>
            <strong>Prompt Processing</strong> — The prefill time behind TTFT, labelled by whichever
            term dominates: weight-dominated (linear) for shorter prompts, attention-dominated
            (quadratic) once the prompt is long enough.
          </li>
        </ul>

        <SubHeading>Recommendations</SubHeading>
        <P>
          When a model doesn&apos;t fit, the calculator suggests actionable steps ranked by
          effectiveness: lower quantization, reduce context length, use KV cache quantization,
          enable offloading, use multiple GPUs, or upgrade GPU.
        </P>

        {/* Why a setting changed */}
        <SectionHeading id="why-changed">Why a setting changed</SectionHeading>
        <P>
          When a change makes another setting impossible, the calculator corrects it and shows one
          notice listing every correction. The rules:
        </P>
        <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 space-y-1 mb-3">
          <li>
            <strong>GPU count above what the part forms in one server</strong> is capped (8 on an
            HGX baseboard, 72 on an NVL72 rack, 1 on a DGX Spark or a Mac).
          </li>
          <li>
            <strong>Tensor-parallel degree vLLM cannot run</strong>: the GPU count must divide the
            model&apos;s attention heads (and match its KV heads). Llama 3.1 70B runs on 1, 2, 4 or
            8 GPUs; asking for 6 sets 4. Choose pipeline parallel to use 6.
          </li>
          <li>
            <strong>Expert parallel on a dense model</strong> switches to tensor parallel.
          </li>
          <li>
            <strong>Unified memory</strong> (Apple Silicon, DGX Spark): RAM is the GPU&apos;s
            memory, so CPU-RAM offload, host-memory KV tiers and CPU optimizer offload are turned
            off. NVMe offload stays available.
          </li>
          <li>
            <strong>Grace host KV tier</strong> needs a Grace host (GB300 NVL72, DGX Station).
          </li>
          <li>
            <strong>KV cache offload</strong> already keeps all KV off the GPU, so a KV tier is
            turned off.
          </li>
          <li>
            <strong>Interconnect variant</strong> not offered on the selected GPU resets to its
            default.
          </li>
          <li>
            <strong>vLLM and TGI</strong> are inference engines: switching to fine-tuning clears
            them. CPU optimizer offload needs a DeepSpeed ZeRO preset.
          </li>
          <li>
            <strong>Out-of-range numbers</strong> (offloaded layers beyond the model, batch 0,
            context below 512, more than 8 servers) are brought into range.
          </li>
        </ul>
        <P>
          Warnings that keep your value: context beyond the model&apos;s native length, a
          tensor-parallel degree above what the interconnect scales to, small clusters of single-GPU
          units, experts that do not split evenly across the GPUs, more pipeline stages than layers.
        </P>
        <P>
          <strong>Resetting.</strong> &quot;Reset advanced settings&quot;, inside the Advanced
          section, puts batch size, KV precision, sharding strategy, fabric, interconnect variant,
          offloading and the KV tier back to their defaults; model, GPU, GPU count, servers, format,
          context and concurrent users are untouched. &quot;Reset all&quot;, the icon button in the
          header, returns to the empty starting state and clears the shared-link URL. Both show one
          &quot;Reset to defaults&quot; notice listing what changed.
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
          Every configuration change is encoded into the URL hash using LZ-String compression. Click
          the link icon in the header to copy the current URL. A link restores the whole
          configuration in one step, including the fine-tuning settings (gradient accumulation,
          gradient checkpointing, Flash Attention, framework preset, CPU optimizer offload) and the
          interconnect variant. If the link encodes a combination that is no longer allowed (or a
          GPU whose data was corrected), it opens corrected with a &quot;Shared link adjusted&quot;
          notice explaining what changed.
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
            (TFLOPS) or memory bandwidth (GB/s). Used here to estimate decode tokens/sec, and
            separately to estimate prefill (prompt processing) time and TTFT.
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
            Latency from prompt submission to the first generated token, dominated by processing the
            entire prompt (prefill phase). TTFT grows with prompt length: a linear weight-processing
            term dominates short prompts, but past a crossover of roughly active parameters divided
            by (layers x hidden size) the quadratic causal-attention term takes over and TTFT rises
            faster than the prompt does. That crossover ranges from about 37K tokens (Qwen3.6 35B
            A3B) to about 107K (Llama 3 70B) across the model database.
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
