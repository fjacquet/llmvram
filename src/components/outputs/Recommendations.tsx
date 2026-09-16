import type {
  InferenceVRAMBreakdown,
  MultiGPUVRAMBreakdown,
  QuantizationFormat,
} from '@engines/types'
import { useUIStore } from '@store/uiStore'
import { clampGPUCount } from '@utils/gpuLimits'
import type { GPU } from '@/types/gpu'

interface RecommendationsProps {
  gpu: GPU
  breakdown: InferenceVRAMBreakdown
  currentQuantization: QuantizationFormat
  currentSequenceLength: number
  numGPUs?: number
  multiGPUBreakdown?: MultiGPUVRAMBreakdown | null
}

interface Recommendation {
  title: string
  description: string
  impact: string
}

export function Recommendations({
  gpu,
  breakdown,
  currentQuantization,
  currentSequenceLength,
  numGPUs = 1,
  multiGPUBreakdown,
}: RecommendationsProps) {
  const offloadingEnabled = useUIStore((s) => s.offloadingEnabled)

  const deficit = breakdown.total.minus(gpu.vram_gb).toNumber()

  // If model fits, don't show recommendations
  if (deficit <= 0) {
    return null
  }

  const recommendations: Recommendation[] = []

  // 1. Lower quantization (if not already at minimum)
  const quantizationOrder: QuantizationFormat[] = [
    'fp32',
    'fp16',
    'bf16',
    'nvfp6',
    'int8',
    'nvfp4',
    'gptq',
    'awq',
    'int4',
    'nf4',
    'gguf-q8_0',
    'gguf-q6_k',
    'gguf-q5_k_s',
    'gguf-q5_k_m',
    'gguf-q5_0',
    'gguf-q4_k_s',
    'gguf-q4_k_m',
    'gguf-q4_0',
    'gguf-q3_k_l',
    'gguf-q3_k_m',
    'gguf-q3_k_s',
    'gguf-q2_k',
  ]

  const currentIndex = quantizationOrder.indexOf(currentQuantization)
  if (currentIndex < quantizationOrder.length - 1) {
    const modelWeightsGB = breakdown.modelWeights.toNumber()

    // Estimate savings based on current format
    let suggestedFormat = ''
    let estimatedSavings = 0

    if (
      currentQuantization === 'fp32' ||
      currentQuantization === 'fp16' ||
      currentQuantization === 'bf16'
    ) {
      suggestedFormat = 'GPTQ or AWQ 4-bit'
      estimatedSavings = modelWeightsGB * 0.5 // ~50% savings from fp16 to 4-bit
    } else if (currentQuantization === 'int8') {
      suggestedFormat = 'INT4 or NF4'
      estimatedSavings = modelWeightsGB * 0.35 // ~35% savings from int8 to int4
    } else if (currentQuantization.startsWith('gptq') || currentQuantization.startsWith('awq')) {
      suggestedFormat = 'GGUF Q4_0'
      estimatedSavings = modelWeightsGB * 0.15 // ~15% savings from GPTQ to GGUF Q4_0
    } else if (!currentQuantization.includes('q4_0')) {
      suggestedFormat = 'GGUF Q4_0 (most compressed)'
      estimatedSavings = modelWeightsGB * 0.1 // ~10% additional savings
    }

    if (suggestedFormat) {
      recommendations.push({
        title: 'Lower quantization',
        description: `Try ${suggestedFormat}`,
        impact: `Could save ~${estimatedSavings.toFixed(1)} GB`,
      })
    }
  }

  // 2. Reduce context length (if current > 512)
  if (currentSequenceLength > 512) {
    const reducedLength = Math.max(512, Math.floor(currentSequenceLength / 2))
    const kvCacheGB = breakdown.kvCache.toNumber()
    const estimatedSavings = kvCacheGB * 0.5 // Halving context approximately halves KV cache

    recommendations.push({
      title: 'Reduce context length',
      description: `From ${currentSequenceLength} to ${reducedLength} tokens`,
      impact: `Could save ~${estimatedSavings.toFixed(1)} GB`,
    })
  }

  // 3. Use KV cache quantization (only if current is fp16)
  if (currentQuantization === 'fp16' || currentQuantization === 'bf16') {
    const kvCacheGB = breakdown.kvCache.toNumber()
    const estimatedSavings = kvCacheGB * 0.75 // INT4 KV cache saves ~75%

    recommendations.push({
      title: 'Use KV cache quantization',
      description: 'Reduce KV cache precision to INT4',
      impact: `Could save ~${estimatedSavings.toFixed(1)} GB`,
    })
  }

  // 4. Enable offloading to CPU/RAM or NVMe (only if not already enabled)
  if (!offloadingEnabled) {
    const offloadGB = deficit
    if (offloadGB > 0) {
      const offloadTarget = offloadGB <= 32 ? 'CPU/RAM' : 'CPU/RAM + NVMe SSD'
      const perfNote =
        offloadGB <= 8 ? '~2-5x slower' : offloadGB <= 24 ? '~5-15x slower' : '~15-50x slower'

      recommendations.push({
        title: `Enable offloading to ${offloadTarget}`,
        description: `Offload ~${offloadGB.toFixed(1)} GB of layers to system memory`,
        impact: `Enables running but ${perfNote} due to PCIe bandwidth`,
      })
    }
  }

  // 5. Use multiple GPUs
  const totalGB = breakdown.total.toNumber()

  // Effective scaling efficiency: use actual value from breakdown, or conservative default
  const scalingEfficiency = multiGPUBreakdown?.scalingEfficiency ?? 0.85

  // Case A: Multi-GPU active and still doesn't fit -> suggest more GPUs
  if (numGPUs > 1 && multiGPUBreakdown && multiGPUBreakdown.totalPerGPU.greaterThan(gpu.vram_gb)) {
    // numGPUs here is the CLUSTER TOTAL (ResultsPanel passes multiGPU.numGPUs),
    // so the ceiling is the whole cluster's, not one node's: max_gpus_per_node
    // GPUs in each of the nodes already configured. Clamping to the per-node
    // bound instead would advise cutting a 32-GPU cluster down to 8.
    const numNodes = multiGPUBreakdown.numNodes || 1
    const clusterCeiling = gpu.max_gpus_per_node * numNodes
    const rawNeeded = Math.max(1, Math.ceil(totalGB / (gpu.vram_gb * scalingEfficiency)))
    const gpusNeeded = Math.min(rawNeeded, clusterCeiling)

    if (gpusNeeded > numGPUs) {
      // Room left inside the nodes already configured.
      recommendations.push({
        title: 'Add more GPUs',
        description: `Current ${numGPUs}x still exceeds capacity. Try ${gpusNeeded}x ${gpu.name} with tensor parallelism`,
        impact: `Would distribute ${totalGB.toFixed(1)} GB across ${gpusNeeded} GPUs (~${(totalGB / gpusNeeded).toFixed(1)} GB per GPU)`,
      })
    } else if (rawNeeded > clusterCeiling) {
      // Every node is full and the model still does not fit, so GPUs per node is
      // no longer the free variable — node count is. Without this the panel says
      // "Need N GB more VRAM. Try:" and then lists nothing, since the "upgrade
      // GPU" recommendation below only fires under 80 GB and says nothing to
      // someone already on a 288 GB part.
      const nodesNeeded = Math.ceil(rawNeeded / gpu.max_gpus_per_node)
      recommendations.push({
        title: 'Add more servers',
        description: `${numNodes} server${numNodes === 1 ? '' : 's'} of ${gpu.max_gpus_per_node}x ${gpu.name} is already the most GPUs one server holds. Try ${nodesNeeded} servers (${nodesNeeded * gpu.max_gpus_per_node} GPUs total)`,
        impact: `Would distribute ${totalGB.toFixed(1)} GB across ${nodesNeeded * gpu.max_gpus_per_node} GPUs (~${(totalGB / (nodesNeeded * gpu.max_gpus_per_node)).toFixed(1)} GB per GPU), with pipeline parallelism between servers`,
      })
    }
  } else if (numGPUs === 1) {
    // Case B: Single GPU -> show multi-GPU recommendation, clamped to what the
    // selected part can actually form. When max_gpus_per_node is 1 (Apple
    // Silicon, GB300 Desktop Superchip) the clamp always lands back at 1x, so
    // gpusNeeded > 1 is false and this recommendation is silently suppressed —
    // falling through to the "upgrade GPU" recommendation below instead of
    // suggesting a configuration the UI has no way to build.
    const gpusNeeded = clampGPUCount(Math.ceil(totalGB / (gpu.vram_gb * scalingEfficiency)), gpu)

    if (gpusNeeded > 1) {
      recommendations.push({
        title: 'Use multiple GPUs',
        description: `Use ${gpusNeeded}x ${gpu.name} with tensor parallelism`,
        impact: `Would distribute ${totalGB.toFixed(1)} GB across ${gpusNeeded} GPUs (~${(totalGB / gpusNeeded).toFixed(1)} GB per GPU)`,
      })
    }
  }
  // Case C: Multi-GPU active and fits -> don't show recommendation (already solved)

  // 6. Upgrade GPU (show if a larger common GPU exists)
  const vramGB = gpu.vram_gb
  let upgradeRecommendation = ''

  if (vramGB < 24 && totalGB <= 32) {
    upgradeRecommendation = 'RTX 4090 (24 GB) or RTX 5090 (32 GB)'
  } else if (vramGB < 48 && totalGB <= 48) {
    upgradeRecommendation = 'A6000 (48 GB)'
  } else if (vramGB < 80 && totalGB <= 80) {
    upgradeRecommendation = 'A100 or H100 (80 GB)'
  }

  if (upgradeRecommendation) {
    recommendations.push({
      title: 'Upgrade GPU',
      description: `Switch to ${upgradeRecommendation}`,
      impact: 'Would fit the model comfortably',
    })
  }

  return (
    <aside
      aria-label="Optimization recommendations"
      className="rounded-lg border-2 border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-6"
    >
      <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
        Recommendations
      </h3>
      <p className="text-sm text-blue-800 dark:text-blue-200 mb-4">
        Need {deficit.toFixed(2)} GB more VRAM. Try:
      </p>

      <ol className="space-y-3">
        {recommendations.map((rec, index) => (
          <li key={rec.title} className="text-sm">
            <div className="flex gap-3">
              <span className="font-bold text-blue-900 dark:text-blue-100 flex-shrink-0">
                {index + 1}.
              </span>
              <div>
                <div className="font-semibold text-blue-900 dark:text-blue-100">{rec.title}</div>
                <div className="text-blue-800 dark:text-blue-200">
                  {rec.description} — {rec.impact}
                </div>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </aside>
  )
}
