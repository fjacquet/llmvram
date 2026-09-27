import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import { type GPU, type Model, validateGPUs, validateModels } from '@utils/schemas'
import { compressToEncodedURIComponent } from 'lz-string'
import { describe, expect, it } from 'vitest'
import { deserializeFromURL, isCustomId, serializeToURL, urlStateToConfig } from './urlSerializer'

const baseState = {
  selectedModel: {
    id: 'meta-llama-llama-3-70b',
    name: 'Llama 3 70B',
    architecture: 'dense' as const,
    num_parameters_billion: 70,
    hidden_size: 8192,
    num_hidden_layers: 80,
    num_attention_heads: 64,
    num_kv_heads: 8,
    intermediate_size: 28672,
  },
  selectedGPU: {
    id: 'nvidia-h100-80gb-sxm',
    name: 'NVIDIA H100 80GB SXM',
    manufacturer: 'nvidia' as const,
    vram_gb: 80,
    memory_bandwidth_gbps: 3352,
    memory_type: 'HBM3',
    bus_width: 5120,
    fp16_tflops: 1979,
    fp32_tflops: 989,
    tier: 'datacenter' as const,
    interconnect: 'nvlink-4' as const,
    max_gpus_per_node: 8,
  },
  quantization: 'gptq' as const,
  sequenceLength: 4096,
  batchSize: 1,
  kvQuantization: 'fp16' as const,
  numGPUs: 2,
  shardingStrategy: 'tensor-parallel' as const,
  offloadingEnabled: false,
  offloadTarget: 'cpu-ram' as const,
  offloadMode: 'percentage' as const,
  offloadPercentage: 0,
  offloadLayers: 0,
  kvCacheOffload: false,
  offloadHostCapacityGB: null,
  mode: 'inference' as const,
  trainingMethod: 'lora' as const,
  optimizer: 'adamw' as const,
  trainingPrecision: 'bf16' as const,
  loraRank: 16,
  loraAlpha: 32,
  targetModulesPercent: 30,
  gradientAccumulationSteps: 1,
  gradientCheckpointing: false,
  flashAttention: false,
  numNodes: 1,
  interNodeFabric: 'ethernet-800g' as const,
  customFabric: null,
  concurrentUsers: 1,
  kvTier: {
    tier: 'none' as const,
    customGBps: null,
    activeShare: 0.25,
    burstSeconds: 30,
    capacityTB: null,
  },
  frameworkPreset: 'none' as const,
  cpuOffloadOptimizer: false,
  interconnectOverride: null,
}

describe('URL Serializer', () => {
  describe('isCustomId', () => {
    it('should correctly identify custom IDs', () => {
      expect(isCustomId('custom-1234')).toBe(true)
      expect(isCustomId('custom-restored')).toBe(true)
      expect(isCustomId('nvidia-h100-80gb-sxm')).toBe(false)
      expect(isCustomId('meta-llama-llama-3-70b')).toBe(false)
      expect(isCustomId('')).toBe(false)
    })
  })

  describe('serializeToURL and deserializeFromURL', () => {
    it('round-trips concurrent users and the KV tier', () => {
      const serialized = serializeToURL({
        ...baseState,
        concurrentUsers: 2500,
        kvTier: {
          tier: 'network',
          customGBps: 20,
          activeShare: 0.1,
          burstSeconds: 45,
          capacityTB: 500,
        },
      })
      const d = deserializeFromURL(serialized)
      expect(d?.cu).toBe(2500)
      expect(d?.kt).toEqual({ t: 'network', g: 20, a: 0.1, b: 45, c: 500 })
    })

    it('omits the tier when it is none and users when there is one', () => {
      const d = deserializeFromURL(
        serializeToURL({
          ...baseState,
          concurrentUsers: 1,
          kvTier: {
            tier: 'none',
            customGBps: null,
            activeShare: 0.25,
            burstSeconds: 30,
            capacityTB: null,
          },
        }),
      )
      expect(d?.cu).toBeUndefined()
      expect(d?.kt).toBeUndefined()
    })

    it('still parses links made before these keys existed', () => {
      const d = deserializeFromURL(serializeToURL(baseState))
      expect(d).not.toBeNull()
      expect(d?.cu).toBeUndefined()
      expect(d?.kt).toBeUndefined()
    })

    it('should round-trip curated model and GPU', () => {
      const serialized = serializeToURL(baseState)
      expect(serialized).toBeTypeOf('string')
      expect(serialized.length).toBeGreaterThan(0)

      const deserialized = deserializeFromURL(serialized)
      expect(deserialized).not.toBeNull()
      expect(deserialized?.modelId).toBe('meta-llama-llama-3-70b')
      expect(deserialized?.gpuId).toBe('nvidia-h100-80gb-sxm')
      expect(deserialized?.q).toBe('gptq')
      expect(deserialized?.sl).toBe(4096)
      expect(deserialized?.bs).toBe(1)
      expect(deserialized?.kvq).toBe('fp16')
      expect(deserialized?.ng).toBe(2)
      expect(deserialized?.ss).toBe('tensor-parallel')
      expect(deserialized?.oe).toBeUndefined() // offloading disabled
      expect(deserialized?.m).toBeUndefined() // inference mode - not serialized
    })

    it('should round-trip custom model', () => {
      const state = {
        ...baseState,
        selectedModel: {
          id: 'custom-12345',
          name: 'My Custom Model',
          architecture: 'dense' as const,
          num_parameters_billion: 7,
          hidden_size: 4096,
          num_hidden_layers: 32,
          num_attention_heads: 32,
          num_kv_heads: 8,
          intermediate_size: 11008,
        },
        selectedGPU: {
          id: 'nvidia-rtx-4090',
          name: 'NVIDIA RTX 4090',
          manufacturer: 'nvidia' as const,
          vram_gb: 24,
          memory_bandwidth_gbps: 1008,
          memory_type: 'GDDR6X',
          bus_width: 384,
          fp16_tflops: 82.6,
          tier: 'consumer' as const,
          interconnect: 'none' as const,
          max_gpus_per_node: 8,
        },
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.modelId).toBeUndefined()
      expect(deserialized?.customModel).toBeDefined()
      expect(deserialized?.customModel?.name).toBe('My Custom Model')
      expect(deserialized?.customModel?.num_parameters_billion).toBe(7)
      expect(deserialized?.customModel?.hidden_size).toBe(4096)
      expect(deserialized?.customModel?.num_hidden_layers).toBe(32)
      expect(deserialized?.customModel?.num_attention_heads).toBe(32)
      expect(deserialized?.customModel?.num_kv_heads).toBe(8)
      expect(deserialized?.customModel?.intermediate_size).toBe(11008)
    })

    it('should round-trip custom GPU', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: {
          id: 'custom-67890',
          name: 'Custom GPU',
          manufacturer: 'nvidia' as const,
          vram_gb: 16,
          memory_bandwidth_gbps: 512,
          memory_type: 'GDDR6',
          bus_width: 256,
          fp16_tflops: 50,
          tier: 'consumer' as const,
          interconnect: 'none' as const,
          max_gpus_per_node: 8,
        },
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.gpuId).toBeUndefined()
      expect(deserialized?.customGPU).toBeDefined()
      expect(deserialized?.customGPU?.name).toBe('Custom GPU')
      expect(deserialized?.customGPU?.vram_gb).toBe(16)
      expect(deserialized?.customGPU?.memory_bandwidth_gbps).toBe(512)
      expect(deserialized?.customGPU?.fp16_tflops).toBe(50)
    })

    it('should serialize offloading parameters when enabled', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
        offloadingEnabled: true,
        offloadTarget: 'nvme' as const,
        offloadMode: 'layers' as const,
        offloadPercentage: 50,
        offloadLayers: 20,
        kvCacheOffload: true,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.oe).toBe(true)
      expect(deserialized?.ot).toBe('nvme')
      expect(deserialized?.om).toBe('layers')
      expect(deserialized?.op).toBe(50)
      expect(deserialized?.ol).toBe(20)
      expect(deserialized?.ko).toBe(true)
      expect(deserialized?.hc).toBeUndefined()
    })

    it('should round-trip the offload host capacity override alongside offloading', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
        offloadingEnabled: true,
        offloadTarget: 'cpu-ram' as const,
        offloadMode: 'percentage' as const,
        offloadPercentage: 100,
        offloadLayers: 0,
        kvCacheOffload: true,
        offloadHostCapacityGB: 4096,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.hc).toBe(4096)
    })

    it('should omit the host capacity override when offloading is disabled', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
        offloadHostCapacityGB: 4096,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized?.hc).toBeUndefined()
    })

    it('tolerates an invalid hand-edited hc (0) instead of discarding the whole hash', () => {
      // A hand-edited hash with hc=0 (or any non-positive value) must not fail the whole
      // schema. hc now passes through raw; config-rules R4 resets it to the default
      // (null) after restore, with a "Shared link adjusted" notice, instead of the
      // schema silently dropping it.
      const hash = compressToEncodedURIComponent(
        JSON.stringify({
          modelId: 'meta-llama-llama-3-70b',
          gpuId: 'nvidia-h100-80gb-sxm',
          q: 'gptq',
          sl: 4096,
          bs: 1,
          kvq: 'fp16',
          ng: 4,
          ss: 'tensor-parallel',
          oe: true,
          ot: 'cpu-ram',
          om: 'percentage',
          op: 100,
          ol: 0,
          ko: true,
          hc: 0,
        }),
      )

      const decoded = deserializeFromURL(hash)

      expect(decoded).not.toBeNull()
      expect(decoded?.modelId).toBe('meta-llama-llama-3-70b')
      expect(decoded?.oe).toBe(true)
      expect(decoded?.hc).toBe(0)
    })

    it('should NOT serialize training fields when mode is inference', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      // Training fields should NOT be in URL
      expect(deserialized?.m).toBeUndefined()
      expect(deserialized?.tm).toBeUndefined()
      expect(deserialized?.to).toBeUndefined()
      expect(deserialized?.tp).toBeUndefined()
      expect(deserialized?.lr).toBeUndefined()
      expect(deserialized?.la).toBeUndefined()
      expect(deserialized?.tmp).toBeUndefined()
    })

    it('should serialize training fields when mode is training', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
        mode: 'training' as const,
        trainingMethod: 'qlora' as const,
        optimizer: 'sgd-momentum' as const,
        trainingPrecision: 'fp16' as const,
        loraRank: 32,
        loraAlpha: 64,
        targetModulesPercent: 50,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      // Training fields SHOULD be in URL
      expect(deserialized?.m).toBe('training')
      expect(deserialized?.tm).toBe('qlora')
      expect(deserialized?.to).toBe('sgd-momentum')
      expect(deserialized?.tp).toBe('fp16')
      expect(deserialized?.lr).toBe(32)
      expect(deserialized?.la).toBe(64)
      expect(deserialized?.tmp).toBe(50)
    })

    it('should round-trip training configuration', () => {
      const state = {
        ...baseState,
        selectedGPU: {
          id: 'nvidia-h100-80gb-sxm',
          name: 'NVIDIA H100 80GB SXM',
          manufacturer: 'nvidia' as const,
          vram_gb: 80,
          memory_bandwidth_gbps: 3352,
          memory_type: 'HBM3',
          bus_width: 5120,
          fp16_tflops: 1979,
          tier: 'datacenter' as const,
          interconnect: 'nvlink-4' as const,
          max_gpus_per_node: 8,
        },
        quantization: 'bf16' as const,
        sequenceLength: 2048,
        batchSize: 4,
        numGPUs: 1,
        mode: 'training' as const,
        optimizer: 'adamw-8bit' as const,
        loraRank: 8,
        loraAlpha: 16,
        targetModulesPercent: 25,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.m).toBe('training')
      expect(deserialized?.tm).toBe('lora')
      expect(deserialized?.to).toBe('adamw-8bit')
      expect(deserialized?.tp).toBe('bf16')
      expect(deserialized?.lr).toBe(8)
      expect(deserialized?.la).toBe(16)
      expect(deserialized?.tmp).toBe(25)
    })

    it('should serialize optimization fields when mode is training', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
        mode: 'training' as const,
        gradientAccumulationSteps: 8,
        gradientCheckpointing: true,
        flashAttention: true,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.ga).toBe(8)
      expect(deserialized?.gc).toBe(true)
      expect(deserialized?.fa).toBe(true)
    })

    it('should NOT serialize optimization fields when mode is inference', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        numGPUs: 1,
        gradientAccumulationSteps: 8,
        gradientCheckpointing: true,
        flashAttention: true,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      // Optimization fields should NOT be in URL when mode is inference
      expect(deserialized?.ga).toBeUndefined()
      expect(deserialized?.gc).toBeUndefined()
      expect(deserialized?.fa).toBeUndefined()
    })

    it('should round-trip optimization values', () => {
      const state = {
        ...baseState,
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        batchSize: 4,
        mode: 'training' as const,
        trainingMethod: 'full' as const,
        gradientAccumulationSteps: 32,
        gradientCheckpointing: true,
      }

      const serialized = serializeToURL(state)
      const deserialized = deserializeFromURL(serialized)

      expect(deserialized).not.toBeNull()
      expect(deserialized?.ga).toBe(32)
      expect(deserialized?.gc).toBe(true)
      expect(deserialized?.fa).toBe(false)
    })
  })

  describe('deserializeFromURL error handling', () => {
    it('should return null for empty string', () => {
      expect(deserializeFromURL('')).toBeNull()
    })

    it('should return null for garbage string', () => {
      expect(deserializeFromURL('garbage-data-that-is-not-valid')).toBeNull()
      expect(deserializeFromURL('12345!@#$%')).toBeNull()
    })

    it('should return null for invalid JSON', () => {
      // Valid base64 but invalid JSON
      const invalidBase64 = 'eyJpbnZhbGlkOiB9' // malformed JSON
      expect(deserializeFromURL(invalidBase64)).toBeNull()
    })

    it('should return null for invalid schema', () => {
      // Create JSON with correct structure but wrong types
      const invalidJson = JSON.stringify({
        q: 'fp16',
        sl: 'not-a-number', // Should be number, not string
        bs: 1,
        kvq: 'fp16',
        ng: 1,
        ss: 'tensor-parallel',
      })

      // Manually compress the invalid JSON
      const compressed = compressToEncodedURIComponent(invalidJson)

      // Should return null because schema validation fails
      expect(deserializeFromURL(compressed)).toBeNull()
    })

    it('should never throw exceptions', () => {
      // Test various edge cases that might cause errors
      expect(() => deserializeFromURL('')).not.toThrow()
      expect(() => deserializeFromURL('null')).not.toThrow()
      expect(() => deserializeFromURL('undefined')).not.toThrow()
      expect(() => deserializeFromURL('{}')).not.toThrow()
      expect(() => deserializeFromURL('[]')).not.toThrow()
    })
  })

  describe('URL safety and size', () => {
    it('should produce URL-safe output', () => {
      const state = {
        ...baseState,
        selectedGPU: {
          id: 'nvidia-h100-80gb-sxm',
          name: 'NVIDIA H100 80GB SXM',
          manufacturer: 'nvidia' as const,
          vram_gb: 80,
          memory_bandwidth_gbps: 3352,
          memory_type: 'HBM3',
          bus_width: 5120,
          fp16_tflops: 1979,
          tier: 'datacenter' as const,
          interconnect: 'nvlink-4' as const,
          max_gpus_per_node: 8,
        },
      }

      const serialized = serializeToURL(state)

      // Check for URL-unsafe characters
      expect(serialized).not.toMatch(/[\s#&]/)
      // lz-string encodeURIComponent produces base64-like with some URL-safe chars
      expect(serialized).toMatch(/^[A-Za-z0-9_\-$.+!*'(),]+$/)
    })

    it('should produce reasonably sized URLs for typical configs', () => {
      const state = {
        ...baseState,
        selectedGPU: {
          id: 'nvidia-h100-80gb-sxm',
          name: 'NVIDIA H100 80GB SXM',
          manufacturer: 'nvidia' as const,
          vram_gb: 80,
          memory_bandwidth_gbps: 3352,
          memory_type: 'HBM3',
          bus_width: 5120,
          fp16_tflops: 1979,
          tier: 'datacenter' as const,
          interconnect: 'nvlink-4' as const,
          max_gpus_per_node: 8,
        },
      }

      const serialized = serializeToURL(state)

      // Should be well under 1800 chars for typical config
      expect(serialized.length).toBeLessThan(1800)
      // Should have meaningful compression (uncompressed JSON is much larger)
      expect(serialized.length).toBeGreaterThan(10)
    })
  })

  describe('sequence length round-trip at long context', () => {
    it('round-trips 1M tokens', () => {
      const deserialized = deserializeFromURL(
        serializeToURL({ ...baseState, sequenceLength: 1048576 }),
      )
      expect(deserialized?.sl).toBe(1048576)
    })

    it('round-trips the maximum sequence length', () => {
      const deserialized = deserializeFromURL(
        serializeToURL({ ...baseState, sequenceLength: 10485760 }),
      )
      expect(deserialized?.sl).toBe(10485760)
    })
  })

  describe('multi-node URL state', () => {
    it('round-trips the node topology', () => {
      const hash = serializeToURL({
        ...baseState,
        numGPUs: 8,
        numNodes: 4,
        interNodeFabric: 'ethernet-1600g',
      })
      const decoded = deserializeFromURL(hash)
      expect(decoded?.ng).toBe(8)
      expect(decoded?.nn).toBe(4)
      expect(decoded?.fab).toBe('ethernet-1600g')
    })

    it('omits the node keys at a single node, keeping shared links short', () => {
      const hash = serializeToURL({ ...baseState, numGPUs: 4, numNodes: 1 })
      const decoded = deserializeFromURL(hash)
      expect(decoded).not.toBeNull()
      expect(decoded?.nn).toBeUndefined()
      expect(decoded?.fab).toBeUndefined()
    })

    it('accepts a pre-feature URL, where ng meant total GPUs', () => {
      // 1 node x 4 GPUs is arithmetically the same configuration as the old
      // "4 GPUs", so old links keep working and keep meaning the same thing.
      const legacy = serializeToURL({ ...baseState, numGPUs: 4, numNodes: 1 })
      const decoded = deserializeFromURL(legacy)
      expect(decoded).not.toBeNull()
      expect(decoded?.ng).toBe(4)
      expect(decoded?.nn ?? 1).toBe(1)
    })

    it('accepts a hash built before the node keys existed', () => {
      // Hand-built payload matching the pre-feature URLStateSchema shape (no
      // nn/fab/fabc keys at all), proving genuinely old links still decode -
      // not just links this version happens to omit the keys from.
      const legacy = compressToEncodedURIComponent(
        JSON.stringify({
          modelId: 'meta-llama-llama-3-70b',
          gpuId: 'nvidia-h100-80gb-sxm',
          q: 'gptq',
          sl: 4096,
          bs: 1,
          kvq: 'fp16',
          ng: 4,
          ss: 'tensor-parallel',
        }),
      )
      const decoded = deserializeFromURL(legacy)
      expect(decoded).not.toBeNull()
      expect(decoded?.ng).toBe(4)
      expect(decoded?.nn).toBeUndefined()
      expect(decoded?.fab).toBeUndefined()
      expect(decoded?.fabc).toBeUndefined()
    })

    it('round-trips a custom fabric', () => {
      const hash = serializeToURL({
        ...baseState,
        numGPUs: 8,
        numNodes: 2,
        interNodeFabric: 'custom',
        customFabric: { name: 'Lab', port_gbps: 25 },
      })
      const decoded = deserializeFromURL(hash)
      expect(decoded?.fab).toBe('custom')
      expect(decoded?.fabc).toEqual({ name: 'Lab', port_gbps: 25 })
    })
  })
})

const realModels = validateModels(modelsData)
const realGPUs = validateGPUs(gpusData)
const lookups = {
  findModel: (id: string): Model | null => realModels.find((m) => m.id === id) ?? null,
  findGPU: (id: string): GPU | null => realGPUs.find((g) => g.id === id) ?? null,
}
function realModel(id: string): Model {
  const m = lookups.findModel(id)
  if (!m) throw new Error(`fixture model not found in models.json: ${id}`)
  return m
}
function realGPU(id: string): GPU {
  const g = lookups.findGPU(id)
  if (!g) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return g
}

describe('urlStateToConfig', () => {
  const everyKey = {
    ...baseState,
    selectedModel: realModel('meta-llama-llama-3.1-70b'),
    selectedGPU: realGPU('nvidia-h100-80gb-sxm'),
    quantization: 'fp8' as const,
    sequenceLength: 32768,
    batchSize: 8,
    kvQuantization: 'fp8' as const,
    numGPUs: 4,
    shardingStrategy: 'pipeline-parallel' as const,
    concurrentUsers: 64,
    kvTier: {
      tier: 'network' as const,
      customGBps: 20,
      activeShare: 0.5,
      burstSeconds: 12,
      capacityTB: 3,
    },
    numNodes: 2,
    interNodeFabric: 'custom' as const,
    customFabric: { name: 'Lab', port_gbps: 25 },
    offloadingEnabled: true,
    offloadTarget: 'nvme' as const,
    offloadMode: 'layers' as const,
    offloadPercentage: 10,
    offloadLayers: 12,
    kvCacheOffload: true,
    offloadHostCapacityGB: 4096,
    mode: 'training' as const,
    trainingMethod: 'qlora' as const,
    optimizer: 'adafactor' as const,
    trainingPrecision: 'fp16' as const,
    loraRank: 64,
    loraAlpha: 128,
    targetModulesPercent: 50,
    gradientAccumulationSteps: 8,
    gradientCheckpointing: true,
    flashAttention: true,
    frameworkPreset: 'deepspeed-zero3' as const,
    cpuOffloadOptimizer: true,
    interconnectOverride: 'pcie-5',
  }

  it('round-trips every serialized key, including ga, gc, fa, fp, co and the new io', () => {
    const decoded = deserializeFromURL(serializeToURL(everyKey))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch, missing } = urlStateToConfig(decoded, lookups)
    expect(missing).toEqual([])
    const { selectedModel, selectedGPU, ...rest } = everyKey
    expect(patch.selectedModel?.id).toBe(selectedModel.id)
    expect(patch.selectedGPU?.id).toBe(selectedGPU.id)
    expect(patch).toMatchObject(rest)
  })

  it('round-trips a custom GPU unified-memory flag', () => {
    const custom = { ...realGPU('apple-m3-ultra'), id: 'custom-1' }
    const decoded = deserializeFromURL(
      serializeToURL({ ...everyKey, mode: 'inference' as const, selectedGPU: custom }),
    )
    if (!decoded) throw new Error('expected the link to parse')
    expect(urlStateToConfig(decoded, lookups).patch.selectedGPU?.unified_memory).toBe(true)
  })

  it('restores fp raw: no auto-optimizations overwrite the link optimizer and flags', () => {
    const state = {
      ...everyKey,
      frameworkPreset: 'unsloth' as const,
      optimizer: 'adamw' as const,
      gradientCheckpointing: false,
    }
    const decoded = deserializeFromURL(serializeToURL(state))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, lookups)
    expect(patch.frameworkPreset).toBe('unsloth')
    expect(patch.optimizer).toBe('adamw')
    expect(patch.gradientCheckpointing).toBe(false)
  })

  it('defaults absent keys like links made before they existed', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(
        JSON.stringify({ q: 'fp16', sl: 4096, bs: 1, kvq: 'fp16', ng: 1, ss: 'tensor-parallel' }),
      ),
    )
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, lookups)
    expect(patch).toMatchObject({
      mode: 'inference',
      concurrentUsers: 1,
      numNodes: 1,
      kvTier: DEFAULT_KV_TIER,
      interconnectOverride: null,
    })
  })

  it('reports an unknown model or GPU id without custom params', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(
        JSON.stringify({
          modelId: 'gone',
          gpuId: 'gone',
          q: 'fp16',
          sl: 4096,
          bs: 1,
          kvq: 'fp16',
          ng: 1,
          ss: 'tensor-parallel',
        }),
      ),
    )
    if (!decoded) throw new Error('expected the link to parse')
    expect(urlStateToConfig(decoded, lookups).missing).toEqual([
      'Model from shared link not found in database',
      'GPU from shared link not found in database',
    ])
  })

  it('parses out-of-range numbers instead of rejecting the whole link (R10 corrects them)', () => {
    const decoded = deserializeFromURL(
      compressToEncodedURIComponent(
        JSON.stringify({
          q: 'fp16',
          sl: 100,
          bs: 0,
          kvq: 'fp16',
          ng: -3,
          ss: 'tensor-parallel',
          nn: 12,
          cu: 0,
        }),
      ),
    )
    expect(decoded).toMatchObject({ sl: 100, bs: 0, ng: -3, nn: 12, cu: 0 })
  })

  it('rejects a hand-edited custom model or GPU with a non-positive numeric field (no rule reads these)', () => {
    const badModel = compressToEncodedURIComponent(
      JSON.stringify({
        q: 'fp16',
        sl: 4096,
        bs: 1,
        kvq: 'fp16',
        ng: 1,
        ss: 'tensor-parallel',
        customModel: {
          name: 'x',
          num_parameters_billion: 0,
          hidden_size: 4096,
          num_hidden_layers: 32,
          num_attention_heads: 32,
          intermediate_size: 11008,
        },
      }),
    )
    expect(deserializeFromURL(badModel)).toBeNull()
    const badGPU = compressToEncodedURIComponent(
      JSON.stringify({
        q: 'fp16',
        sl: 4096,
        bs: 1,
        kvq: 'fp16',
        ng: 1,
        ss: 'tensor-parallel',
        customGPU: { name: 'x', vram_gb: 0, memory_bandwidth_gbps: 1000 },
      }),
    )
    expect(deserializeFromURL(badGPU)).toBeNull()
  })
})
