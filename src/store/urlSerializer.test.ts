import { compressToEncodedURIComponent } from 'lz-string'
import { describe, expect, it } from 'vitest'
import { deserializeFromURL, isCustomId, serializeToURL } from './urlSerializer'

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
    it('should round-trip curated model and GPU', () => {
      const state = {
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
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
      }

      const serialized = serializeToURL(state)
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
        },
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        batchSize: 1,
        kvQuantization: 'fp16' as const,
        numGPUs: 1,
        shardingStrategy: 'tensor-parallel' as const,
        offloadingEnabled: false,
        offloadTarget: 'cpu-ram' as const,
        offloadMode: 'percentage' as const,
        offloadPercentage: 0,
        offloadLayers: 0,
        kvCacheOffload: false,
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
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
        },
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        batchSize: 1,
        kvQuantization: 'fp16' as const,
        numGPUs: 1,
        shardingStrategy: 'tensor-parallel' as const,
        offloadingEnabled: false,
        offloadTarget: 'cpu-ram' as const,
        offloadMode: 'percentage' as const,
        offloadPercentage: 0,
        offloadLayers: 0,
        kvCacheOffload: false,
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
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
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        batchSize: 1,
        kvQuantization: 'fp16' as const,
        numGPUs: 1,
        shardingStrategy: 'tensor-parallel' as const,
        offloadingEnabled: true,
        offloadTarget: 'nvme' as const,
        offloadMode: 'layers' as const,
        offloadPercentage: 50,
        offloadLayers: 20,
        kvCacheOffload: true,
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
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
    })

    it('should NOT serialize training fields when mode is inference', () => {
      const state = {
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        batchSize: 1,
        kvQuantization: 'fp16' as const,
        numGPUs: 1,
        shardingStrategy: 'tensor-parallel' as const,
        offloadingEnabled: false,
        offloadTarget: 'cpu-ram' as const,
        offloadMode: 'percentage' as const,
        offloadPercentage: 0,
        offloadLayers: 0,
        kvCacheOffload: false,
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
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
        selectedModel: null,
        selectedGPU: null,
        quantization: 'fp16' as const,
        sequenceLength: 2048,
        batchSize: 1,
        kvQuantization: 'fp16' as const,
        numGPUs: 1,
        shardingStrategy: 'tensor-parallel' as const,
        offloadingEnabled: false,
        offloadTarget: 'cpu-ram' as const,
        offloadMode: 'percentage' as const,
        offloadPercentage: 0,
        offloadLayers: 0,
        kvCacheOffload: false,
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
          tier: 'datacenter' as const,
          interconnect: 'nvlink-4' as const,
        },
        quantization: 'bf16' as const,
        sequenceLength: 2048,
        batchSize: 4,
        kvQuantization: 'fp16' as const,
        numGPUs: 1,
        shardingStrategy: 'tensor-parallel' as const,
        offloadingEnabled: false,
        offloadTarget: 'cpu-ram' as const,
        offloadMode: 'percentage' as const,
        offloadPercentage: 0,
        offloadLayers: 0,
        kvCacheOffload: false,
        mode: 'training' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw-8bit' as const,
        trainingPrecision: 'bf16' as const,
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
          tier: 'datacenter' as const,
          interconnect: 'nvlink-4' as const,
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
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
      }

      const serialized = serializeToURL(state)

      // Check for URL-unsafe characters
      expect(serialized).not.toMatch(/[\s#&]/)
      // lz-string encodeURIComponent produces base64-like with some URL-safe chars
      expect(serialized).toMatch(/^[A-Za-z0-9_\-$.+!*'(),]+$/)
    })

    it('should produce reasonably sized URLs for typical configs', () => {
      const state = {
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
          tier: 'datacenter' as const,
          interconnect: 'nvlink-4' as const,
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
        mode: 'inference' as const,
        trainingMethod: 'lora' as const,
        optimizer: 'adamw' as const,
        trainingPrecision: 'bf16' as const,
        loraRank: 16,
        loraAlpha: 32,
        targetModulesPercent: 30,
      }

      const serialized = serializeToURL(state)

      // Should be well under 1800 chars for typical config
      expect(serialized.length).toBeLessThan(1800)
      // Should have meaningful compression (uncompressed JSON is much larger)
      expect(serialized.length).toBeGreaterThan(10)
    })
  })
})
