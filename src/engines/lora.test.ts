import type { Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import {
  calculateLoRAAdapterParams,
  calculateLoRAFineTuningVRAM,
  calculateQLoRAFineTuningVRAM,
} from './lora'

// Test model: Llama 2 7B
const llama7b: Model = {
  id: 'test-llama-7b',
  name: 'Llama 2 7B',
  architecture: 'dense',
  num_parameters_billion: 7.0,
  hidden_size: 4096,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_kv_heads: 32,
  intermediate_size: 11008,
}

// Test model: Llama 2 70B
const llama70b: Model = {
  id: 'test-llama-70b',
  name: 'Llama 2 70B',
  architecture: 'dense',
  num_parameters_billion: 70.0,
  hidden_size: 8192,
  num_hidden_layers: 80,
  num_attention_heads: 64,
  num_kv_heads: 8,
  intermediate_size: 28672,
}

describe('calculateLoRAAdapterParams', () => {
  it('calculates adapter params for rank=16, targetModulesPercent=30 (2 modules)', () => {
    const result = calculateLoRAAdapterParams({
      hiddenSize: 4096,
      numHiddenLayers: 32,
      rank: 16,
      targetModulesPercent: 30,
    })

    // 2 * rank * hidden * targetModuleCount * layers
    // 2 * 16 * 4096 * 2 * 32 = 8,388,608
    expect(result.toNumber()).toBeCloseTo(8_388_608, -2)
  })

  it('calculates adapter params for rank=16, targetModulesPercent=50 (4 modules)', () => {
    const result = calculateLoRAAdapterParams({
      hiddenSize: 4096,
      numHiddenLayers: 32,
      rank: 16,
      targetModulesPercent: 50,
    })

    // 2 * 16 * 4096 * 4 * 32 = 16,777,216
    expect(result.toNumber()).toBeCloseTo(16_777_216, -2)
  })

  it('calculates adapter params for rank=16, targetModulesPercent=100 (7 modules)', () => {
    const result = calculateLoRAAdapterParams({
      hiddenSize: 4096,
      numHiddenLayers: 32,
      rank: 16,
      targetModulesPercent: 100,
    })

    // 2 * 16 * 4096 * 7 * 32 = 29,360,128
    expect(result.toNumber()).toBeCloseTo(29_360_128, -2)
  })

  it('calculates adapter params for rank=64, targetModulesPercent=50', () => {
    const result = calculateLoRAAdapterParams({
      hiddenSize: 4096,
      numHiddenLayers: 32,
      rank: 64,
      targetModulesPercent: 50,
    })

    // 2 * 64 * 4096 * 4 * 32 = 67,108,864
    expect(result.toNumber()).toBeCloseTo(67_108_864, -2)
  })

  it('calculates adapter params for rank=256, targetModulesPercent=100 (maximum)', () => {
    const result = calculateLoRAAdapterParams({
      hiddenSize: 4096,
      numHiddenLayers: 32,
      rank: 256,
      targetModulesPercent: 100,
    })

    // 2 * 256 * 4096 * 7 * 32 = 469,762,048
    expect(result.toNumber()).toBeCloseTo(469_762_048, -2)
  })

  it('respects minimum of 1 target module', () => {
    const result = calculateLoRAAdapterParams({
      hiddenSize: 4096,
      numHiddenLayers: 32,
      rank: 16,
      targetModulesPercent: 1, // Would be 0.07 modules, should round to 1
    })

    // At least 1 module should be targeted
    expect(result.toNumber()).toBeGreaterThan(0)
  })
})

describe('calculateLoRAFineTuningVRAM', () => {
  it('calculates LoRA VRAM for 7B model with BF16, AdamW, batch=1, seq=2048', () => {
    const breakdown = calculateLoRAFineTuningVRAM({
      model: llama7b,
      trainingPrecision: 'bf16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // Verify adapter parameter count (~16.8M)
    expect(breakdown.adapterParameters.toNumber()).toBeCloseTo(0.01678, 4) // In billions

    // Verify components
    expect(breakdown.baseWeights.toNumber()).toBeCloseTo(13.04, 1) // 7B * 2 bytes (BF16)
    expect(breakdown.adapterWeights.toNumber()).toBeCloseTo(0.031, 2) // ~16.8M * 2 bytes
    expect(breakdown.masterWeights.toNumber()).toBeCloseTo(0.063, 2) // FP32 master of adapters
    expect(breakdown.gradients.toNumber()).toBeCloseTo(0.031, 2) // BF16 gradients
    expect(breakdown.optimizerStates.toNumber()).toBeCloseTo(0.125, 2) // AdamW on adapters

    // Verify total is in expected range (~19.8 GB)
    expect(breakdown.total.toNumber()).toBeGreaterThan(18)
    expect(breakdown.total.toNumber()).toBeLessThan(22)

    // Verify method and trainable params
    expect(breakdown.method).toBe('lora')
    expect(breakdown.trainableParameters.toNumber()).toBeCloseTo(0.01678, 4)
    expect(breakdown.totalParameters.toNumber()).toBe(7.0)
  })

  it('calculates LoRA VRAM for FP32 training (no master weights)', () => {
    const breakdown = calculateLoRAFineTuningVRAM({
      model: llama7b,
      trainingPrecision: 'fp32',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // FP32 training should have zero master weights (they ARE the weights)
    expect(breakdown.masterWeights.toNumber()).toBe(0)

    // Base weights should be FP32 (4 bytes)
    expect(breakdown.baseWeights.toNumber()).toBeCloseTo(26.08, 1) // 7B * 4 bytes
  })

  it('applies optimizer states only to adapter parameters', () => {
    const breakdown = calculateLoRAFineTuningVRAM({
      model: llama7b,
      trainingPrecision: 'bf16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // Optimizer states should be 8 bytes * adapter params (NOT full 7B)
    // ~16.8M params * 8 bytes = ~0.125 GB
    expect(breakdown.optimizerStates.toNumber()).toBeCloseTo(0.125, 2)
    expect(breakdown.optimizerStates.toNumber()).toBeLessThan(1) // Much less than full model
  })
})

describe('calculateQLoRAFineTuningVRAM', () => {
  it('calculates QLoRA VRAM for 7B model with AdamW, batch=1, seq=2048', () => {
    const breakdown = calculateQLoRAFineTuningVRAM({
      model: llama7b,
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // QLoRA uses NF4 for base (0.5 bytes/param)
    expect(breakdown.baseWeights.toNumber()).toBeCloseTo(3.26, 1) // 7B * 0.5

    // Adapters in FP16 (~16.8M * 2 bytes)
    expect(breakdown.adapterWeights.toNumber()).toBeCloseTo(0.031, 2)

    // Master weights in FP32 (adapters only)
    expect(breakdown.masterWeights.toNumber()).toBeCloseTo(0.063, 2)

    // Gradients in FP16 (adapters only)
    expect(breakdown.gradients.toNumber()).toBeCloseTo(0.031, 2)

    // Optimizer states (adapters only, AdamW = 8 bytes)
    expect(breakdown.optimizerStates.toNumber()).toBeCloseTo(0.125, 2)

    // Total should be in research range (6-12 GB)
    expect(breakdown.total.toNumber()).toBeGreaterThan(6)
    expect(breakdown.total.toNumber()).toBeLessThan(12)

    // Verify method
    expect(breakdown.method).toBe('qlora')
  })

  it('calculates QLoRA VRAM for 70B model (still significant base weight)', () => {
    const breakdown = calculateQLoRAFineTuningVRAM({
      model: llama70b,
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // 70B in NF4: 70B * 0.5 bytes = ~32.6 GB
    expect(breakdown.baseWeights.toNumber()).toBeCloseTo(32.6, 1)

    // Total should be 40+ GB (much larger than 7B due to base weight)
    expect(breakdown.total.toNumber()).toBeGreaterThan(40)
  })

  it('QLoRA always uses three-precision architecture', () => {
    const breakdown = calculateQLoRAFineTuningVRAM({
      model: llama7b,
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // Verify three-precision architecture:
    // 1. Base in NF4 (0.5 bytes)
    expect(breakdown.baseWeights.toNumber()).toBeCloseTo(3.26, 1)

    // 2. Adapters in FP16 (2 bytes)
    const adapterCount = breakdown.adapterParameters.toNumber() * 1e9
    const expectedAdapterWeights = (adapterCount * 2) / 1024 ** 3
    expect(breakdown.adapterWeights.toNumber()).toBeCloseTo(expectedAdapterWeights, 3)

    // 3. Optimizer states in FP32 (8 bytes for AdamW)
    const expectedOptimizerStates = (adapterCount * 8) / 1024 ** 3
    expect(breakdown.optimizerStates.toNumber()).toBeCloseTo(expectedOptimizerStates, 3)
  })

  it('QLoRA applies gradients only to adapter parameters', () => {
    const breakdown = calculateQLoRAFineTuningVRAM({
      model: llama7b,
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
      loraRank: 16,
      targetModulesPercent: 50,
    })

    // Gradients should be FP16 * adapter params (NOT full 7B)
    // ~16.8M params * 2 bytes = ~0.031 GB
    expect(breakdown.gradients.toNumber()).toBeCloseTo(0.031, 2)
    expect(breakdown.gradients.toNumber()).toBeLessThan(1) // Much less than full model
  })
})
