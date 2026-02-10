import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { BYTES_PER_GB, TRAINING_FRAMEWORK_OVERHEAD_GB } from './constants'
import {
  calculateFullFineTuningVRAM,
  calculateOptimizerStateMemory,
  calculateTrainingActivationMemory,
} from './training'

describe('calculateOptimizerStateMemory', () => {
  it('calculates correct memory for AdamW optimizer', () => {
    // AdamW: 2 FP32 states = 8 bytes/param
    // 7B params * 8 / 1024^3 = ~52.15 GB
    const result = calculateOptimizerStateMemory(7.0, 'adamw')
    const expected = new Decimal(7e9).mul(8).div(BYTES_PER_GB)
    expect(result.toNumber()).toBeCloseTo(expected.toNumber(), 2)
  })

  it('calculates correct memory for SGD-momentum optimizer', () => {
    // SGD-momentum: 1 FP32 state = 4 bytes/param
    // 7B params * 4 / 1024^3 = ~26.08 GB
    const result = calculateOptimizerStateMemory(7.0, 'sgd-momentum')
    const expected = new Decimal(7e9).mul(4).div(BYTES_PER_GB)
    expect(result.toNumber()).toBeCloseTo(expected.toNumber(), 2)
  })

  it('calculates correct memory for 8-bit AdamW optimizer', () => {
    // AdamW-8bit: 2 8-bit states = 2 bytes/param
    // 7B params * 2 / 1024^3 = ~13.04 GB
    const result = calculateOptimizerStateMemory(7.0, 'adamw-8bit')
    const expected = new Decimal(7e9).mul(2).div(BYTES_PER_GB)
    expect(result.toNumber()).toBeCloseTo(expected.toNumber(), 2)
  })

  it('calculates correct memory for Adafactor optimizer', () => {
    // Adafactor: factored approximation = 4 bytes/param
    // 7B params * 4 / 1024^3 = ~26.08 GB
    const result = calculateOptimizerStateMemory(7.0, 'adafactor')
    const expected = new Decimal(7e9).mul(4).div(BYTES_PER_GB)
    expect(result.toNumber()).toBeCloseTo(expected.toNumber(), 2)
  })

  it('scales correctly for small models', () => {
    // 0.5B params with AdamW
    // 0.5B * 8 / 1024^3 = ~3.73 GB
    const result = calculateOptimizerStateMemory(0.5, 'adamw')
    const expected = new Decimal(0.5e9).mul(8).div(BYTES_PER_GB)
    expect(result.toNumber()).toBeCloseTo(expected.toNumber(), 2)
  })
})

describe('calculateTrainingActivationMemory', () => {
  const model7B: Model = {
    id: 'test-7b',
    name: 'Test 7B',
    organization: 'test',
    num_parameters_billion: 7.0,
    hidden_size: 4096,
    num_layers: 32,
    num_heads: 32,
    num_kv_heads: 32,
    intermediate_size: 11008,
    vocab_size: 32000,
    max_position_embeddings: 4096,
    architecture: 'dense',
    license: 'test',
    release_date: '2024-01-01',
  }

  it('calculates activation memory for 7B model with batch=1, seq=2048', () => {
    // Formula: batch * seq * hidden * layers * 10 * 2 / 1024^3
    // 1 * 2048 * 4096 * 32 * 10 * 2 / 1024^3 = ~5.0 GB
    const result = calculateTrainingActivationMemory(model7B, 1, 2048)
    const expected = new Decimal(1).mul(2048).mul(4096).mul(32).mul(10).mul(2).div(BYTES_PER_GB)
    expect(result.toNumber()).toBeCloseTo(expected.toNumber(), 1)
  })

  it('scales linearly with batch size', () => {
    // batch=4 should be 4x the memory of batch=1
    const batch1 = calculateTrainingActivationMemory(model7B, 1, 2048)
    const batch4 = calculateTrainingActivationMemory(model7B, 4, 2048)
    expect(batch4.div(batch1).toNumber()).toBeCloseTo(4, 1)
  })

  it('scales linearly with sequence length', () => {
    // seq=4096 should be 2x the memory of seq=2048
    const seq2048 = calculateTrainingActivationMemory(model7B, 1, 2048)
    const seq4096 = calculateTrainingActivationMemory(model7B, 1, 4096)
    expect(seq4096.div(seq2048).toNumber()).toBeCloseTo(2, 1)
  })

  it('handles MoE models with active parameter ratio', () => {
    const moeModel: Model = {
      ...model7B,
      id: 'test-mixtral',
      name: 'Test Mixtral',
      num_parameters_billion: 46.7,
      architecture: 'moe',
      num_experts: 8,
      num_experts_per_token: 2,
    }

    const result = calculateTrainingActivationMemory(moeModel, 1, 2048)
    // Should use active param ratio for scaling
    // Active ratio = 2/8 = 0.25 for expert params
    // Should be less than full 46.7B activation memory
    expect(result.toNumber()).toBeGreaterThan(0)
  })
})

describe('calculateFullFineTuningVRAM', () => {
  const model7B: Model = {
    id: 'test-7b',
    name: 'Test 7B',
    organization: 'test',
    num_parameters_billion: 7.0,
    hidden_size: 4096,
    num_layers: 32,
    num_heads: 32,
    num_kv_heads: 32,
    intermediate_size: 11008,
    vocab_size: 32000,
    max_position_embeddings: 4096,
    architecture: 'dense',
    license: 'test',
    release_date: '2024-01-01',
  }

  it('calculates complete breakdown for 7B BF16 AdamW training', () => {
    const result = calculateFullFineTuningVRAM({
      model: model7B,
      trainingPrecision: 'bf16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
    })

    // Expected components:
    // modelWeights: 7B * 2 / 1024^3 = ~13.04 GB (BF16)
    // masterWeights: 7B * 4 / 1024^3 = ~26.08 GB (FP32 for mixed precision)
    // gradients: 7B * 2 / 1024^3 = ~13.04 GB (BF16)
    // optimizerStates: 7B * 8 / 1024^3 = ~52.15 GB (FP32 AdamW)
    // activations: ~5.0 GB (from calculateTrainingActivationMemory)
    // frameworkOverhead: 1.5 GB
    // total: sum of all above

    expect(result.modelWeights.toNumber()).toBeCloseTo(13.04, 1)
    expect(result.masterWeights.toNumber()).toBeCloseTo(26.08, 1)
    expect(result.gradients.toNumber()).toBeCloseTo(13.04, 1)
    expect(result.optimizerStates.toNumber()).toBeCloseTo(52.15, 1)
    expect(result.activations.toNumber()).toBeCloseTo(5.0, 0)
    expect(result.frameworkOverhead.equals(TRAINING_FRAMEWORK_OVERHEAD_GB)).toBe(true)
    expect(result.total.toNumber()).toBeCloseTo(110.81, 1)
    expect(result.trainableParameters.toNumber()).toBe(7.0)
    expect(result.totalParameters.toNumber()).toBe(7.0)
    expect(result.method).toBe('full')
  })

  it('calculates FP32 training without master weights', () => {
    const result = calculateFullFineTuningVRAM({
      model: model7B,
      trainingPrecision: 'fp32',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
    })

    // FP32 training:
    // modelWeights: 7B * 4 = ~26.08 GB
    // masterWeights: 0 GB (no master copy in pure FP32)
    // gradients: 7B * 4 = ~26.08 GB
    // optimizerStates: 7B * 8 = ~52.15 GB

    expect(result.modelWeights.toNumber()).toBeCloseTo(26.08, 1)
    expect(result.masterWeights.toNumber()).toBe(0)
    expect(result.gradients.toNumber()).toBeCloseTo(26.08, 1)
    expect(result.optimizerStates.toNumber()).toBeCloseTo(52.15, 1)
  })

  it('calculates FP16 training with master weights', () => {
    const result = calculateFullFineTuningVRAM({
      model: model7B,
      trainingPrecision: 'fp16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
    })

    // FP16 training (mixed precision):
    // modelWeights: 7B * 2 = ~13.04 GB
    // masterWeights: 7B * 4 = ~26.08 GB (FP32 master copy)
    // gradients: 7B * 2 = ~13.04 GB

    expect(result.modelWeights.toNumber()).toBeCloseTo(13.04, 1)
    expect(result.masterWeights.toNumber()).toBeCloseTo(26.08, 1)
    expect(result.gradients.toNumber()).toBeCloseTo(13.04, 1)
  })

  it('scales with different optimizers', () => {
    const adamw = calculateFullFineTuningVRAM({
      model: model7B,
      trainingPrecision: 'bf16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
    })

    const sgd = calculateFullFineTuningVRAM({
      model: model7B,
      trainingPrecision: 'bf16',
      optimizer: 'sgd-momentum',
      batchSize: 1,
      sequenceLength: 2048,
    })

    // SGD should have less optimizer state memory (4 bytes vs 8 bytes)
    expect(sgd.optimizerStates.toNumber()).toBeLessThan(adamw.optimizerStates.toNumber())
    expect(sgd.total.toNumber()).toBeLessThan(adamw.total.toNumber())
  })

  it('handles 70B model scaling', () => {
    const model70B: Model = {
      ...model7B,
      id: 'test-70b',
      name: 'Test 70B',
      num_parameters_billion: 70.0,
    }

    const result = calculateFullFineTuningVRAM({
      model: model70B,
      trainingPrecision: 'bf16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
    })

    // Should be approximately 10x the 7B model
    expect(result.total.toNumber()).toBeGreaterThan(1000) // Much higher than 7B
  })

  it('returns all required breakdown fields', () => {
    const result = calculateFullFineTuningVRAM({
      model: model7B,
      trainingPrecision: 'bf16',
      optimizer: 'adamw',
      batchSize: 1,
      sequenceLength: 2048,
    })

    expect(result).toHaveProperty('modelWeights')
    expect(result).toHaveProperty('masterWeights')
    expect(result).toHaveProperty('gradients')
    expect(result).toHaveProperty('optimizerStates')
    expect(result).toHaveProperty('activations')
    expect(result).toHaveProperty('frameworkOverhead')
    expect(result).toHaveProperty('total')
    expect(result).toHaveProperty('trainableParameters')
    expect(result).toHaveProperty('totalParameters')
    expect(result).toHaveProperty('method')
  })
})
