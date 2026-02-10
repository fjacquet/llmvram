import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import {
  applyFlashAttention,
  applyGradientCheckpointing,
  calculateEffectiveBatchSize,
} from './optimizations'

describe('calculateEffectiveBatchSize', () => {
  it('calculates correct batch size for single GPU', () => {
    // (2, 4, 1) -> 2 * 4 * 1 = 8
    expect(calculateEffectiveBatchSize(2, 4, 1)).toBe(8)
  })

  it('calculates correct batch size for multiple GPUs', () => {
    // (2, 4, 4) -> 2 * 4 * 4 = 32
    expect(calculateEffectiveBatchSize(2, 4, 4)).toBe(32)
  })

  it('handles single accumulation step', () => {
    // (4, 1, 1) -> 4 * 1 * 1 = 4
    expect(calculateEffectiveBatchSize(4, 1, 1)).toBe(4)
  })

  it('handles multiple accumulation steps', () => {
    // (1, 8, 1) -> 1 * 8 * 1 = 8
    expect(calculateEffectiveBatchSize(1, 8, 1)).toBe(8)
  })

  it('defaults to 1 GPU when not specified', () => {
    // (2, 4) -> 2 * 4 * 1 = 8
    expect(calculateEffectiveBatchSize(2, 4)).toBe(8)
  })

  it('handles large configurations', () => {
    // (8, 16, 8) -> 8 * 16 * 8 = 1024
    expect(calculateEffectiveBatchSize(8, 16, 8)).toBe(1024)
  })
})

describe('applyGradientCheckpointing', () => {
  it('reduces activation memory by 60% when enabled', () => {
    // 10.0 GB * 0.4 = 4.0 GB (retain 40%, reduce 60%)
    const result = applyGradientCheckpointing(new Decimal(10.0), true)
    expect(result.toNumber()).toBeCloseTo(4.0, 2)
  })

  it('returns identity when disabled', () => {
    // 10.0 GB -> 10.0 GB (no change)
    const result = applyGradientCheckpointing(new Decimal(10.0), false)
    expect(result.toNumber()).toBeCloseTo(10.0, 2)
  })

  it('handles zero activation memory', () => {
    // 0 GB -> 0 GB (zero stays zero)
    const result = applyGradientCheckpointing(new Decimal(0), true)
    expect(result.toNumber()).toBe(0)
  })

  it('preserves small activation values', () => {
    // 0.5 GB * 0.4 = 0.2 GB
    const result = applyGradientCheckpointing(new Decimal(0.5), true)
    expect(result.toNumber()).toBeCloseTo(0.2, 2)
  })

  it('preserves large activation values', () => {
    // 100 GB * 0.4 = 40 GB
    const result = applyGradientCheckpointing(new Decimal(100), true)
    expect(result.toNumber()).toBeCloseTo(40, 2)
  })
})

describe('applyFlashAttention', () => {
  it('applies 15% reduction for short sequences (<2048)', () => {
    // 10.0 GB * 0.85 = 8.5 GB (retain 85%, reduce 15%)
    const result = applyFlashAttention(new Decimal(10.0), 1024, true)
    expect(result.toNumber()).toBeCloseTo(8.5, 2)
  })

  it('applies 50% reduction for medium sequences (2048-8191)', () => {
    // 10.0 GB * 0.5 = 5.0 GB (retain 50%, reduce 50%)
    const result = applyFlashAttention(new Decimal(10.0), 4096, true)
    expect(result.toNumber()).toBeCloseTo(5.0, 2)
  })

  it('applies 70% reduction for long sequences (>=8192)', () => {
    // 10.0 GB * 0.3 = 3.0 GB (retain 30%, reduce 70%)
    const result = applyFlashAttention(new Decimal(10.0), 16384, true)
    expect(result.toNumber()).toBeCloseTo(3.0, 2)
  })

  it('returns identity when disabled', () => {
    // 10.0 GB -> 10.0 GB (no change)
    const result = applyFlashAttention(new Decimal(10.0), 4096, false)
    expect(result.toNumber()).toBeCloseTo(10.0, 2)
  })

  it('handles boundary at 2048 (short to medium)', () => {
    // At exactly 2048, should use medium retention (50%)
    const result = applyFlashAttention(new Decimal(10.0), 2048, true)
    expect(result.toNumber()).toBeCloseTo(5.0, 2)
  })

  it('handles boundary at 8192 (medium to long)', () => {
    // At exactly 8192, should use long retention (30%)
    const result = applyFlashAttention(new Decimal(10.0), 8192, true)
    expect(result.toNumber()).toBeCloseTo(3.0, 2)
  })

  it('handles zero activation memory', () => {
    // 0 GB -> 0 GB (zero stays zero)
    const result = applyFlashAttention(new Decimal(0), 4096, true)
    expect(result.toNumber()).toBe(0)
  })
})

describe('combined optimizations', () => {
  it('stacks gradient checkpointing and flash attention multiplicatively', () => {
    // Base: 10 GB
    // After checkpointing: 10 * 0.4 = 4 GB
    // After flash attention (medium): 4 * 0.5 = 2 GB
    let activations = new Decimal(10.0)
    activations = applyGradientCheckpointing(activations, true)
    activations = applyFlashAttention(activations, 4096, true)
    expect(activations.toNumber()).toBeCloseTo(2.0, 2)
  })

  it('order of application does not matter (commutative)', () => {
    // Checkpointing first, then flash attention
    let path1 = new Decimal(10.0)
    path1 = applyGradientCheckpointing(path1, true)
    path1 = applyFlashAttention(path1, 4096, true)

    // Flash attention first, then checkpointing
    let path2 = new Decimal(10.0)
    path2 = applyFlashAttention(path2, 4096, true)
    path2 = applyGradientCheckpointing(path2, true)

    // Both should produce same result: 10 * 0.4 * 0.5 = 2.0
    expect(path1.toNumber()).toBeCloseTo(path2.toNumber(), 2)
    expect(path1.toNumber()).toBeCloseTo(2.0, 2)
  })

  it('handles all optimizations disabled', () => {
    // No optimizations -> identity
    let activations = new Decimal(10.0)
    activations = applyGradientCheckpointing(activations, false)
    activations = applyFlashAttention(activations, 4096, false)
    expect(activations.toNumber()).toBeCloseTo(10.0, 2)
  })

  it('handles only checkpointing enabled', () => {
    // Only checkpointing: 10 * 0.4 = 4
    let activations = new Decimal(10.0)
    activations = applyGradientCheckpointing(activations, true)
    activations = applyFlashAttention(activations, 4096, false)
    expect(activations.toNumber()).toBeCloseTo(4.0, 2)
  })

  it('handles only flash attention enabled', () => {
    // Only flash attention: 10 * 0.5 = 5
    let activations = new Decimal(10.0)
    activations = applyGradientCheckpointing(activations, false)
    activations = applyFlashAttention(activations, 4096, true)
    expect(activations.toNumber()).toBeCloseTo(5.0, 2)
  })

  it('handles maximum reduction (checkpointing + long flash)', () => {
    // Base: 10 GB
    // Checkpointing: 10 * 0.4 = 4 GB
    // Flash (long): 4 * 0.3 = 1.2 GB
    let activations = new Decimal(10.0)
    activations = applyGradientCheckpointing(activations, true)
    activations = applyFlashAttention(activations, 16384, true)
    expect(activations.toNumber()).toBeCloseTo(1.2, 2)
  })

  it('handles minimum reduction (no checkpointing + short flash)', () => {
    // Base: 10 GB
    // No checkpointing: 10 GB
    // Flash (short): 10 * 0.85 = 8.5 GB
    let activations = new Decimal(10.0)
    activations = applyGradientCheckpointing(activations, false)
    activations = applyFlashAttention(activations, 1024, true)
    expect(activations.toNumber()).toBeCloseTo(8.5, 2)
  })
})
