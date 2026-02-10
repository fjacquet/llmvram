import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import type { OptimizerType, TrainingPrecision, LoRAVRAMBreakdown } from './types'

/**
 * Calculate LoRA adapter parameter count
 */
export function calculateLoRAAdapterParams(params: {
  hiddenSize: number
  numHiddenLayers: number
  rank: number
  targetModulesPercent: number
}): Decimal {
  // TODO: Implement
  return new Decimal(0)
}

/**
 * Calculate LoRA fine-tuning VRAM breakdown
 */
export function calculateLoRAFineTuningVRAM(params: {
  model: Model
  trainingPrecision: TrainingPrecision
  optimizer: OptimizerType
  batchSize: number
  sequenceLength: number
  loraRank: number
  targetModulesPercent: number
}): LoRAVRAMBreakdown {
  // TODO: Implement
  return {
    modelWeights: new Decimal(0),
    masterWeights: new Decimal(0),
    gradients: new Decimal(0),
    optimizerStates: new Decimal(0),
    activations: new Decimal(0),
    frameworkOverhead: new Decimal(0),
    total: new Decimal(0),
    trainableParameters: new Decimal(0),
    totalParameters: new Decimal(0),
    method: 'lora',
    baseWeights: new Decimal(0),
    adapterWeights: new Decimal(0),
    adapterParameters: new Decimal(0),
  }
}

/**
 * Calculate QLoRA fine-tuning VRAM breakdown
 */
export function calculateQLoRAFineTuningVRAM(params: {
  model: Model
  optimizer: OptimizerType
  batchSize: number
  sequenceLength: number
  loraRank: number
  targetModulesPercent: number
}): LoRAVRAMBreakdown {
  // TODO: Implement
  return {
    modelWeights: new Decimal(0),
    masterWeights: new Decimal(0),
    gradients: new Decimal(0),
    optimizerStates: new Decimal(0),
    activations: new Decimal(0),
    frameworkOverhead: new Decimal(0),
    total: new Decimal(0),
    trainableParameters: new Decimal(0),
    totalParameters: new Decimal(0),
    method: 'qlora',
    baseWeights: new Decimal(0),
    adapterWeights: new Decimal(0),
    adapterParameters: new Decimal(0),
  }
}
