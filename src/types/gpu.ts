import type { GPU as GPUBase } from '@utils/schemas'

// Re-export the Zod-inferred type
export type GPU = GPUBase

// Type for custom GPU input (all fields optional except vram_gb)
export interface CustomGPUInput {
  name: string
  vram_gb: number
  memory_bandwidth_gbps?: number
  fp16_tflops?: number
  fp32_tflops?: number
}

// Helper to convert custom input to GPU
export function createCustomGPU(input: CustomGPUInput): GPU {
  return {
    id: `custom-${Date.now()}`,
    name: input.name,
    manufacturer: 'nvidia', // Default for custom
    vram_gb: input.vram_gb,
    memory_bandwidth_gbps: input.memory_bandwidth_gbps || 0,
    memory_type: 'Custom',
    bus_width: 0,
    fp16_tflops: input.fp16_tflops,
    fp32_tflops: input.fp32_tflops,
    tier: 'consumer',
    interconnect: 'none',
  }
}
