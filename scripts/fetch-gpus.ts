import { writeFile } from 'node:fs/promises'
import { validateGPUs, type GPU } from '../src/utils/schemas'

/**
 * GPU Database Refresh Script
 *
 * This script provides a template for updating GPU specs.
 * Sources:
 * - NVIDIA H100/A100: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
 * - RTX Series: https://www.nvidia.com/en-us/geforce/graphics-cards/ and TechPowerUp GPU Database
 * - AMD MI300X: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
 * - Apple Silicon: Wikipedia and official Apple specs
 *
 * Update the GPUS array below with specs from official sources, then run:
 * npm run refresh:gpus
 */

const GPUS: GPU[] = [
  // NVIDIA Datacenter
  {
    id: 'nvidia-h100-80gb-pcie',
    name: 'NVIDIA H100 80GB PCIe',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 2000,
    memory_type: 'HBM3',
    bus_width: 5120,
    fp16_tflops: 989,
    fp32_tflops: 51,
    tdp_watts: 350,
    interconnect: 'nvlink-4',
    tier: 'datacenter',
  },
  {
    id: 'nvidia-h100-80gb-sxm',
    name: 'NVIDIA H100 80GB SXM',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 3350,
    memory_type: 'HBM3',
    bus_width: 5120,
    fp16_tflops: 989,
    fp32_tflops: 51,
    tdp_watts: 700,
    interconnect: 'nvlink-4',
    tier: 'datacenter',
  },
  {
    id: 'nvidia-h200-141gb',
    name: 'NVIDIA H200 141GB',
    manufacturer: 'nvidia',
    vram_gb: 141,
    memory_bandwidth_gbps: 4800,
    memory_type: 'HBM3e',
    bus_width: 5120,
    fp16_tflops: 989,
    fp32_tflops: 51,
    tdp_watts: 700,
    interconnect: 'nvlink-4',
    tier: 'datacenter',
  },
  {
    id: 'nvidia-b200-192gb',
    name: 'NVIDIA B200 192GB',
    manufacturer: 'nvidia',
    vram_gb: 192,
    memory_bandwidth_gbps: 8000,
    memory_type: 'HBM3e',
    bus_width: 8192,
    fp16_tflops: 4500,
    fp32_tflops: 90,
    tdp_watts: 1000,
    interconnect: 'nvlink-5',
    tier: 'datacenter',
  },
  {
    id: 'nvidia-a100-80gb-pcie',
    name: 'NVIDIA A100 80GB PCIe',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 1935,
    memory_type: 'HBM2e',
    bus_width: 5120,
    fp16_tflops: 312,
    fp32_tflops: 19.5,
    tdp_watts: 300,
    interconnect: 'nvlink',
    tier: 'datacenter',
  },
  {
    id: 'nvidia-a100-80gb-sxm',
    name: 'NVIDIA A100 80GB SXM',
    manufacturer: 'nvidia',
    vram_gb: 80,
    memory_bandwidth_gbps: 2039,
    memory_type: 'HBM2e',
    bus_width: 5120,
    fp16_tflops: 312,
    fp32_tflops: 19.5,
    tdp_watts: 400,
    interconnect: 'nvlink',
    tier: 'datacenter',
  },

  // NVIDIA Consumer
  {
    id: 'nvidia-rtx-5090',
    name: 'NVIDIA RTX 5090',
    manufacturer: 'nvidia',
    vram_gb: 32,
    memory_bandwidth_gbps: 1792,
    memory_type: 'GDDR7',
    bus_width: 512,
    fp16_tflops: 318,
    fp32_tflops: 159,
    tdp_watts: 575,
    interconnect: 'none',
    tier: 'consumer',
  },
  {
    id: 'nvidia-rtx-4090',
    name: 'NVIDIA RTX 4090',
    manufacturer: 'nvidia',
    vram_gb: 24,
    memory_bandwidth_gbps: 1008,
    memory_type: 'GDDR6X',
    bus_width: 384,
    fp16_tflops: 165,
    fp32_tflops: 82.6,
    tdp_watts: 450,
    interconnect: 'none',
    tier: 'consumer',
  },
  {
    id: 'nvidia-rtx-3090',
    name: 'NVIDIA RTX 3090',
    manufacturer: 'nvidia',
    vram_gb: 24,
    memory_bandwidth_gbps: 936,
    memory_type: 'GDDR6X',
    bus_width: 384,
    fp16_tflops: 71,
    fp32_tflops: 35.6,
    tdp_watts: 350,
    interconnect: 'none',
    tier: 'consumer',
  },

  // AMD Datacenter
  {
    id: 'amd-mi300x',
    name: 'AMD MI300X',
    manufacturer: 'amd',
    vram_gb: 192,
    memory_bandwidth_gbps: 5300,
    memory_type: 'HBM3',
    bus_width: 8192,
    fp16_tflops: 1307,
    fp32_tflops: 163,
    tdp_watts: 750,
    interconnect: 'infinity-fabric',
    tier: 'datacenter',
  },

  // Apple Silicon
  {
    id: 'apple-m1-ultra',
    name: 'Apple M1 Ultra',
    manufacturer: 'apple',
    vram_gb: 128,
    memory_bandwidth_gbps: 800,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 21,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
  {
    id: 'apple-m2-ultra',
    name: 'Apple M2 Ultra',
    manufacturer: 'apple',
    vram_gb: 192,
    memory_bandwidth_gbps: 800,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 27,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
  {
    id: 'apple-m3-ultra',
    name: 'Apple M3 Ultra',
    manufacturer: 'apple',
    vram_gb: 192,
    memory_bandwidth_gbps: 800,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 40,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
  {
    id: 'apple-m4-max',
    name: 'Apple M4 Max',
    manufacturer: 'apple',
    vram_gb: 128,
    memory_bandwidth_gbps: 546,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 29,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
  {
    id: 'apple-m3-max',
    name: 'Apple M3 Max',
    manufacturer: 'apple',
    vram_gb: 128,
    memory_bandwidth_gbps: 400,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 29,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
  {
    id: 'apple-m2-max',
    name: 'Apple M2 Max',
    manufacturer: 'apple',
    vram_gb: 96,
    memory_bandwidth_gbps: 400,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 13.6,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
  {
    id: 'apple-m1-max',
    name: 'Apple M1 Max',
    manufacturer: 'apple',
    vram_gb: 64,
    memory_bandwidth_gbps: 400,
    memory_type: 'Unified',
    bus_width: 0,
    fp16_tflops: 10.4,
    tier: 'apple-silicon',
    interconnect: 'unified',
  },
]

async function main() {
  console.log(`Validating ${GPUS.length} GPU entries against schema...`)

  // Validate all GPUs against schema
  try {
    validateGPUs(GPUS)
    console.log('✓ All GPUs valid')
  } catch (error) {
    console.error('✗ Validation failed:', error)
    process.exit(1)
  }

  // Write to JSON file
  const outputPath = 'src/data/gpus.json'
  await writeFile(outputPath, JSON.stringify(GPUS, null, 2))
  console.log(`\n✓ Wrote ${GPUS.length} GPUs to ${outputPath}`)
}

main().catch(console.error)
