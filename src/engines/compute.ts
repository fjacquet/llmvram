/**
 * Which tensor core a weight format computes on (spec D5)
 *
 * This table is vLLM's kernel choice: vLLM is the reference engine (the user's decision,
 * 2026-11-10). Another engine would get its own table. Sources, vLLM stable docs:
 * - FP8 W8A8: "FP8 computation is supported on NVIDIA GPUs with compute capability >= 8.9";
 *   below that vLLM runs weight-only W8A16 Marlin, so FP8 falls back to the 16-bit rate.
 * - `CompressedTensorsW4A4Mxfp4`: "On SM100+ with FlashInfer it performs true W4A4 ...
 *   otherwise it uses W4A16 weight-only via Marlin". Marlin computes in 16-bit, so FP4 falls
 *   back to FP16, never to FP8.
 * - `TorchNvFp4LinearKernel` requires sm_10x (Blackwell); `MarlinNvFp4LinearKernel` is
 *   weight-only W4A16 elsewhere. NVFP4 is NVIDIA-only: AMD runs MXFP4, not NVFP4.
 * - nvfp6 and every integer / GGUF / AWQ / GPTQ format compute at 16-bit (our call for nvfp6:
 *   no confirmed native vLLM FP6 path).
 *
 * A GPU without a figure for a precision (fp8_tflops / fp4_tflops absent) has no native path.
 */
import type { GPU } from '@utils/schemas'
import type { QuantizationFormat } from './types'

export type ComputePrecision = 'fp4' | 'fp8' | 'fp16'

export interface ComputeRate {
  precision: ComputePrecision
  tflops: number
}

// Exhaustive on purpose: a new format fails to typecheck until someone classifies it.
const FORMAT_PRECISION: Record<QuantizationFormat, ComputePrecision> = {
  fp32: 'fp16',
  fp16: 'fp16',
  bf16: 'fp16',
  fp8: 'fp8',
  mxfp4: 'fp4',
  nvfp4: 'fp4',
  nvfp6: 'fp16',
  int8: 'fp16',
  int4: 'fp16',
  nf4: 'fp16',
  gptq: 'fp16',
  awq: 'fp16',
  'gguf-q8_0': 'fp16',
  'gguf-q6_k': 'fp16',
  'gguf-q5_k_s': 'fp16',
  'gguf-q5_k_m': 'fp16',
  'gguf-q5_0': 'fp16',
  'gguf-q4_k_s': 'fp16',
  'gguf-q4_k_m': 'fp16',
  'gguf-q4_0': 'fp16',
  'gguf-q3_k_l': 'fp16',
  'gguf-q3_k_m': 'fp16',
  'gguf-q3_k_s': 'fp16',
  'gguf-q2_k': 'fp16',
}

/** Rate attention FLOPs run at: always the 16-bit figure (fp16 ?? fp32 ?? 0) */
export function attentionComputeRate(gpu: GPU): ComputeRate {
  return { precision: 'fp16', tflops: gpu.fp16_tflops ?? gpu.fp32_tflops ?? 0 }
}

/** Rate the linear (GEMM) FLOPs run at for this format on this GPU (spec D5). tflops 0 = no FLOPS data */
export function linearComputeRate(gpu: GPU, format: QuantizationFormat): ComputeRate {
  const precision = FORMAT_PRECISION[format]
  if (precision === 'fp8' && gpu.fp8_tflops !== undefined) {
    return { precision: 'fp8', tflops: gpu.fp8_tflops }
  }
  const nativeFp4 =
    gpu.fp4_tflops !== undefined && (format !== 'nvfp4' || gpu.manufacturer === 'nvidia')
  if (precision === 'fp4' && nativeFp4 && gpu.fp8_tflops !== undefined) {
    // Spec D9 fallback: no prefill-isolating FP4 measurement was found, so a native FP4
    // format computes at the card's FP8 rate rather than claiming the FP4 peak.
    return { precision: 'fp8', tflops: gpu.fp8_tflops }
  }
  return attentionComputeRate(gpu)
}
