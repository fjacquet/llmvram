import type { Model } from '@utils/schemas'
import Decimal from 'decimal.js'
import { BYTES_PER_GB, KV_PRECISION_BYTES } from './constants'
import type { KVCachePrecision } from './types'

/**
 * Calculate KV cache VRAM requirement with GQA/MQA support
 *
 * KV cache stores key and value tensors for all tokens in the sequence.
 * Size depends on sequence length, batch size, model architecture, and precision.
 *
 * GQA (Grouped Query Attention) optimization:
 * - Standard MHA: num_kv_heads = num_attention_heads (ratio = 1.0, full cache)
 * - GQA: num_kv_heads < num_attention_heads (ratio < 1.0, reduced cache)
 * - MQA: num_kv_heads = 1 (ratio = 1/num_heads, maximum reduction)
 * - Example: Llama 3 70B has 8 KV heads / 64 query heads = 0.125 ratio (8x reduction)
 *
 * Formula from NVIDIA TensorRT-LLM documentation:
 * KV_bytes = 2 * num_layers * hidden_size * seq_len * batch * precision_bytes * gqa_ratio
 *
 * @param params - Calculation parameters
 * @param params.model - Model configuration (must include num_hidden_layers, hidden_size, num_attention_heads, optionally num_kv_heads)
 * @param params.sequenceLength - Maximum sequence length (prompt + generation)
 * @param params.batchSize - Number of concurrent sequences
 * @param params.kvPrecision - KV cache quantization precision (default: fp16)
 * @returns KV cache VRAM requirement in GB as Decimal
 *
 * @example
 * ```ts
 * // Llama 3 70B with GQA (8 KV heads, 64 query heads)
 * const llama70b = { num_hidden_layers: 80, hidden_size: 8192, num_attention_heads: 64, num_kv_heads: 8, ... }
 * calculateKVCacheVRAM({ model: llama70b, sequenceLength: 4096, batchSize: 1, kvPrecision: 'fp16' })
 * // Returns ~0.94 GB (8x smaller than MHA equivalent)
 *
 * // Standard MHA model (no GQA)
 * const llama7b = { num_hidden_layers: 32, hidden_size: 4096, num_attention_heads: 32, ... }
 * calculateKVCacheVRAM({ model: llama7b, sequenceLength: 2048, batchSize: 1, kvPrecision: 'fp16' })
 * // Returns ~0.5 GB (no GQA reduction, ratio = 1.0)
 * ```
 */
export function calculateKVCacheVRAM(params: {
  model: Model
  sequenceLength: number
  batchSize: number
  kvPrecision: KVCachePrecision
}): Decimal {
  const { model, sequenceLength, batchSize, kvPrecision } = params

  // GQA ratio: (num_kv_heads ?? num_attention_heads) / num_attention_heads
  // - If num_kv_heads is undefined, fallback to num_attention_heads (standard MHA, ratio = 1.0)
  // - If num_kv_heads < num_attention_heads, GQA reduction applies
  // - If num_kv_heads = 1, MQA (maximum reduction)
  const numKVHeads = model.num_kv_heads ?? model.num_attention_heads
  const gqaRatio = new Decimal(numKVHeads).div(model.num_attention_heads)

  // Precision bytes for KV cache elements
  const precisionBytes = KV_PRECISION_BYTES[kvPrecision]

  // KV cache formula: 2 (Key + Value) * layers * hidden_size * seq_len * batch * precision * gqa_ratio
  const kvBytes = new Decimal(2)
    .mul(model.num_hidden_layers)
    .mul(model.hidden_size)
    .mul(sequenceLength)
    .mul(batchSize)
    .mul(precisionBytes)
    .mul(gqaRatio)

  // Convert to GB
  return kvBytes.div(BYTES_PER_GB)
}
