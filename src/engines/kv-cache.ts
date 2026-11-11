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
 * @param params.model - Model configuration (must include num_hidden_layers, hidden_size, num_attention_heads, optionally num_kv_heads, kv_cache_elements_per_token)
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
  return sessionKV(model, sequenceLength, sequenceLength, batchSize, kvPrecision)
}

/**
 * KV bytes one decode step reads, in GB: as calculateKVCacheVRAM, except that sparse attention
 * reads the full-attention KV of only attendedTokens(context) tokens. The indexer still reads
 * its cache over the whole context, and sliding / linear layers are not sparse.
 */
export function calculateDecodeKVReadVRAM(params: {
  model: Model
  sequenceLength: number
  batchSize: number
  kvPrecision: KVCachePrecision
}): Decimal {
  const { model, sequenceLength, batchSize, kvPrecision } = params
  const attended = attendedTokens(model, sequenceLength)
  return sessionKV(model, attended, sequenceLength, batchSize, kvPrecision)
}

function sessionKV(
  model: Model,
  attended: number,
  sequenceLength: number,
  batchSize: number,
  kvPrecision: KVCachePrecision,
): Decimal {
  return fullAttentionKV(model, attended, batchSize, kvPrecision)
    .add(slidingWindowKV(model, sequenceLength, batchSize, kvPrecision))
    .add(calculateLinearStateVRAM(model, batchSize))
    .add(calculateSparseIndexerVRAM(model, sequenceLength, batchSize))
}

/** Tokens the main attention reads per query: min(context, topk) under sparse attention (DSA) */
export function attendedTokens(model: Model, contextTokens: number): number {
  return model.sparse_attention_topk
    ? Math.min(contextTokens, model.sparse_attention_topk)
    : contextTokens
}

/**
 * Sparse-attention indexer cache (DSA): every token is indexed, so it grows with the full
 * context. Raw bytes: vLLM stores it uint8 (FP8 + scales) whatever the KV precision.
 * shortcut: it rides inside the KV total, so TP splits it by kvCacheTPShards; vLLM replicates
 * it (max_tp_shards=1), which matches only while kvCacheTPShards is 1 (MLA or one KV head, true
 * of every DSA model so far). Split it out if a DSA model shards its KV.
 *
 * @returns GB for `sessions` sessions
 */
export function calculateSparseIndexerVRAM(
  model: Model,
  sequenceLength: number,
  sessions: number,
): Decimal {
  if (!model.sparse_indexer_bytes_per_token) return new Decimal(0)
  return new Decimal(model.sparse_indexer_bytes_per_token)
    .mul(sequenceLength)
    .mul(sessions)
    .div(BYTES_PER_GB)
}

/**
 * Linear-attention / SSM state (Qwen3.6/3.8, Kimi K3 and Linear, Ling 3.0, Nemotron 3,
 * LFM2.5): a fixed conv + recurrent state per session, independent of context length
 * and of KV quantization (vLLM sizes it from the model's own dtype).
 *
 * @returns GB for `sessions` sessions
 */
export function calculateLinearStateVRAM(model: Model, sessions: number): Decimal {
  if (!model.linear_state_bytes_per_session) return new Decimal(0)
  return new Decimal(model.linear_state_bytes_per_session).mul(sessions).div(BYTES_PER_GB)
}

/**
 * Sliding-window / chunked-local layers (Gemma 3/4, gpt-oss, Llama 4, DeepSeek V4):
 * vLLM allocates them at min(window, context) tokens, so they stop growing at the window.
 */
function slidingWindowKV(
  model: Model,
  sequenceLength: number,
  batchSize: number,
  kvPrecision: KVCachePrecision,
): Decimal {
  if (!model.kv_sliding_elements_per_token || !model.kv_sliding_window) return new Decimal(0)
  return new Decimal(model.kv_sliding_elements_per_token)
    .mul(Math.min(model.kv_sliding_window, sequenceLength))
    .mul(batchSize)
    .mul(KV_PRECISION_BYTES[kvPrecision])
    .div(BYTES_PER_GB)
}

/**
 * vLLM stores an FP8 KV cache of a DSA model (dsa_config; here `sparse_attention_topk`) as
 * `fp8_ds_mla`: 512 FP8 + 4 float32 scales + 64 BF16 RoPE = 656 B per token per layer, not 576.
 * Source: vllm/v1/attention/backends/mla/flashmla_sparse.py (header note, main 2026-11-11) and
 * vllm/model_executor/models/config.py (verify_and_update_config maps fp8 -> "fp8_ds_mla" when
 * dsa_config is set). Other MLA models at FP8 keep a plain 1 B per element. On SM10x with no
 * explicit backend vLLM may pick per-tensor fp8 (576 B) instead; we charge 656, the safe side.
 */
const MLA_FP8_DS_KV_BYTES_PER_ELEMENT_FACTOR = new Decimal(656).div(576)

/** Bytes per cached element: the precision's bytes, 656/576 of that for a DSA model at FP8. */
function kvElementBytes(model: Model, kvPrecision: KVCachePrecision): Decimal {
  const bytes = KV_PRECISION_BYTES[kvPrecision]
  return model.use_mla && model.sparse_attention_topk && kvPrecision === 'fp8'
    ? bytes.mul(MLA_FP8_DS_KV_BYTES_PER_ELEMENT_FACTOR)
    : bytes
}

function fullAttentionKV(
  model: Model,
  sequenceLength: number,
  batchSize: number,
  kvPrecision: KVCachePrecision,
): Decimal {
  // Exotic-attention override (MLA, mamba/linear hybrids, explicit head_dim):
  // the model declares its exact per-token cache size; precision applies identically.
  // Constant-state memory (mamba SSM / linear-attention state) does not scale with
  // sequence length and is intentionally excluded.
  if (model.kv_cache_elements_per_token) {
    return new Decimal(model.kv_cache_elements_per_token)
      .mul(sequenceLength)
      .mul(batchSize)
      .mul(kvElementBytes(model, kvPrecision))
      .div(BYTES_PER_GB)
  }

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
