import type { Model } from '@utils/schemas'
import { describe, expect, it } from 'vitest'
import {
  compareModel,
  configFields,
  knownGoodGaps,
  nativeFormat,
  pickReference,
  refDrift,
  sameModel,
  totalGiB,
  weightFiles,
} from '../scripts/model-audit'

const GiB = 1024 ** 3
const model: Model = {
  id: 'google-gemma-4-31b',
  name: 'Gemma 4 31B',
  architecture: 'dense',
  num_parameters_billion: 32.7,
  hidden_size: 5376,
  num_hidden_layers: 60,
  num_attention_heads: 32,
  num_kv_heads: 16,
  intermediate_size: 21504,
  context_length: 262144,
}

describe('configFields', () => {
  it('reads multimodal configs through text_config', () => {
    const cfg = {
      text_config: {
        hidden_size: 5376,
        num_hidden_layers: 60,
        num_attention_heads: 32,
        num_key_value_heads: 16,
        max_position_embeddings: 262144,
      },
    }
    expect(configFields(cfg)).toEqual({
      hidden_size: 5376,
      num_hidden_layers: 60,
      num_attention_heads: 32,
      num_kv_heads: 16,
      context_length: 262144,
    })
  })

  it('reads MoE expert counts under their various names', () => {
    expect(configFields({ n_routed_experts: 256, num_experts_per_tok: 8 })).toMatchObject({
      num_experts: 256,
      num_experts_per_token: 8,
    })
  })
})

describe('compareModel', () => {
  it('reports integer fields that differ and parameters off by more than 1%', () => {
    const drift = compareModel(model, { hidden_size: 5376, num_hidden_layers: 62 }, 34)
    expect(drift).toEqual([
      { field: 'num_hidden_layers', curated: 60, measured: 62 },
      { field: 'num_parameters_billion', curated: 32.7, measured: 34 },
    ])
  })

  it('accepts parameters within 1%', () => {
    expect(compareModel(model, {}, 32.9)).toEqual([])
  })
})

describe('weightFiles', () => {
  it('sums safetensors, skipping original/ and metal/ copies and consolidated duplicates', () => {
    const files = [
      { path: 'model-00001-of-00002.safetensors', size: 5 * GiB },
      { path: 'model-00002-of-00002.safetensors', size: 5 * GiB },
      { path: 'consolidated.safetensors', size: 10 * GiB },
      { path: 'original/model.safetensors', size: 10 * GiB },
      { path: 'metal/model.safetensors', size: 10 * GiB },
      { path: 'config.json', size: 1000 },
    ]
    const picked = weightFiles(files, 'fp8')
    expect(picked !== 'ambiguous' && totalGiB(picked)).toBe(10)
  })

  it('matches the exact GGUF tag, not Q2_K_L / Q2_K_XL, and skips mmproj', () => {
    const files = [
      { path: 'Model-Q2_K.gguf', size: 3 * GiB },
      { path: 'Model-Q2_K_L.gguf', size: 4 * GiB },
      { path: 'Model-Q2_K_XL.gguf', size: 5 * GiB },
      { path: 'mmproj-Q2_K.gguf', size: 1 * GiB },
    ]
    const picked = weightFiles(files, 'gguf-q2_k')
    expect(picked !== 'ambiguous' && totalGiB(picked)).toBe(3)
  })

  it('sums split GGUF shards of one set', () => {
    const files = [
      { path: 'Q4_K_M/Model-Q4_K_M-00001-of-00002.gguf', size: 20 * GiB },
      { path: 'Q4_K_M/Model-Q4_K_M-00002-of-00002.gguf', size: 10 * GiB },
    ]
    const picked = weightFiles(files, 'gguf-q4_k_m')
    expect(picked !== 'ambiguous' && totalGiB(picked)).toBe(30)
  })

  it('reports two file sets for one tag as ambiguous', () => {
    const files = [
      { path: 'Model-Q8_0.gguf', size: 30 * GiB },
      { path: 'Q8_0/Model-Q8_0-00001-of-00002.gguf', size: 15 * GiB },
      { path: 'Q8_0/Model-Q8_0-00002-of-00002.gguf', size: 15 * GiB },
    ]
    expect(weightFiles(files, 'gguf-q8_0')).toBe('ambiguous')
  })

  it('reports a sharded safetensors set plus a standalone full file as ambiguous', () => {
    const files = [
      { path: 'model-00001-of-00002.safetensors', size: 5 * GiB },
      { path: 'model-00002-of-00002.safetensors', size: 5 * GiB },
      { path: 'model.safetensors', size: 10 * GiB },
    ]
    expect(weightFiles(files, 'fp8')).toBe('ambiguous')
  })

  it('reports a sharded safetensors set in a subdirectory plus a root set as ambiguous', () => {
    const files = [
      { path: 'fp8/model-00001-of-00002.safetensors', size: 5 * GiB },
      { path: 'fp8/model-00002-of-00002.safetensors', size: 5 * GiB },
      { path: 'model-00001-of-00002.safetensors', size: 5 * GiB },
      { path: 'model-00002-of-00002.safetensors', size: 5 * GiB },
    ]
    expect(weightFiles(files, 'fp8')).toBe('ambiguous')
  })

  it('returns an empty list when no safetensors files match', () => {
    expect(weightFiles([{ path: 'config.json', size: 1000 }], 'fp8')).toEqual([])
  })
})

describe('nativeFormat', () => {
  it('reads the quantization_config', () => {
    expect(nativeFormat({ quantization_config: { quant_method: 'fp8' } })).toBe('fp8')
    expect(nativeFormat({ quantization_config: { quant_method: 'mxfp4' } })).toBe('mxfp4')
    expect(
      nativeFormat({
        quantization_config: {
          quant_method: 'compressed-tensors',
          config_groups: { g: { weights: { num_bits: 4, type: 'int' } } },
        },
      }),
    ).toBe('int4')
    const float4 = (group: number) => ({
      quantization_config: {
        quant_method: 'compressed-tensors',
        config_groups: { g: { weights: { num_bits: 4, type: 'float', group_size: group } } },
      },
    })
    expect(nativeFormat(float4(32))).toBe('mxfp4')
    expect(nativeFormat(float4(16))).toBe('nvfp4')
  })

  it('falls back to the dtype mix, else bf16', () => {
    expect(nativeFormat({}, { F8_E4M3: 600e9, BF16: 40e9 })).toBe('fp8')
    expect(nativeFormat({}, { BF16: 32e9 })).toBe('bf16')
  })
})

describe('reference selection', () => {
  const candidates = [
    'someone/Gemma-4-31B-IT-NVFP4',
    'nvidia/Gemma-4-31B-IT-NVFP4',
    'RedHatAI/gemma-4-31B-it-FP8-block',
    'cyankiwi/gemma-4-31B-it-AWQ-4bit',
    'bartowski/google_gemma-4-31B-it-GGUF',
    'unsloth/gemma-4-31B-it-GGUF',
  ]

  it('prefers the vendor recipe, then unsloth over bartowski for GGUF', () => {
    expect(pickReference('nvfp4', candidates)).toBe('nvidia/Gemma-4-31B-IT-NVFP4')
    expect(pickReference('fp8', candidates)).toBe('RedHatAI/gemma-4-31B-it-FP8-block')
    expect(pickReference('awq', candidates)).toBe('cyankiwi/gemma-4-31B-it-AWQ-4bit')
    expect(pickReference('gguf-q4_k_m', candidates)).toBe('unsloth/gemma-4-31B-it-GGUF')
    expect(pickReference('gptq', candidates)).toBeNull()
  })

  it('matches derivatives of the same model only', () => {
    expect(sameModel('unsloth/gemma-4-31B-it-GGUF', 'google/gemma-4-31B-it')).toBe(true)
    expect(sameModel('unsloth/gemma-4-12B-it-GGUF', 'google/gemma-4-31B-it')).toBe(false)
    expect(
      sameModel('RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8', 'meta-llama/Llama-3.1-8B-Instruct'),
    ).toBe(true)
    expect(sameModel('unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF', 'deepseek-ai/DeepSeek-R1')).toBe(
      false,
    )
  })
})

describe('refDrift', () => {
  it('passes within 1%, reports beyond, and skips gated repos', () => {
    const files = [{ path: 'model.safetensors', size: 30.4 * GiB }]
    expect(refDrift('fp8', 30.5, files)).toBeNull()
    expect(refDrift('fp8', 28, files)).toMatch(/28.*30.4/)
    expect(refDrift('fp8', 30.4, null)).toBe('skipped (gated)')
  })

  it('reports a lookup failure when no weight files are found', () => {
    expect(refDrift('fp8', 30.4, [{ path: 'config.json', size: 1000 }])).toBe(
      'fp8: no weight files found',
    )
  })
})

describe('knownGoodGaps', () => {
  it('flags MLA, sliding and linear-attention configs missing their curated field', () => {
    expect(knownGoodGaps(model, { kv_lora_rank: 512 })).toContain('use_mla')
    expect(
      knownGoodGaps(model, { layer_types: ['sliding_attention', 'full_attention'] }),
    ).toContain('kv_sliding_window')
    expect(knownGoodGaps(model, { linear_attn_config: { kda_layers: [1] } })).toContain(
      'linear_state_bytes_per_session',
    )
  })

  it('is silent when the curated fields are present', () => {
    const curated: Model = {
      ...model,
      use_mla: true,
      kv_sliding_elements_per_token: 1,
      kv_sliding_window: 1024,
      linear_state_bytes_per_session: 1,
    }
    expect(
      knownGoodGaps(curated, {
        kv_lora_rank: 512,
        layer_types: ['sliding_attention', 'linear_attention'],
      }),
    ).toEqual([])
  })
})
