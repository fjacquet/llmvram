/**
 * Pure functions behind `npm run refresh:models`: compare curated models.json entries
 * with Hugging Face configs and checkpoints, and pick / measure reference weight files.
 * No network here (see scripts/hf.ts). Spec:
 * docs/superpowers/specs/2026-09-27-weight-refs-and-model-audit-design.md
 */
import type { QuantizationFormat } from '../src/engines/types'
import type { Model } from '../src/utils/schemas'

export interface RepoFile {
  path: string
  size: number
}

export interface MeasuredFields {
  hidden_size?: number
  num_hidden_layers?: number
  num_attention_heads?: number
  num_kv_heads?: number
  num_experts?: number
  num_experts_per_token?: number
  context_length?: number
}

export interface Drift {
  field: string
  curated: number | undefined
  measured: number
}

type Config = Record<string, unknown>

export const GGUF_TAGS: Partial<Record<QuantizationFormat, string>> = {
  'gguf-q8_0': 'Q8_0',
  'gguf-q6_k': 'Q6_K',
  'gguf-q5_k_m': 'Q5_K_M',
  'gguf-q5_k_s': 'Q5_K_S',
  'gguf-q5_0': 'Q5_0',
  'gguf-q4_k_m': 'Q4_K_M',
  'gguf-q4_k_s': 'Q4_K_S',
  'gguf-q4_0': 'Q4_0',
  'gguf-q3_k_l': 'Q3_K_L',
  'gguf-q3_k_m': 'Q3_K_M',
  'gguf-q3_k_s': 'Q3_K_S',
  'gguf-q2_k': 'Q2_K',
}

/** Formats `--measure` looks for beyond the native release */
export const MEASURED_FORMATS: QuantizationFormat[] = [
  'nvfp4',
  'fp8',
  'int4',
  'awq',
  'gptq',
  ...(Object.keys(GGUF_TAGS) as QuantizationFormat[]),
]

const num = (v: unknown): number | undefined => (typeof v === 'number' ? v : undefined)

export function textConfig(cfg: Config): Config {
  return (cfg.text_config ?? cfg.llm_config ?? cfg.language_config ?? cfg) as Config
}

export function configFields(cfg: Config): MeasuredFields {
  const t = textConfig(cfg)
  const first = (...keys: string[]) => keys.map((k) => num(t[k])).find((v) => v !== undefined)
  const fields: MeasuredFields = {
    hidden_size: first('hidden_size', 'd_model', 'dim'),
    num_hidden_layers: first('num_hidden_layers', 'n_layers', 'num_layers'),
    num_attention_heads: first('num_attention_heads', 'n_heads'),
    num_kv_heads: first('num_key_value_heads', 'n_kv_heads'),
    num_experts: first('n_routed_experts', 'num_experts', 'num_local_experts'),
    num_experts_per_token: first('num_experts_per_tok', 'num_experts_per_token', 'moe_topk'),
    context_length: first('max_position_embeddings') ?? num(cfg.max_position_embeddings),
  }
  return Object.fromEntries(
    Object.entries(fields).filter(([, v]) => v !== undefined),
  ) as MeasuredFields
}

export function compareModel(
  model: Model,
  fields: MeasuredFields,
  paramsB: number | null,
): Drift[] {
  const drift: Drift[] = []
  for (const [field, measured] of Object.entries(fields) as [keyof MeasuredFields, number][]) {
    const curated = model[field]
    if (curated !== measured) drift.push({ field, curated, measured })
  }
  if (
    paramsB !== null &&
    Math.abs(paramsB - model.num_parameters_billion) > model.num_parameters_billion * 0.01
  ) {
    drift.push({
      field: 'num_parameters_billion',
      curated: model.num_parameters_billion,
      measured: paramsB,
    })
  }
  return drift
}

export function weightFiles(
  files: RepoFile[],
  format: QuantizationFormat,
): RepoFile[] | 'ambiguous' {
  const tag = GGUF_TAGS[format]
  if (tag) {
    const re = new RegExp(`(^|[-_./])${tag}(\\.gguf$|-\\d{5}-of-|/)`, 'i')
    const hits = files.filter(
      (f) => f.path.endsWith('.gguf') && !/mmproj|draft|UD-|IQ\d/i.test(f.path) && re.test(f.path),
    )
    // One file set = one name once the shard suffix is removed
    const sets = new Set(hits.map((f) => f.path.replace(/-\d{5}-of-\d{5}\.gguf$/, '.gguf')))
    return sets.size > 1 ? 'ambiguous' : hits
  }
  const st = files.filter(
    (f) => f.path.endsWith('.safetensors') && !/^(original|metal)\//.test(f.path),
  )
  const hf = st.filter((f) => !/consolidated/.test(f.path))
  return hf.length ? hf : st
}

export function totalGiB(files: RepoFile[]): number {
  return Math.round((files.reduce((s, f) => s + f.size, 0) / 1024 ** 3) * 100) / 100
}

export function nativeFormat(cfg: Config, dtypes?: Record<string, number>): QuantizationFormat {
  const q = (cfg.quantization_config ?? textConfig(cfg).quantization_config) as Config | undefined
  if (q) {
    const method = String(q.quant_method ?? '').toLowerCase()
    const s = JSON.stringify(q)
    if (method === 'fp8' || method === 'fbgemm_fp8') return 'fp8'
    if (method === 'mxfp4') return 'mxfp4'
    if (/nvfp4/i.test(s)) return 'nvfp4'
    if (/"num_bits":\s*4/.test(s) && /"type":\s*"int"/.test(s)) return 'int4'
    // 4-bit float: NVFP4 scales per 16 values, MXFP4 per 32 (Kimi K3 native)
    if (/"num_bits":\s*4/.test(s) && /"type":\s*"float"/.test(s)) {
      return /"group_size":\s*16\b/.test(s) ? 'nvfp4' : 'mxfp4'
    }
    if (/"num_bits":\s*8/.test(s) && /"type":\s*"float"/.test(s)) return 'fp8'
  }
  if (dtypes) {
    const total = Object.values(dtypes).reduce((a, b) => a + b, 0)
    const f8 = Object.entries(dtypes)
      .filter(([k]) => /F8/.test(k))
      .reduce((a, [, v]) => a + v, 0)
    if (total > 0 && f8 / total > 0.5) return 'fp8'
  }
  return 'bf16'
}

const QUANT_SUFFIX =
  /-(gguf|fp8(-dynamic|-block)?|nvfp4|fp4|awq(-4bit)?|gptq(-int4)?|int4|w4a16|4bit|dynamic|instruct|it|bf16|mlx)\b/g

const baseName = (repo: string) =>
  (repo.split('/')[1] ?? repo)
    .toLowerCase()
    .replace(/^(google|meta|qwen|mistralai|deepseek-ai|moonshotai)_/, '')
    .replace(/^meta-(?=llama)/, '')
    .replace(QUANT_SUFFIX, '')
    .replace(/-\d{4}$/, '')

export function sameModel(candidate: string, baseRepo: string): boolean {
  return baseName(candidate) === baseName(baseRepo)
}

export function pickReference(format: QuantizationFormat, candidates: string[]): string | null {
  const find = (...res: RegExp[]) => {
    for (const re of res) {
      const hit = candidates.find((c) => re.test(c))
      if (hit) return hit
    }
    return null
  }
  if (GGUF_TAGS[format]) return find(/^unsloth\/.*GGUF$/i, /^bartowski\/.*GGUF$/i)
  switch (format) {
    case 'nvfp4':
      return find(/^nvidia\/.*NVFP4/i, /NVFP4/i)
    case 'fp8':
      return find(/^RedHatAI\/.*FP8/i, /FP8/i)
    case 'int4':
      return find(/^RedHatAI\/.*(w4a16|INT4)/i, /^(?!.*(AWQ|GPTQ)).*(w4a16|INT4)/i)
    case 'awq':
      return find(/AWQ/i)
    case 'gptq':
      return find(/GPTQ/i)
    default:
      return null
  }
}

export function knownGoodGaps(model: Model, cfg: Config): string[] {
  const t = textConfig(cfg)
  const types = JSON.stringify(t.layer_types ?? [])
  const gaps: string[] = []
  if (t.kv_lora_rank && !model.use_mla) gaps.push('use_mla')
  if (/sliding|chunk/.test(types) && !model.kv_sliding_window) gaps.push('kv_sliding_window')
  const linear =
    t.linear_attn_config !== undefined ||
    /linear|conv|mamba/.test(types) ||
    /M/.test(String(t.hybrid_override_pattern ?? ''))
  if (linear && !model.linear_state_bytes_per_session) gaps.push('linear_state_bytes_per_session')
  return gaps
}

export function refDrift(
  format: QuantizationFormat,
  curatedGiB: number,
  files: RepoFile[] | null,
): string | null {
  if (files === null) return 'skipped (gated)'
  const picked = weightFiles(files, format)
  if (picked === 'ambiguous') return `${format}: ambiguous file sets`
  const measured = totalGiB(picked)
  if (Math.abs(measured - curatedGiB) <= curatedGiB * 0.01) return null
  return `${format}: curated ${curatedGiB} GiB, measured ${measured} GiB`
}
