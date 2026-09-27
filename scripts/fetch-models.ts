/**
 * npm run refresh:models              audit models.json against Hugging Face (drift report)
 * npm run refresh:models -- --strict  same, exit 1 on any drift
 * npm run refresh:models -- --measure <model-id>   print weight_refs JSON to paste
 * npm run refresh:models -- --draft   draft roster ids missing from models.json
 * Never writes models.json. Spec: docs/superpowers/specs/2026-09-27-weight-refs-and-model-audit-design.md
 */
import { writeFile } from 'node:fs/promises'
import curated from '../src/data/models.json' with { type: 'json' }
import type { QuantizationFormat } from '../src/engines/types'
import type { Model } from '../src/utils/schemas'
import { fetchConfig, fetchSafetensors, fetchTree, searchRepos } from './hf'
import {
  compareModel,
  configFields,
  knownGoodGaps,
  MEASURED_FORMATS,
  nativeFormat,
  pickReference,
  refDrift,
  sameModel,
  totalGiB,
  weightFiles,
} from './model-audit'

// Model IDs to fetch — current-generation curated roster (2026-08-18 refresh).
// NOTE: multimodal models (Gemma 4, Qwen3.6, MiniMax M3, Mistral 3) expose the
// transformer fields under config.json `text_config`, which this script does not read;
// and several models are gated or custom-code. Treat fetched output as a starting point
// and hand-curate num_parameters_billion / MoE fields against models.json.
// `kv_cache_elements_per_token` is also hand-curated (MLA latent dims, hybrid
// layer mixes, and explicit head_dim are not derivable from top-level config
// fields) — this script never writes it.
const MODEL_IDS = [
  // Gemma 4 (Google)
  'google/gemma-4-31B-it',
  'google/gemma-4-12B-it',
  'google/gemma-4-26B-A4B-it',

  // Qwen3.8 / Qwen3.6 (Alibaba)
  'Qwen/Qwen3.8-27B',
  'Qwen/Qwen3.8-2.4T-A95B',
  'Qwen/Qwen3.6-27B',
  'Qwen/Qwen3.6-35B-A3B',

  // DeepSeek V4
  'deepseek-ai/DeepSeek-V4-Flash-0731',
  'deepseek-ai/DeepSeek-V4-Pro-0813',

  // GLM (Z.ai)
  'zai-org/GLM-5.2',

  // Kimi (Moonshot)
  'moonshotai/Kimi-K3',
  'moonshotai/Kimi-K2.6',
  'moonshotai/Kimi-K2.7-Code',
  'moonshotai/Kimi-K2-Thinking',
  'moonshotai/Kimi-Linear-48B-A3B-Instruct',

  // Ling (inclusionAI)
  'inclusionAI/Ling-3.0-flash',
  'inclusionAI/Ling-3.0-tiny',

  // Mistral
  'mistralai/Mistral-Medium-3.5-128B',

  // MiniMax
  'MiniMaxAI/MiniMax-M3',

  // NVIDIA Nemotron
  'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
  'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16',

  // Liquid AI
  'LiquidAI/LFM2.5-2.6B',
]

const models = curated as Model[]
const repoOf = (m: Model) => (m.hf_url ?? '').replace('https://huggingface.co/', '')

async function audit(strict: boolean) {
  let problems = 0
  for (const m of models) {
    const repo = repoOf(m)
    const [cfg, st] = await Promise.all([fetchConfig(repo), fetchSafetensors(repo)])
    const lines: string[] = []
    if (!cfg) lines.push('config: skipped (gated or missing)')
    else {
      for (const d of compareModel(m, configFields(cfg), st ? st.total / 1e9 : null)) {
        lines.push(`${d.field}: curated ${d.curated}, measured ${d.measured}`)
      }
      for (const g of knownGoodGaps(m, cfg)) lines.push(`missing curated field: ${g}`)
    }
    for (const [format, ref] of Object.entries(m.weight_refs ?? {})) {
      if (!ref) continue
      const msg = refDrift(format as QuantizationFormat, ref.gib, await fetchTree(ref.repo))
      if (msg) lines.push(`weight_refs.${msg}`)
    }
    if (lines.length) {
      problems += lines.filter((l) => !l.includes('skipped')).length
      console.log(`${m.name}\n  ${lines.join('\n  ')}`)
    }
  }
  console.log(problems ? `\n${problems} drift item(s)` : '\nNo drift')
  if (strict && problems) process.exit(1)
}

async function measureRefs(repo: string) {
  const [cfg, st, tree] = await Promise.all([
    fetchConfig(repo),
    fetchSafetensors(repo),
    fetchTree(repo),
  ])
  const refs: Partial<Record<QuantizationFormat, { repo: string; gib: number }>> = {}
  const native = nativeFormat(cfg ?? {}, st?.parameters)
  const own = tree ? weightFiles(tree, native) : null
  if (own && own !== 'ambiguous' && own.length) refs[native] = { repo, gib: totalGiB(own) }
  const candidates = (await searchRepos(repo.split('/')[1] ?? repo)).filter(
    (c) => c !== repo && sameModel(c, repo),
  )
  const trees = new Map<string, Awaited<ReturnType<typeof fetchTree>>>()
  for (const format of MEASURED_FORMATS) {
    if (refs[format]) continue
    const ref = pickReference(format, candidates)
    if (!ref) continue
    if (!trees.has(ref)) trees.set(ref, await fetchTree(ref))
    const files = trees.get(ref)
    if (!files) continue
    const picked = weightFiles(files, format)
    if (picked === 'ambiguous') {
      console.warn(`WARN ${ref} ${format}: ambiguous file sets, skipped`)
      continue
    }
    if (picked.length) refs[format] = { repo: ref, gib: totalGiB(picked) }
  }
  return refs
}

async function measure(id: string) {
  const m = models.find((x) => x.id === id)
  if (!m) throw new Error(`unknown model id ${id}`)
  console.log(JSON.stringify({ [id]: await measureRefs(repoOf(m)) }, null, 2))
}

async function draft() {
  const known = new Set(models.map(repoOf))
  const drafts = []
  for (const repo of MODEL_IDS.filter((r) => !known.has(r))) {
    const cfg = await fetchConfig(repo)
    const st = await fetchSafetensors(repo)
    if (!cfg) {
      console.warn(`WARN ${repo}: skipped (gated or missing)`)
      continue
    }
    const f = configFields(cfg)
    drafts.push({
      id: repo.replace('/', '-').toLowerCase(),
      name: repo.split('/')[1],
      architecture: f.num_experts ? 'moe' : 'dense',
      num_parameters_billion: st ? Math.round((st.total / 1e9) * 10) / 10 : undefined,
      ...f,
      hf_url: `https://huggingface.co/${repo}`,
      weight_refs: await measureRefs(repo),
    })
  }
  await writeFile('src/data/models-fetched.json', `${JSON.stringify(drafts, null, 2)}\n`)
  console.log(
    `Wrote ${drafts.length} draft(s) to src/data/models-fetched.json; review and merge by hand.`,
  )
}

const args = process.argv.slice(2)
const at = args.indexOf('--measure')
if (at >= 0) await measure(args[at + 1] ?? '')
else if (args.includes('--draft')) await draft()
else await audit(args.includes('--strict'))
