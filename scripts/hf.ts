/** Hugging Face Hub access for the model auditor. Token: HF_TOKEN or ~/.cache/huggingface/token. */
import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import type { RepoFile } from './model-audit'

function readToken(): string | undefined {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN
  try {
    return readFileSync(`${homedir()}/.cache/huggingface/token`, 'utf8').trim()
  } catch {
    return undefined
  }
}

const token = readToken()
const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {}

async function getJSON(url: string): Promise<unknown | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const r = await fetch(url, { headers })
    if (r.ok) return r.json()
    if (r.status !== 429) return null
    await new Promise((s) => setTimeout(s, 2000 * (attempt + 1)))
  }
  return null
}

export async function fetchConfig(repo: string): Promise<Record<string, unknown> | null> {
  return (await getJSON(`https://huggingface.co/${repo}/resolve/main/config.json`)) as Record<
    string,
    unknown
  > | null
}

export async function fetchSafetensors(
  repo: string,
): Promise<{ total: number; parameters: Record<string, number> } | null> {
  const info = (await getJSON(
    `https://huggingface.co/api/models/${repo}?expand[]=safetensors`,
  )) as {
    safetensors?: { total: number; parameters: Record<string, number> }
  } | null
  return info?.safetensors ?? null
}

export async function fetchTree(repo: string): Promise<RepoFile[] | null> {
  const t = await getJSON(`https://huggingface.co/api/models/${repo}/tree/main?recursive=true`)
  if (!Array.isArray(t)) return null
  return t
    .filter((f: { type: string }) => f.type === 'file')
    .map((f: { path: string; size: number; lfs?: { size: number } }) => ({
      path: f.path,
      size: f.lfs?.size ?? f.size,
    }))
}

export async function searchRepos(term: string): Promise<string[]> {
  const r = await getJSON(
    `https://huggingface.co/api/models?search=${encodeURIComponent(term)}&limit=100&sort=downloads`,
  )
  return Array.isArray(r) ? r.map((m: { id: string }) => m.id) : []
}
