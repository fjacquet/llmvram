import gpusData from '@data/gpus.json'
import modelsData from '@data/models.json'
import { DEFAULT_KV_TIER } from '@engines/kv-tier'
import type { GPU, Model } from '@utils/schemas'
import { validateGPUs, validateModels } from '@utils/schemas'
import { compressToEncodedURIComponent } from 'lz-string'
import { beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * The real uiStore wraps its state in zustand's `persist` middleware, which
 * needs localStorage — unavailable in this jsdom environment (see
 * KVTierPanel.test.tsx / NodeCountSelector.test.tsx for the same constraint).
 * Mocking `persist` as a pass-through (the real state creator, no storage)
 * lets this file exercise the ACTUAL setSelectedGPU/setKVTier from
 * uiStore.ts, not a hand-written reimplementation — a reimplementation can
 * silently drift from the real logic and pass even when the real store has
 * the bug (as happened here: an earlier version of this file mirrored the
 * store and never caught that setKVTier itself lacked the Grace guard).
 */
vi.mock('zustand/middleware', async (importOriginal) => {
  const actual = await importOriginal<typeof import('zustand/middleware')>()
  return {
    ...actual,
    persist: (config: unknown) => config,
  }
})

// jsdom has no matchMedia; uiStore reads it once at module init for isDarkMode.
beforeEach(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: vi.fn((query: string) => ({
      matches: false,
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
})

const gpus = validateGPUs(gpusData)

/** A real GPU row from the database, looked up by id (never hand-written). */
function findGPU(id: string): GPU {
  const gpu = gpus.find((g) => g.id === id)
  if (!gpu) throw new Error(`fixture GPU not found in gpus.json: ${id}`)
  return gpu
}

const models = validateModels(modelsData)

/** A real model row from the database, looked up by id (never hand-written). */
function findModel(id: string): Model {
  const model = models.find((m) => m.id === id)
  if (!model) throw new Error(`fixture model not found in models.json: ${id}`)
  return model
}

const L70 = findModel('meta-llama-llama-3.1-70b')
const DSR1 = findModel('deepseek-r1')
const H100 = findGPU('nvidia-h100-80gb-pcie')
const NVL72 = findGPU('nvidia-gb300-nvl72')
const UNIFIED_M3 = findGPU('apple-m3-ultra')

async function freshStore() {
  const { useUIStore, DEFAULT_UI_CONFIG } = await import('@store/uiStore')
  useUIStore.setState({ ...DEFAULT_UI_CONFIG, pendingNotice: null })
  return useUIStore
}

describe('uiStore: host-grace KV tier guard', () => {
  beforeEach(async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.setState({ mode: 'inference', selectedGPU: null, kvTier: DEFAULT_KV_TIER })
  })

  it('URL restore order (GPU first, then tier): falls back to none for a non-Grace GPU', async () => {
    const { useUIStore } = await import('@store/uiStore')
    // useURLSync restores the GPU, then the tier, from the same hash.
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-h100-80gb-pcie'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().kvTier.tier).toBe('none')
  })

  it('URL restore order: falls back to none for the x86 HGX B300 too', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-gb300-288gb'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().kvTier.tier).toBe('none')
  })

  // 2026-11-11, sizing T5: was "keeps host-grace for a Grace-host GPU" in inference. The only
  // Grace-host GPUs that are not unified memory are racks, now selectable in sizing only;
  // selecting one switches the mode, and R3 keeps the tier there (it has a Grace link).
  it('a Grace-host rack GPU switches to sizing and keeps host-grace', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-gb300-nvl72'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    expect(useUIStore.getState().mode).toBe('sizing')
    expect(useUIStore.getState().kvTier.tier).toBe('host-grace')
  })

  // 2026-11-11, mode split: Sizing lists rack GPUs only, so a server GPU picked there is
  // replaced by the first rack GPU (GB300, a Grace host) and the tier stays valid.
  it('a server GPU in Sizing becomes the default rack GPU, with a notice, and keeps the tier', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-gb300-nvl72'))
    useUIStore.getState().setKVTier({ tier: 'host-grace' })
    useUIStore.getState().setSelectedGPU(findGPU('nvidia-h100-80gb-pcie'))
    expect(useUIStore.getState().mode).toBe('sizing')
    expect(useUIStore.getState().selectedGPU?.id).toBe('nvidia-gb300-nvl72')
    expect(useUIStore.getState().kvTier.tier).toBe('host-grace')
    expect(useUIStore.getState().pendingNotice?.lines.join(' ')).toMatch(
      /Sizing lists rack GPUs only/,
    )
  })
})

describe('uiStore: Sizing mode and server GPUs (mode split, 2026-11-11)', () => {
  it('a shared sizing link with an H100 restores with the default rack GPU, adjusted', async () => {
    const store = await freshStore()
    const { deserializeFromURL, urlStateToConfig } = await import('@store/urlSerializer')
    const { findGPUById, findModelById } = await import('@store/uiStore')
    const json = {
      q: 'fp16',
      sl: 4096,
      bs: 1,
      kvq: 'fp16',
      ng: 1,
      ss: 'tensor-parallel',
      m: 'sizing',
      modelId: L70.id,
      gpuId: H100.id,
      kt: { t: 'host-grace', a: 0.25, b: 30 },
    }
    const decoded = deserializeFromURL(compressToEncodedURIComponent(JSON.stringify(json)))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, { findModel: findModelById, findGPU: findGPUById })
    store.getState().restoreConfig(patch)
    expect(store.getState().mode).toBe('sizing')
    expect(store.getState().selectedGPU?.id).toBe(NVL72.id)
    expect(store.getState().pendingNotice?.title).toBe('Shared link adjusted')
  })
})

describe('uiStore: offloadHostCapacityGB clamp', () => {
  beforeEach(async () => {
    const { useUIStore } = await import('@store/uiStore')
    // 2026-11-11, sizing T5: reset mode too; selecting a rack GPU above now switches to sizing
    useUIStore.setState({ mode: 'inference', selectedGPU: null, offloadHostCapacityGB: null })
  })

  it('accepts a positive finite value', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setOffloadHostCapacityGB(4096)
    expect(useUIStore.getState().offloadHostCapacityGB).toBe(4096)
  })

  it('clamps non-finite, zero, negative, and null to null', async () => {
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setOffloadHostCapacityGB(4096)
    useUIStore.getState().setOffloadHostCapacityGB(Number.NaN)
    expect(useUIStore.getState().offloadHostCapacityGB).toBeNull()

    useUIStore.getState().setOffloadHostCapacityGB(4096)
    useUIStore.getState().setOffloadHostCapacityGB(0)
    expect(useUIStore.getState().offloadHostCapacityGB).toBeNull()

    useUIStore.getState().setOffloadHostCapacityGB(4096)
    useUIStore.getState().setOffloadHostCapacityGB(-10)
    expect(useUIStore.getState().offloadHostCapacityGB).toBeNull()

    useUIStore.getState().setOffloadHostCapacityGB(4096)
    useUIStore.getState().setOffloadHostCapacityGB(null)
    expect(useUIStore.getState().offloadHostCapacityGB).toBeNull()
  })
})

describe('uiStore: every action normalizes, one notice per action', () => {
  it('setNumGPUs(6) on Llama 3.1 70B snaps to 4 with one notice titled for the model', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(6)
    const state = store.getState()
    expect(state.numGPUs).toBe(4)
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${L70.name}`)
    expect(state.pendingNotice?.lines).toHaveLength(1)
  })

  it('a GPU switch reports the count clamp and the Grace tier reset in ONE notice', async () => {
    const store = await freshStore()
    store.setState({
      selectedModel: L70,
      selectedGPU: NVL72,
      numGPUs: 64,
      kvTier: { ...DEFAULT_KV_TIER, tier: 'host-grace' },
    })
    store.getState().setSelectedGPU(H100)
    const state = store.getState()
    expect(state.numGPUs).toBe(8)
    expect(state.kvTier.tier).toBe('none')
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${H100.name}`)
    expect(state.pendingNotice?.lines).toHaveLength(2)
  })

  it('leaves pendingNotice untouched when nothing is corrected', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(4)
    expect(store.getState().pendingNotice).toBeNull()
  })

  it('gives an identical repeated notice a new id, so it toasts again', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(6)
    const first = store.getState().pendingNotice
    store.getState().setNumGPUs(6)
    const second = store.getState().pendingNotice
    expect(second?.lines).toEqual(first?.lines)
    expect(second?.id).not.toBe(first?.id)
  })

  it('model change to a dense model resets EP and snaps 6 -> 4 in one notice (R2 then R14)', async () => {
    const store = await freshStore()
    store.setState({
      selectedModel: DSR1,
      selectedGPU: H100,
      shardingStrategy: 'expert-parallel',
      numGPUs: 6,
    })
    store.getState().setSelectedModel(L70)
    const state = store.getState()
    expect(state.shardingStrategy).toBe('tensor-parallel')
    expect(state.numGPUs).toBe(4)
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${L70.name}`)
    expect(state.pendingNotice?.lines).toHaveLength(2)
  })

  it('training ZeRO-3 on 6 GPUs keeps 6 silently; switching to inference snaps to 4 with a mode notice', async () => {
    const store = await freshStore()
    store.setState({
      selectedModel: L70,
      selectedGPU: H100,
      mode: 'training',
      frameworkPreset: 'deepspeed-zero3',
    })
    store.getState().setNumGPUs(6)
    expect(store.getState().numGPUs).toBe(6)
    expect(store.getState().pendingNotice).toBeNull()
    store.getState().setMode('inference')
    expect(store.getState().numGPUs).toBe(4)
    expect(store.getState().pendingNotice?.title).toBe('Adjusted for inference mode')
  })

  it('picking vLLM in training switches to inference (action intent) and keeps the preset', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, mode: 'training' })
    store.getState().setFrameworkPreset('vllm')
    expect(store.getState().mode).toBe('inference')
    expect(store.getState().frameworkPreset).toBe('vllm')
  })

  it('switching to training with vLLM clears the preset (R7), titled for the mode', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, frameworkPreset: 'vllm' })
    store.getState().setMode('training')
    expect(store.getState().frameworkPreset).toBe('none')
    expect(store.getState().pendingNotice?.title).toBe('Adjusted for fine-tuning mode')
  })

  it('switching ZeRO-3 -> Unsloth turns optimizer offload off (R8) and keeps auto-optimizations', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, mode: 'training' })
    store.getState().setFrameworkPreset('deepspeed-zero3')
    store.getState().setCpuOffloadOptimizer(true)
    expect(store.getState().cpuOffloadOptimizer).toBe(true)
    store.getState().setFrameworkPreset('unsloth')
    const state = store.getState()
    expect(state.cpuOffloadOptimizer).toBe(false)
    expect(state.optimizer).toBe('adamw-8bit')
    expect(state.pendingNotice?.lines).toEqual([
      'CPU optimizer offload turned off: needs a DeepSpeed ZeRO preset.',
    ])
  })

  it('enabling offloading on unified memory with target cpu-ram switches to NVMe, with a notice', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: UNIFIED_M3, offloadTarget: 'cpu-ram' })
    store.getState().setOffloadingEnabled(true)
    const state = store.getState()
    expect(state.offloadingEnabled).toBe(true)
    expect(state.offloadTarget).toBe('nvme')
    expect(state.pendingNotice?.title).toBe(`Adjusted for ${UNIFIED_M3.name}`)
    expect(state.pendingNotice?.lines).toEqual([
      `Offload target set to NVMe: ${UNIFIED_M3.name} has unified memory, RAM is the same pool.`,
    ])
  })

  it('switching to a unified-memory GPU while offloading is off produces no notice', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100, offloadingEnabled: false })
    store.getState().setSelectedGPU(UNIFIED_M3)
    const state = store.getState()
    expect(state.selectedGPU).toBe(UNIFIED_M3)
    expect(state.offloadingEnabled).toBe(false)
    expect(state.pendingNotice).toBeNull()
  })

  it('clearNotice empties the channel', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().setNumGPUs(6)
    store.getState().clearNotice()
    expect(store.getState().pendingNotice).toBeNull()
  })

  it('resetAdvancedSettings resets exactly the Advanced-section fields, keeps the essentials, one line per changed group', async () => {
    const store = await freshStore()
    const { DEFAULT_UI_CONFIG } = await import('@store/uiStore')
    store.setState({
      selectedModel: L70,
      selectedGPU: H100,
      numGPUs: 4,
      numNodes: 2,
      quantization: 'fp8',
      sequenceLength: 8192,
      contextMix: [{ tokens: 8192, share: 1 }],
      concurrentUsers: 10,
      batchSize: 8,
      kvQuantization: 'fp8',
      shardingStrategy: 'pipeline-parallel',
      offloadingEnabled: true,
      offloadTarget: 'nvme',
      kvTier: { ...DEFAULT_KV_TIER, tier: 'network' },
    })
    store.getState().resetAdvancedSettings()
    const state = store.getState()
    // Advanced fields back to their store defaults
    expect(state.batchSize).toBe(DEFAULT_UI_CONFIG.batchSize)
    expect(state.kvQuantization).toBe(DEFAULT_UI_CONFIG.kvQuantization)
    expect(state.shardingStrategy).toBe(DEFAULT_UI_CONFIG.shardingStrategy)
    expect(state.offloadingEnabled).toBe(false)
    expect(state.offloadTarget).toBe(DEFAULT_UI_CONFIG.offloadTarget)
    expect(state.kvTier).toEqual(DEFAULT_UI_CONFIG.kvTier)
    // Essentials kept exactly as set
    expect(state.selectedModel).toBe(L70)
    expect(state.selectedGPU).toBe(H100)
    expect(state.numGPUs).toBe(4)
    expect(state.numNodes).toBe(2)
    expect(state.quantization).toBe('fp8')
    expect(state.sequenceLength).toBe(8192)
    expect(state.concurrentUsers).toBe(10)
    // One notice, one line per group that actually changed (not per raw field)
    expect(state.pendingNotice?.title).toBe('Reset to defaults')
    expect(state.pendingNotice?.lines).toEqual([
      'Batch size reset to 1.',
      'KV precision reset to FP16.',
      'Sharding strategy reset to tensor parallel.',
      'Offloading reset to off.',
      'KV tier reset to none.',
    ])
  })

  it('resetAdvancedSettings in training only resets batch size: the other Advanced inputs are hidden as inert', async () => {
    const store = await freshStore()
    store.setState({
      selectedModel: L70,
      selectedGPU: H100,
      mode: 'training',
      batchSize: 8,
      kvQuantization: 'fp8',
      offloadingEnabled: true,
    })
    store.getState().resetAdvancedSettings()
    const state = store.getState()
    expect(state.batchSize).toBe(1)
    expect(state.kvQuantization).toBe('fp8')
    expect(state.offloadingEnabled).toBe(true)
    expect(state.pendingNotice?.lines).toEqual(['Batch size reset to 1.'])
  })

  it('a reset that cascades into a rule still surfaces as one notice (R14 after strategy resets to tensor-parallel)', async () => {
    const store = await freshStore()
    store.setState({
      selectedModel: L70,
      selectedGPU: H100,
      numGPUs: 6,
      shardingStrategy: 'pipeline-parallel',
    })
    store.getState().resetAdvancedSettings()
    const state = store.getState()
    // shardingStrategy resets to tensor-parallel, which R14 then finds numGPUs=6
    // invalid for (64 heads); "keeps GPU count" yields to normalizeConfig here.
    expect(state.numGPUs).toBe(4)
    expect(state.pendingNotice?.title).toBe('Reset to defaults')
    expect(state.pendingNotice?.lines).toEqual([
      'Sharding strategy reset to tensor parallel.',
      `GPU count set to 4: vLLM can't split ${L70.name}'s 64 attention heads across 6 GPUs. Use pipeline parallel for 6.`,
    ])
  })

  it('resetAdvancedSettings produces no notice when the advanced fields are already at their defaults', async () => {
    const store = await freshStore()
    store.setState({ selectedModel: L70, selectedGPU: H100 })
    store.getState().resetAdvancedSettings()
    expect(store.getState().pendingNotice).toBeNull()
  })

  it('resetAdvancedSettings with numNodes 1 resets a non-default fabric silently, no fabric line (item D)', async () => {
    const store = await freshStore()
    const { DEFAULT_UI_CONFIG } = await import('@store/uiStore')
    store.setState({
      selectedModel: L70,
      selectedGPU: H100,
      numNodes: 1,
      interNodeFabric: 'infiniband-xdr',
    })
    store.getState().resetAdvancedSettings()
    const state = store.getState()
    // Reset silently: the fabric field itself still goes back to default...
    expect(state.interNodeFabric).toBe(DEFAULT_UI_CONFIG.interNodeFabric)
    // ...but since it's inert at numNodes 1 (countAdvancedChanges' own gate), it must
    // not produce a notice line, and nothing else changed, so there's no notice at all.
    expect(state.pendingNotice).toBeNull()
  })

  it('resetAll clears a stale pendingNotice even when already at defaults', async () => {
    const store = await freshStore()
    store.setState({ pendingNotice: { title: 'Adjusted for something', lines: ['x'], id: 1 } })
    store.getState().resetAll()
    expect(store.getState().pendingNotice).toBeNull()
  })

  it('resetAll returns to the initial empty state, clears the hash, with one line', async () => {
    const store = await freshStore()
    const { DEFAULT_UI_CONFIG } = await import('@store/uiStore')
    store.setState({
      selectedModel: L70,
      selectedGPU: H100,
      numGPUs: 4,
      batchSize: 8,
    })
    window.history.replaceState(null, '', '#some-encoded-state')
    expect(window.location.hash).toBe('#some-encoded-state')
    store.getState().resetAll()
    const state = store.getState()
    expect(state.selectedModel).toBeNull()
    expect(state.selectedGPU).toBeNull()
    expect(state.numGPUs).toBe(DEFAULT_UI_CONFIG.numGPUs)
    expect(state.batchSize).toBe(DEFAULT_UI_CONFIG.batchSize)
    expect(state.pendingNotice?.title).toBe('Reset to defaults')
    expect(state.pendingNotice?.lines).toEqual(['Configuration reset to defaults.'])
    expect(window.location.hash).toBe('')
  })

  it('resetAll produces no notice from a fresh store', async () => {
    const store = await freshStore()
    store.getState().resetAll()
    expect(store.getState().pendingNotice).toBeNull()
  })
})

describe('uiStore: shared-link restore is one normalized action', () => {
  async function restore(json: Record<string, unknown>) {
    const store = await freshStore()
    const { deserializeFromURL, urlStateToConfig } = await import('@store/urlSerializer')
    const { findGPUById, findModelById } = await import('@store/uiStore')
    const decoded = deserializeFromURL(compressToEncodedURIComponent(JSON.stringify(json)))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, { findModel: findModelById, findGPU: findGPUById })
    store.getState().restoreConfig(patch)
    return store.getState()
  }
  const base = { q: 'fp16', sl: 4096, bs: 1, kvq: 'fp16', ng: 1, ss: 'tensor-parallel' }

  it('opens a hostile link corrected, with one "Shared link adjusted" notice', async () => {
    const state = await restore({
      ...base,
      modelId: L70.id,
      gpuId: H100.id,
      ss: 'expert-parallel',
      ng: 6,
      sl: 100,
      bs: 0,
    })
    expect(state.shardingStrategy).toBe('tensor-parallel')
    expect(state.numGPUs).toBe(4)
    expect(state.sequenceLength).toBe(512)
    expect(state.batchSize).toBe(1)
    expect(state.pendingNotice?.title).toBe('Shared link adjusted')
    expect(state.pendingNotice?.lines).toHaveLength(4)
  })

  it('a link with 2 GPUs on an HGX baseboard restores as the 8-GPU machine, adjusted', async () => {
    const state = await restore({
      ...base,
      modelId: L70.id,
      gpuId: findGPU('nvidia-h100-80gb-sxm').id,
      ng: 2,
    })
    expect(state.numGPUs).toBe(8)
    expect(state.pendingNotice?.title).toBe('Shared link adjusted')
    expect(state.pendingNotice?.lines.join(' ')).toMatch(/8-GPU machine/)
  })

  it('selecting a baseboard from 1 GPU sets 8 with an "Adjusted for" notice', async () => {
    const store = await freshStore()
    store.getState().setSelectedGPU(findGPU('nvidia-h100-80gb-sxm'))
    expect(store.getState().numGPUs).toBe(8)
    expect(store.getState().pendingNotice?.title).toBe('Adjusted for NVIDIA H100 80GB SXM')
  })

  it('clamps non-integer and negative counts', async () => {
    expect((await restore({ ...base, modelId: L70.id, gpuId: H100.id, ng: 2.5 })).numGPUs).toBe(2)
    expect((await restore({ ...base, modelId: L70.id, gpuId: H100.id, ng: -3 })).numGPUs).toBe(1)
    const clamped = await restore({ ...base, modelId: L70.id, gpuId: H100.id, nn: 12, cu: 0 })
    expect(clamped.numNodes).toBe(8)
    expect(clamped.concurrentUsers).toBe(1)
  })

  it('keeps EP and 6 GPUs when the link model is unknown; the first model pick then corrects both', async () => {
    const state = await restore({
      ...base,
      modelId: 'no-such-model',
      gpuId: H100.id,
      ss: 'expert-parallel',
      ng: 6,
    })
    expect(state.selectedModel).toBeNull()
    expect(state.shardingStrategy).toBe('expert-parallel')
    expect(state.numGPUs).toBe(6)
    const { useUIStore } = await import('@store/uiStore')
    useUIStore.getState().setSelectedModel(L70)
    expect(useUIStore.getState().shardingStrategy).toBe('tensor-parallel')
    expect(useUIStore.getState().numGPUs).toBe(4)
    expect(useUIStore.getState().pendingNotice?.lines).toHaveLength(2)
  })

  it('restores training settings that links carried but never restored (ga, gc, fa, co)', async () => {
    const state = await restore({
      ...base,
      modelId: L70.id,
      gpuId: H100.id,
      m: 'training',
      ga: 8,
      gc: true,
      fa: true,
      fp: 'deepspeed-zero3',
      co: true,
    })
    expect(state).toMatchObject({
      mode: 'training',
      gradientAccumulationSteps: 8,
      gradientCheckpointing: true,
      flashAttention: true,
      frameworkPreset: 'deepspeed-zero3',
      cpuOffloadOptimizer: true,
    })
    expect(state.pendingNotice).toBeNull()
  })

  it('a link with offloading on, target cpu-ram, on a unified-memory GPU restores with offloading OFF (R6), not NVMe', async () => {
    const state = await restore({
      ...base,
      modelId: L70.id,
      gpuId: UNIFIED_M3.id,
      oe: true,
      ot: 'cpu-ram',
    })
    // Unlike setOffloadingEnabled's action intent (unit test above), a link restore
    // applies the plain rule: R6 turns offloading off, it does not pick NVMe.
    expect(state.offloadingEnabled).toBe(false)
    expect(state.offloadTarget).toBe('cpu-ram')
    expect(state.pendingNotice?.title).toBe('Shared link adjusted')
    expect(state.pendingNotice?.lines).toEqual([
      `Offloading turned off: ${UNIFIED_M3.name} has unified memory, RAM is the same pool.`,
    ])
  })

  it('gives the same result whatever order the link keys come in', async () => {
    const link = {
      ...base,
      modelId: L70.id,
      gpuId: H100.id,
      ss: 'expert-parallel',
      ng: 6,
      bs: 0,
      sl: 100,
      nn: 12,
    }
    const entries = Object.entries(link)
    const orders = [
      entries,
      [...entries].reverse(),
      [...entries].sort(([a], [b]) => a.localeCompare(b)),
    ]
    const results = []
    for (const order of orders) {
      const state = await restore(Object.fromEntries(order))
      results.push({
        numGPUs: state.numGPUs,
        shardingStrategy: state.shardingStrategy,
        batchSize: state.batchSize,
        sequenceLength: state.sequenceLength,
        numNodes: state.numNodes,
        lines: state.pendingNotice?.lines,
      })
    }
    expect(results[1]).toEqual(results[0])
    expect(results[2]).toEqual(results[0])
  })
})

describe('uiStore: sizing mode and rack GPUs (2026-11-11, sizing T5)', () => {
  it('setMode(inference) with a rack GPU selected clears it, with a notice', async () => {
    const store = await freshStore()
    store.getState().setMode('sizing')
    store.getState().setSelectedGPU(NVL72)
    store.getState().clearNotice()
    store.getState().setMode('inference')
    const state = store.getState()
    expect(state.mode).toBe('inference')
    expect(state.selectedGPU).toBeNull()
    expect(state.pendingNotice?.lines.join(' ')).toContain('is sized as a rack; pick a server GPU')
  })

  it('setMode(training) with a rack GPU selected clears it too', async () => {
    const store = await freshStore()
    store.getState().setMode('sizing')
    store.getState().setSelectedGPU(NVL72)
    store.getState().setMode('training')
    expect(store.getState().selectedGPU).toBeNull()
  })

  it('setMode(sizing) with a server GPU swaps in the default rack GPU, with a notice, no loop', async () => {
    const store = await freshStore()
    store.getState().setSelectedGPU(H100)
    store.getState().setMode('sizing')
    const state = store.getState()
    expect(state.mode).toBe('sizing')
    expect(state.selectedGPU?.id).toBe(NVL72.id)
    expect(state.pendingNotice?.lines.join(' ')).toContain('Sizing lists rack GPUs only')
  })

  it('setMode(inference) with a server GPU keeps it and writes no notice', async () => {
    const store = await freshStore()
    store.getState().setMode('inference')
    store.getState().setSelectedGPU(H100)
    store.getState().setMode('inference')
    expect(store.getState().selectedGPU?.id).toBe(H100.id)
  })

  it('a link with m: inference and a rack GPU restores in sizing, "Shared link adjusted"', async () => {
    const store = await freshStore()
    const { deserializeFromURL, urlStateToConfig } = await import('@store/urlSerializer')
    const { findGPUById, findModelById } = await import('@store/uiStore')
    const json = {
      q: 'fp16',
      sl: 4096,
      bs: 1,
      kvq: 'fp16',
      ng: 1,
      ss: 'tensor-parallel',
      m: 'inference',
      modelId: L70.id,
      gpuId: NVL72.id,
    }
    const decoded = deserializeFromURL(compressToEncodedURIComponent(JSON.stringify(json)))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, { findModel: findModelById, findGPU: findGPUById })
    store.getState().restoreConfig(patch)
    const state = store.getState()
    expect(state.mode).toBe('sizing')
    expect(state.selectedGPU?.id).toBe(NVL72.id)
    expect(state.pendingNotice?.title).toBe('Shared link adjusted')
  })
})

describe('uiStore: context mix and TTFT target', () => {
  async function restore(json: Record<string, unknown>) {
    const store = await freshStore()
    const { deserializeFromURL, urlStateToConfig } = await import('@store/urlSerializer')
    const { findGPUById, findModelById } = await import('@store/uiStore')
    const decoded = deserializeFromURL(compressToEncodedURIComponent(JSON.stringify(json)))
    if (!decoded) throw new Error('expected the link to parse')
    const { patch } = urlStateToConfig(decoded, { findModel: findModelById, findGPU: findGPUById })
    store.getState().restoreConfig(patch)
    return store.getState()
  }
  const base = {
    q: 'fp16',
    sl: 4096,
    bs: 1,
    kvq: 'fp16',
    ng: 1,
    ss: 'tensor-parallel',
    modelId: L70.id,
    gpuId: H100.id,
  }

  it('an old link restores the default mix and no target, without a notice', async () => {
    const s = await restore(base)
    expect(s.contextMix).toEqual([{ tokens: 4096, share: 1 }])
    expect(s.ttftTargetSeconds).toBeNull()
    expect(s.pendingNotice).toBeNull()
  })

  it('shares at 120% restore at 100% with "Shared link adjusted"', async () => {
    const s = await restore({
      ...base,
      cm: [
        [4096, 70],
        [32768, 50],
      ],
    })
    expect(s.contextMix.reduce((a, k) => a + k.share, 0)).toBeCloseTo(1)
    expect(s.sequenceLength).toBe(32768)
    expect(s.pendingNotice?.title).toBe('Shared link adjusted')
  })

  it('tt: 0 restores 0.1 and is noticed', async () => {
    const s = await restore({ ...base, tt: 0 })
    expect(s.ttftTargetSeconds).toBe(0.1)
    expect(s.pendingNotice?.title).toBe('Shared link adjusted')
  })

  it('setContextMix sets sequenceLength to the largest class, no toast', async () => {
    const store = await freshStore()
    store.getState().setContextMix([
      { tokens: 4096, share: 0.7 },
      { tokens: 262144, share: 0.3 },
    ])
    expect(store.getState().sequenceLength).toBe(262144)
    expect(store.getState().pendingNotice).toBeNull()
  })

  it('the sequence slider edits the one class, or the largest of several', async () => {
    const store = await freshStore()
    store.getState().setSequenceLength(8192)
    expect(store.getState().contextMix).toEqual([{ tokens: 8192, share: 1 }])
    store.getState().setContextMix([
      { tokens: 4096, share: 0.7 },
      { tokens: 32768, share: 0.3 },
    ])
    store.getState().setSequenceLength(65536)
    expect(store.getState().contextMix.map((k) => k.tokens)).toEqual([4096, 65536])
    expect(store.getState().sequenceLength).toBe(65536)
  })

  it('training leaves the mix and target as they are', async () => {
    const store = await freshStore()
    store.getState().setMode('training')
    store.getState().setTTFTTarget(0)
    expect(store.getState().ttftTargetSeconds).toBe(0)
  })

  it('Training edits sequenceLength even with a several-class mix', async () => {
    const store = await freshStore()
    store.getState().setContextMix([
      { tokens: 4096, share: 0.7 },
      { tokens: 32768, share: 0.3 },
    ])
    store.getState().setMode('training')
    store.getState().setSequenceLength(8192)
    expect(store.getState().sequenceLength).toBe(8192)
  })

  it('a restored one-class cm that disagrees with sl is noticed', async () => {
    const s = await restore({ ...base, cm: [[8192, 100]] })
    expect(s.pendingNotice?.title).toBe('Shared link adjusted')
  })

  it('a thirds mix round-trips with no notice', async () => {
    const store = await freshStore()
    const { serializeToURL, deserializeFromURL, urlStateToConfig } = await import(
      '@store/urlSerializer'
    )
    const { findGPUById, findModelById } = await import('@store/uiStore')
    store.getState().setSelectedModel(L70)
    store.getState().setSelectedGPU(H100)
    store.getState().setContextMix([
      { tokens: 4096, share: 1 / 3 },
      { tokens: 32768, share: 1 / 3 },
      { tokens: 131072, share: 1 / 3 },
    ])
    const url = serializeToURL(store.getState())
    const d = deserializeFromURL(url)
    if (!d) throw new Error('parse')
    const fresh = await freshStore()
    fresh
      .getState()
      .restoreConfig(urlStateToConfig(d, { findModel: findModelById, findGPU: findGPUById }).patch)
    expect(fresh.getState().pendingNotice).toBeNull()
    expect(serializeToURL(fresh.getState())).toBe(url)
  })

  it('round-trips URL, store, URL identically', async () => {
    const store = await freshStore()
    const { serializeToURL, deserializeFromURL, urlStateToConfig } = await import(
      '@store/urlSerializer'
    )
    const { findGPUById, findModelById } = await import('@store/uiStore')
    store.getState().setSelectedModel(L70)
    store.getState().setSelectedGPU(H100)
    store.getState().setContextMix([
      { tokens: 4096, share: 0.7 },
      { tokens: 131072, share: 0.3 },
    ])
    store.getState().setTTFTTarget(2.5)
    const url = serializeToURL(store.getState())
    const d = deserializeFromURL(url)
    if (!d) throw new Error('parse')
    const fresh = await freshStore()
    fresh
      .getState()
      .restoreConfig(urlStateToConfig(d, { findModel: findModelById, findGPU: findGPUById }).patch)
    expect(serializeToURL(fresh.getState())).toBe(url)
  })
})
