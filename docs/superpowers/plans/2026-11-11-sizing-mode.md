# Sizing Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a third mode, "Sizing". The user gives a model, a context and the number of simultaneous users. The tool answers "N racks" or "N servers" of the chosen GPU, using vLLM-valid replicas and calibrated expected speed.

**Architecture:**
- One new pure module `src/engines/sizing.ts`. It composes the existing engine functions (`calculateInferenceVRAM`, `calculateMultiNodeVRAM`, `estimatePerformance`, `maxConcurrentSessions`, `validTPDegrees`, `fitsInference`) over a small grid of candidate replicas and picks the fewest units.
- `ConfigMode` gains `'sizing'`.
- The calculation worker gets a `CALCULATE_SIZING` message.
- A Sizing results card replaces the per-replica results in that mode.

There are three gates first: the FP8 MLA KV layout, load-band decode efficiency, and valid TP degrees in Recommendations.

**Tech Stack:** React 19, TypeScript strict, Zustand, decimal.js, Vitest, Biome.

**Spec:** `docs/superpowers/specs/2026-11-11-rack-sizing-mode-design.md`. It was approved 2026-11-11. Read it; it is the authority, and this plan argues from it.

## Global Constraints

- **Branch:** `feat/sizing-mode`, in the main checkout. Never use a worktree. It is stacked on `feat/calibration-sparse-attention`.
- **Engine purity:** engines stay pure, with no React or DOM. The solver adds no second copy of the memory or speed logic (spec anti-goal). Use KISS, DRY and functional style: small pure functions, no classes, no mutation of inputs.
- **Speed:** unit counts use **expected** speed, never the ceiling. The ceiling is shown beside every speed figure.
- **Fit:** inference and sizing fit with `fitsInference` (90% of `vram_gb`). Training is unchanged.
- **TP degrees:** only those vLLM accepts, from `validTPDegrees(model, max)` in `src/engines/config-rules.ts`. Never write a new list.
- **Rack GPU:** a GPU with `max_gpus_per_node > 8`. Today those are `nvidia-gb200-nvl72`, `nvidia-gb300-nvl72` and `nvidia-vera-rubin-nvl72`. Rack GPUs are selectable in Sizing only.
- **IDs:** database ids are never renamed. `src/data/gpus.json` is never hand-edited. `models.json` is sorted by codepoint.
- **Re-pins:** every changed pin carries a dated reason comment (`// 2026-11-11, sizing T<n>: was X, now Y because …`).
- **Fixtures:** test fixtures are built from the real engine or factory, not hand literals. Test files are not typechecked.
- **Biome:** 2-space indent, single quotes, no semicolons, 100 columns, no `!`.
- **Commits:** write each message to a file and use `git commit -F`. End every message with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_0148C11Vkt7xNbSVsEoYPcmB`
- **Staging:** stage explicit paths only. Never stage `.serena/project.yml`, which is the user's own edit.
- **Verify** with `npx vitest run`, `./node_modules/.bin/tsc -b --force`, `npx @biomejs/biome check .` and `npm run build`. Use these direct binaries, not rtk, which can fabricate output.

## Review Focus

1. **A model too big for any candidate**, for example Kimi K2-class on a single-GPU workstation part. Expect a "does not fit" result naming the closest candidate, never a crash or 0 units. Task 4 pins it.
2. **Users = 1, or users below the sessions of one replica.** Expect exactly 1 replica and 1 unit, and per-user speed = the replica's speed at batch 1. Task 4 pins it.
3. **72 not divisible by the replica size** (for example TP 16 or 32 in a rack). Expect the idle GPUs shown and the count right: 72 / 16 gives 4 replicas and 8 idle. Task 4 pins it.
4. **Switching Sizing → Inference with a rack GPU selected.** Expect the GPU cleared with one toast; nothing crashes on a null GPU. Task 5 pins it.
5. **A sparse or MLA model at FP8 KV in Sizing.** Expect 656 B charged and sessions consistent with Inference mode at the same replica. Tasks 1 and 4 pin it.

---

### Task 1: Gate 1, the FP8 MLA KV layout

**Files:** `src/engines/kv-cache.ts`, `src/engines/kv-cache.test.ts`, and the pins that move in `src/engines/*.test.ts` and `src/components/**/*.test.tsx`.

**Interfaces:**
- Produces: `MLA_FP8_KV_BYTES_PER_ELEMENT_FACTOR`, or an equivalent single source, so that at FP8 KV on a `use_mla` model each layer charges 656 B per token instead of 576. Every consumer of KV bytes (memory, decode reads, sessions) goes through it.

- [ ] **Step 1: Confirm the layout in vLLM source.**
  - Use Context7 (`/vllm-project/vllm`) or GitHub. Look for `fp8_ds_mla`: the FlashMLA sparse / DeepSeek MLA FP8 KV layout, which per token is 512 FP8 NoPE + 4×4 B scales + 64 BF16 RoPE = 656 B.
  - Record the file path and line in a code comment and in the task report.
  - Note which models vLLM applies it to: all `use_mla` models at `kv_cache_dtype=fp8`, or only sparse/DeepSeek ones.
  - **Not confirmable:** apply 656 to all `use_mla` models at FP8 KV (the safe direction), and say so in the comment.
- [ ] **Step 2: Failing test.** In `kv-cache.test.ts`, for Mistral Large 4 (from `models.json`) at FP8 KV, a 1k context and 1 session, KV bytes equal `kv_cache_elements_per_token / 576 × 656 × 1024`. The element count already covers every layer, so 576 → 656 per layer scales it. Also check that FP16 KV is unchanged.
- [ ] **Step 3:** Implement it in the one place KV bytes per element are resolved. Do not touch `models.json` values; they stay vLLM element counts.
- [ ] **Step 4:** Run `npx vitest run`. Re-pin each moved test with a dated reason. Only MLA models at FP8 KV may move.
- [ ] **Step 5:** Commit `fix(engine): FP8 MLA KV charges vLLM's fp8_ds_mla 656 B layout`.

### Task 2: Gate 3, decode efficiency by load band

**Files:** `src/engines/constants.ts`, `src/engines/performance.ts`, `src/engines/types.ts` (JSDoc), `src/engines/performance.test.ts`, plus moved pins.

**Interfaces:**
- Produces: `decodeEfficiency(batchSize: number): Decimal` in `constants.ts`. It returns 0.47 for batch ≤ 32, 0.425 for 33–128 and 0.25 above 128.
- Produces: `DECODE_EFFICIENCY_BANDS`, an exported readonly array `{ maxBatch: number, efficiency: number, n: number, provisional?: true }`. The UI info tip and the docs read it.
- Consumes: the batch that `estimatePerformance` already receives (`params.batchSize`).

- [ ] **Step 1: Failing tests.**
  - `decodeEfficiency(1)` = 0.47, `(32)` = 0.47, `(33)` = 0.425, `(128)` = 0.425 and `(129)` = 0.25.
  - `estimatePerformance` at batch 64 gives `expected.tokensPerSecond` = ceiling × 0.425.
  - At batch 8, × 0.47.
- [ ] **Step 2:** Replace the flat `DECODE_EFFICIENCY` use in `performance.ts` with `decodeEfficiency(batchSize)`. Delete `DECODE_EFFICIENCY` and update its importers; grep for them.
  - Source comment for the bands: medians per concurrency band of the accepted points in `docs/calibration/2026-11-11-vllm-efficiency-points.md`. Band 2–8: 0.481 (n=203). Band 9–32: 0.472 (n=188). Band 33–128: 0.425 (n=103). Above 128: 0.252 (n=1), provisional.
  - Bands ≤ 32 keep the overall median 0.47, the user's decision of 2026-11-11.
  - `PREFILL_EFFICIENCY` 0.469 is unchanged.
- [ ] **Step 3:** In the existing Verdict InfoTip, say that efficiency depends on load. Show the bands, and mark the value above 128 "provisional (1 point)".
- [ ] **Step 4:** Run the full suite. Re-pin expected-speed pins at batch > 32 with dated reasons. Pins at batch ≤ 32 must not move.
- [ ] **Step 5:** Commit `feat(engine): decode efficiency by load band (0.47 / 0.425 / 0.25 provisional)`.

### Task 3: Gate 2, Recommendations advise only valid TP degrees

**Files:** `src/components/outputs/Recommendations.tsx` and `Recommendations.test.tsx`.

**Interfaces:** consumes `validTPDegrees(model, maxGPUsFor(gpu))`.

- [ ] **Step 1: Failing test.** DeepSeek R1 Distill 70B (64 heads) at fp16 with a 32k context on an H100 SXM. The advised "Use Nx" must satisfy `isValidTPDegree(model, N)`, so N = 4, not 3. Build it through the real engine, as the existing I1 tests do.
- [ ] **Step 2:** Round the memory-derived GPU count up to the next value in `validTPDegrees(model, max)`. If none is ≤ max, keep the existing "exceeds one server" path.
- [ ] **Step 3:** Run the suite, then commit `fix(ui): Recommendations advise only TP degrees vLLM accepts`.

### Task 4: The pure solver `src/engines/sizing.ts`

**Files:**
- Create: `src/engines/sizing.ts` and `src/engines/sizing.test.ts`.
- Modify: `src/engines/index.ts` (export) and `src/engines/types.ts` (result types).

**Interfaces (produces; Tasks 5-7 consume these exact names):**
```ts
export type SizingUnit = 'server' | 'rack'
export interface SizingInput {
  model: Model
  gpu: GPU
  quantization: QuantizationFormat
  kvQuantization: KVCachePrecision
  sequenceLength: number
  users: number
  fabric: FabricSpec
}
export interface ReplicaCandidate {
  tp: number
  nodes: number
  gpusPerReplica: number // tp * nodes
  sessionsPerReplica: number // memory-bound, >= 1 when it fits
  perUserTokensPerSecond: Decimal
  perUserCeilingTokensPerSecond: Decimal
  timeToFirstToken: Decimal
  ceilingTimeToFirstToken: Decimal
  totalPerGPUGB: number
}
export interface SizingResult {
  fits: boolean
  unit: SizingUnit
  units: number // racks or servers
  replicas: number
  replicasPerUnit: number // 0 for a multi-server replica
  idleGPUsPerUnit: number // rack: 72 - replicasPerUnit*tp; server: max_gpus_per_node - replicasPerUnit*tp
  chosen: ReplicaCandidate | null
  closest: { candidate: ReplicaCandidate; lackingGB: number } | null // set when !fits
}
export function isRackGPU(gpu: GPU): boolean // max_gpus_per_node > 8
export function replicaCandidates(model: Model, gpu: GPU): Array<{ tp: number; nodes: number }>
export function evaluateReplica(
  input: SizingInput,
  tp: number,
  nodes: number,
): ReplicaCandidate & { fits: boolean; lackingGB: number }
export function sizeDeployment(input: SizingInput): SizingResult
```

- [ ] **Step 1: Failing tests** in `sizing.test.ts`. Every model and GPU comes from `models.json` and `gpus.json`, with fabric from `resolveFabricSpec('ethernet-800g', null)`.
  - **`replicaCandidates`:**
    - Rack (GB300 NVL72): every `validTPDegrees(model, 72)` with nodes 1.
    - Server (H100 SXM): every `validTPDegrees(model, 8)` with nodes 1, plus `{tp: 8, nodes: 2..8}`. The latter only when 8 is valid for the model.
    - A refused degree such as 3 for 64 heads is never present.
  - **Small model on HGX:** Llama 3.1 8B fp16 at 8k on H100 SXM, 100 users. Expect tp 1, `replicasPerUnit` 8, `unit` `'server'`, and `units` = ceil(replicas / 8).
  - **Model larger than one HGX:** a model whose fp16 weights exceed 8 × 80 × 0.9 GB. Pick one from `models.json` (for example the largest dense or MoE above 600 GB at fp16), on H100 SXM. Expect `chosen.nodes` ≥ 2, `replicasPerUnit` 0, and `units` = replicas × nodes.
  - **Rack split:**
    - A 64-head model where TP 16 is chosen gives `idleGPUsPerUnit` 8 with `replicasPerUnit` 4.
    - Where TP 8 is chosen it gives 9 and 0.
    - Assert on `replicasPerUnit * chosen.tp + idleGPUsPerUnit === 72`.
  - **Doubling users** (1000 to 2000) doubles `replicas` within ±1.
  - **Users = 1:** `replicas` 1, `units` 1, and `perUserTokensPerSecond` equals the expected tok/s at batch 1 of the same replica.
  - **Does not fit:** a model too big for any candidate of a small GPU (for example GLM 5.2 fp16 on an RTX-class 24 GB part). Expect `fits` false, `closest` set, `lackingGB` > 0 and `units` 0.
  - **Expected vs ceiling:** `chosen.perUserTokensPerSecond` < `chosen.perUserCeilingTokensPerSecond`.
  - **Tie-break:** among candidates with equal `units`, the chosen one has the fewest `replicas × gpusPerReplica`, then the smallest `gpusPerReplica`.
- [ ] **Step 2:** Run `npx vitest run src/engines/sizing.test.ts` and confirm it fails because the module is missing.
- [ ] **Step 3: Implement.** `evaluateReplica` is one straight composition with no branches beyond nodes > 1:
  1. `calculateInferenceVRAM({ model, quantization, sequenceLength, batchSize: 1, kvQuantization, concurrentUsers: 1 })`.
  2. Then `calculateMultiNodeVRAM({ singleGPU, model, gpuVramGB: gpu.vram_gb, gpusPerNode: tp, numNodes: nodes, intraNodeStrategy: 'tensor-parallel', gpu, fabric, batchSize: 1, sequenceLength, quantization })`, when tp × nodes > 1.
  3. `sessionsPerReplica` = `maxConcurrentSessions({ totalPerGPUGB, kvPerGPUGB, concurrentUsers: 1, gpuVramGB: gpu.vram_gb })`, capped at `users`.
  4. `fits` = `fitsInference(totalPerGPUGB, gpu.vram_gb)` && sessions ≥ 1.
  5. Speed: `estimatePerformance` at `batchSize = sessionsPerReplica` (the replica's real batch), using the multi-GPU result recomputed at that batch. Per-user speed is `expected.tokensPerSecond / sessions` and the ceiling likewise.
  6. `lackingGB` = `max(0, totalPerGPUGB - gpu.vram_gb * 0.9)` at one session.

  `sizeDeployment` then:
  - maps the candidates through `evaluateReplica` and keeps those that fit;
  - computes `replicas = ceil(users / sessions)`;
  - computes units: for a rack, `replicasPerUnit = floor(72 / tp)` and `units = ceil(replicas / replicasPerUnit)`; for a server, with `nodes` = 1, `replicasPerUnit = floor(max_gpus_per_node / tp)` and `units = ceil(replicas / replicasPerUnit)`, else `units = replicas × nodes`;
  - sorts by `(units, replicas × gpusPerReplica, gpusPerReplica)`.

  Use `72` as `gpu.max_gpus_per_node` for racks, not a literal.
- [ ] **Step 4:** Run the sizing tests and see them pass. Then run the full suite.
- [ ] **Step 5:** Commit `feat(engine): pure sizing solver — fewest servers or racks for N simultaneous users`.

### Task 4b: KV storage tier in the solver (spec amendment)

**Files:** `src/engines/sizing.ts`, `src/engines/sizing.test.ts`, `src/engines/types.ts`.

**Interfaces:**
- `SizingInput` gains `kvTier: KVTierSettings`.
- `ReplicaCandidate` gains `hotSessions: number`, `decodeBatch: number`, `tierUsed: KVTierType | null`, `tierOverloaded: boolean` and `resumeSeconds: number | null`.
- `sessionsPerReplica` becomes the open sessions.
- Consumes `kvTierSummary` and `clampKVTier` from `src/engines/kv-tier.ts`, unchanged.

- [ ] **Step 1: Failing tests.** All use real models and GPUs.
  - With tier `none`, results are identical to today. Pin the Task 4 literal cases unchanged.
  - With `host-grace` on GB300 NVL72, activeShare 0.25 and default burst, `sessionsPerReplica` equals `kvTierSummary(...).sessionsHeld`, which is greater than `hotSessions`. `decodeBatch = min(hot, ceil(open × 0.25))`, and `units` is at most the `none` case.
  - With an overloaded tier (for example `network` with a tiny `customGBps`), the result falls back to the HBM count with `tierOverloaded: true`.
  - Per-user speed is computed at `decodeBatch`, not at open sessions.
- [ ] **Step 2: Implement.**
  - In `evaluateReplica`, after the HBM session count, call `kvTierSummary` with the replica's per-GPU and total KV.
  - Set `multi = { strategy: 'tensor-parallel', gpusPerNode: tp, numNodes: nodes, numGPUs: tp × nodes }`.
  - Pass `recomputeSeconds` = the ceiling prefill at one session, and `gpuId: gpu.id`.
- [ ] **Step 3:** Run the suite, then commit `feat(engine): sizing models the KV storage tier (open vs decoding sessions)`.

### Task 5: The `'sizing'` mode in config rules, store and URL

**Files:**
- `src/engines/config-rules.ts` and its test.
- `src/store/uiStore.ts`.
- `src/store/urlSerializer.ts` and its test.
- `src/components/inputs/ModeToggle.tsx` and `src/components/inputs/GPUSelector.tsx`.
- `src/hooks/useAllowedOptions.ts`, if it filters GPUs.

**Interfaces:**
- `ConfigMode = 'inference' | 'training' | 'sizing'`.
- `allowedOptions(...)` gains `gpuIds: (gpu: GPU) => boolean`, or an equivalent `gpuAllowed(mode, gpu)` predicate exported from `config-rules.ts`. Rack GPUs are allowed only in `'sizing'`.
- URL `m` accepts `'sizing'`.

- [ ] **Step 1: Failing tests.**
  - `config-rules.test.ts`:
    - Every rule declares an explicit mode list.
    - The model, GPU, quantization and KV-precision rules run in `'sizing'`.
    - The rules for the fields Sizing hides do not. Those fields are numGPUs, numNodes, shardingStrategy, batchSize, offload*, kvTier and the interconnect override.
  - A new rule restores a config with a rack GPU in `'inference'` or `'training'` as `mode: 'sizing'`, with a correction of subject `'mode'`.
  - `normalizeConfig` converges within `MAX_NORMALIZE_PASSES`.
  - Store: `setMode('inference')` while a rack GPU is selected clears `selectedGPU` and sets a `pendingNotice` that contains "is sized as a rack; pick a server GPU".
  - URL: a link `m: 'sizing'` round-trips.
  - A link with `m: 'inference'` and `gpuId: 'nvidia-gb300-nvl72'` restores in sizing with "Shared link adjusted".
  - `ModeToggle` renders three choices (Inference / Sizing / Training) and sets the mode.
  - `GPUSelector` in inference does not list the NVL72 ids; in sizing it lists all GPUs.
- [ ] **Step 2: Implement.**
  - **ModeToggle:** a three-option segmented control replaces the Switch. Use Headless UI `RadioGroup`, already a dependency; check the Context7 docs for `@headlessui/react` v2 `RadioGroup` before writing it.
  - **Labels:** Inference, Sizing, Training. The description lines are: "Size servers or racks for N simultaneous users" for Sizing, and the existing lines for the other two.
- [ ] **Step 3:** Run the full suite. Re-pin the tests that assumed two modes, with dated reasons.
- [ ] **Step 4:** Commit `feat(config): sizing mode; rack GPUs selectable in Sizing only`.

### Task 6: Worker message, hook and the Sizing results card

**Files:**
- `src/workers/calculation.worker.ts` and its test.
- Create `src/hooks/useSizingCalculation.ts`.
- Create `src/components/outputs/SizingResults.tsx` and its test.
- `src/components/layout/ResultsPanel.tsx` and `InputPanel.tsx`.
- `src/components/guide/GuidePage.tsx`.

**Interfaces:**
- Consumes `sizeDeployment` and `SizingResult` (Task 4), plus the mode (Task 5).
- Worker: `{ type: 'CALCULATE_SIZING', payload }` returns `{ type: 'SIZING_RESULT', result: SerializedSizingResult }`. Decimals are serialized as strings and reconstructed in the hook, following `reconstructPerformanceEstimate`.
- `useSizingCalculation(): { result: SizingResult | null; isCalculating: boolean; error: string | null }`.

- [ ] **Step 1: Failing tests.**
  - The worker round-trips a sizing result. This is a real `sizeDeployment` call, not a mock.
  - `SizingResults`, rendered from a real `sizeDeployment` result for GLM 5.2 fp8 at 32k on GB300 NVL72 with 2000 users, shows:
    - the headline `${units} racks NVIDIA GB300 NVL72`, singular "rack" when there is 1;
    - the replica line `replica = ${tp} GPUs (tensor parallel)`, with ` × ${nodes} servers` when nodes > 1;
    - the replicas, replicas per rack, idle GPUs and sessions per replica;
    - the per-user expected tok/s with "Theoretical ceiling: …" and the expected TTFT with the ceiling.
  - Server GPUs read `N servers <GPU>`.
  - When it does not fit: "Does not fit: closest is TP {tp} × {nodes}, lacking {X} GB per GPU".
  - InputPanel in sizing hides GPUs per replica, servers, sharding strategy, batch size, offloading and the KV tier. It shows the model, quantization, KV precision, GPU, context, users and fabric. Fabric shows for server GPUs only.
- [ ] **Step 2:** Implement it. ResultsPanel renders `SizingResults` in sizing mode and the existing results otherwise. Follow the existing results card styles and use `InfoTip` for the efficiency note.
- [ ] **Step 3:** Add a "Sizing" section to the in-app guide that explains how the solver works, in 4-6 short sentences.
- [ ] **Step 4:** Run the suite and a browser check with the `npx vite` dev server. Look at the sizing card at 1800 px and 390 px, in the dark and light themes.
- [ ] **Step 5:** Commit `feat(ui): Sizing results — N racks or servers, replica split, expected speed with ceiling`.

### Task 7: Comparison and exports for sizing

**Files:**
- `src/store/comparisonStore.ts`.
- `src/components/comparison/ComparisonColumn.tsx`.
- `src/components/layout/ResultsPanel.tsx` (Save to Compare).
- `src/utils/exportPptx.ts`.
- `src/hooks/useResultExports.ts` and their tests.

**Interfaces:** `ConfigSnapshot.results` gains optional `sizing?: { unit: SizingUnit; units: number; replicas: number; tp: number; nodes: number; sessionsPerReplica: number }`.

- [ ] **Step 1: Failing tests.**
  - Saving in sizing mode stores `sizing` with the result's values.
  - `ComparisonColumn` shows "Units" (for example "3 racks") and "Replica" (`TP 8 × 1`) rows. They are blank ("—") for inference snapshots.
  - The PPTX deck in sizing mode has a sizing summary slide (headline, replica, replicas, sessions, per-user speed with the ceiling) and no per-replica memory chart slides.
  - The PDF export captures the Sizing card. Assert that the export path receives the sizing DOM node, as the existing PDF test does.
- [ ] **Step 2:** Implement, run the suite, and commit `feat(export): sizing in comparison and exports`.

### Task 8: Docs

**Files:**
- Create `docs/adr/0010-sizing-mode.md` and add its row to `docs/adr/README.md`.
- Modify `CHANGELOG.md` (`## [Unreleased]`), `README.md`, `CLAUDE.md` (Key Patterns) and `ARCHITECTURE.md`.
- Modify `docs/calibration/2026-11-11-vllm-efficiency-points.md`: add the band table.
- Modify `docs/adr/0009-…md`: one line pointing to 0010 for the bands.

- [ ] **Step 1:** ADR 0010 covers:
  - the user's decisions: mode C, rack GPUs in Sizing only, users = simultaneous sessions, no speed floor, efficiency by band;
  - the solver (candidate grid, tie-break, units rule, idle GPUs);
  - the three gates and their outcomes (cite the vLLM file for fp8_ds_mla);
  - the non-goals.
- [ ] **Step 2:** CHANGELOG, under Breaking:
  - expected decode speed falls at batch > 32 (0.425) and > 128 (0.25, provisional);
  - FP8 MLA KV is 656 B, so fewer sessions;
  - rack GPUs leave Inference and Training, and shared links with them restore in Sizing.

  Under Added: Sizing mode. Under Fixed: Recommendations TP degrees.
- [ ] **Step 3:** CLAUDE.md Key Patterns gets one terse bullet each:
  - Sizing (`sizeDeployment` composes the engine; rack GPUs are Sizing-only);
  - decode efficiency by band.

  Fix any existing bullet these contradict.
- [ ] **Step 4:** Run the full verification set and put the counts in the commit body. Commit `docs: ADR 0010 sizing mode, CHANGELOG, README, CLAUDE.md`.
