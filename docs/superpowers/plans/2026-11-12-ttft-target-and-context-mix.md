# TTFT Target and Context Mix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A context mix (1-3 classes summing to 100%) that drives KV memory and decode, a TTFT
under a burst of concurrent prompts, and an optional TTFT target that Inference checks and
Sizing sizes to.

**Architecture:**
- The mix enters the engine at two points:
  - `calculateInferenceVRAM` weights every context-dependent KV term by the mix.
  - `estimatePerformance` weights the decode KV read by the mix.
- Everything downstream (multi-GPU, sessions, Sizing) inherits it.
- A new pure module `ttft.ts` prices each class's prompt at batch 1 with the existing engine and
  builds the burst queue.
- The Sizing solver adds a burst cap. Rules, URL, UI and exports carry the two new inputs.

**Tech Stack:** TypeScript strict, React 19, Zustand, Zod, Vitest, decimal.js, Biome.

**Spec:** `docs/superpowers/specs/2026-11-12-ttft-target-and-context-mix-design.md`

## Global Constraints

- **Default mix:** one class, 100% at `sequenceLength`. With it every existing figure and test
  is unchanged.
- **Mix shape:** 1 to 3 classes. Each share is in (0, 1], and the shares sum to exactly 1 (the
  last class takes the remainder).
- **`sequenceLength` with a mix** is the largest class's tokens.
- **Presets** (editable sizes): Chat 4096, RAG 32768, Long documents 262144.
- **TTFT target:**
  - optional, in seconds, from 0.1 to 3600, empty by default;
  - shown in Inference and Sizing, hidden in Training.
- **Mix in the UI:** shown in Inference and Sizing, hidden in Training. Training keeps
  `sequenceLength`.
- **Headline TTFT** = first served: a lone prompt at the largest class, today's figure.
- **Last served** = the sum over classes of burst x share x the class's expected TTFT at batch 1.
  - It is labelled "worst case: all N prompts at once, first come first served, no prefix cache".
  - It is never labelled "expected".
- **The target applies to last served.**
- **Burst:**
  - Inference: `concurrentUsers`.
  - Sizing: the users per replica.
  - With a KV tier in use: the decode batch, ceil(open x activeShare).
- **Burst cap** = the largest B whose last served meets the target.
- **Sizing users per replica** = min(memory sessions, burst cap). The result names the bound
  that decided (`'memory' | 'ttft'`).
- **Memory label:** "memory sized on the mix average".
- **Guard:** one session at the largest class must fit at 90%.
- **Gates:**
  - `npx vitest run`, `npx tsc -b --force` and `npm run build` must pass.
  - Run `./node_modules/.bin/biome check .` and read its FULL output: the rtk wrapper can
    print a bogus "clean".
- **Ids and URLs:** never rename ids. Old shared links without the new URL keys restore the
  default mix and no target.
- **Commit trailers:**
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_0148C11Vkt7xNbSVsEoYPcmB`.

## Review Focus

1. **Classes beyond the native context.** A class larger than the model's native context (a
   256K preset on a 128K model) warns per class and never clamps. Pinned in Task 5.
2. **Target below one lone prompt.** Inference shows "max users within target: 0" and Sizing
   returns `neverMeetsTarget`, naming the largest class. Pinned in Tasks 3 and 4.
3. **Sliding-window, linear-state and sparse models with a mix.** Sliding KV is weighted per
   class at min(window, tokens). The linear state stays per session, independent of the mix.
   Pinned in Task 2 on Gemma 3 and Kimi K3.
4. **Offloading or KV tier with a mix.** The tier prices resume and traffic on the mix mean KV,
   and offload bytes follow the mix. Pinned in Task 2 (offload) and Task 4 (tier burst).
5. **Users = 1 or burst = 1.** Last served equals first served for one class. Pinned in Task 3.

---

### Task 1: Context mix module

**Files:**
- Create: `src/engines/context-mix.ts`
- Create: `src/engines/context-mix.test.ts`
- Modify: `src/engines/types.ts` (export the types)
- Modify: `src/engines/index.ts` (re-export)

**Interfaces:**
- Produces:
  - `interface ContextClass { tokens: number; share: number }`
  - `type ContextMix = ContextClass[]`
  - `CONTEXT_PRESETS: ReadonlyArray<{ label: 'Chat' | 'RAG' | 'Long documents'; tokens: number }>`
  - `defaultMix(sequenceLength: number): ContextMix`
  - `normalizeMix(mix: ContextMix): ContextMix`: keeps 1 to 3 classes, drops non-positive
    shares, and lets the last class take the remainder so the shares sum to 1. An empty mix
    throws.
  - `largestContext(mix: ContextMix): number`
  - `mixWeighted(mix: ContextMix, f: (tokens: number) => Decimal): Decimal`: the sum of share
    times f(tokens).
  - `isDefaultMix(mix: ContextMix, sequenceLength: number): boolean`

- [ ] **Step 1: Failing tests** in `context-mix.test.ts`:
  - `defaultMix(4096)` equals `[{tokens: 4096, share: 1}]`.
  - `normalizeMix([{tokens:4096,share:0.7},{tokens:32768,share:0.2},{tokens:262144,share:0.5}])`
    gives last share 0.1 and a total of 1 within 1e-12.
  - Four classes are cut to three.
  - `largestContext` of that mix is 262144.
  - `mixWeighted(defaultMix(8192), t => new Decimal(t))` equals 8192.
  - `isDefaultMix(defaultMix(8192), 8192)` is true.
- [ ] **Step 2:** Run `npx vitest run src/engines/context-mix.test.ts`. Expect FAIL: module not
  found.
- [ ] **Step 3:** Implement the functions as pure functions, with no store or React import.
- [ ] **Step 4:** Run the tests. Expect PASS.
- [ ] **Step 5:** Commit `feat(engine): context mix module`.

### Task 2: Memory and decode follow the mix

**Files:**
- Modify: `src/engines/inference.ts`: `calculateInferenceVRAM` gains `contextMix?: ContextMix`.
- Modify: `src/engines/performance.ts`: `PerformanceParams` gains `contextMix?: ContextMix`,
  used for the decode KV read only.
- Modify: `src/engines/offloading.ts` and `src/engines/kv-tier.ts`, only where they re-derive KV
  from `sequenceLength`. They take the mix-weighted KV instead.
- Test: `src/engines/inference.test.ts`, `src/engines/performance.test.ts`.

**Interfaces:**
- Consumes: Task 1, `mixWeighted` and `largestContext`.
- Produces: `calculateInferenceVRAM({ ..., contextMix })`.
  - `kvCache` equals `mixWeighted(mix, t => calculateKVCacheVRAM({ model, sequenceLength: t, batchSize: users, kvPrecision }))`.
  - That covers the full-attention, sliding-window and sparse-indexer terms, which all live in
    `calculateKVCacheVRAM`.
  - `linearState` is unchanged (per session).
  - Activations stay at `sequenceLength` (the largest class).
- Produces: `estimatePerformance({ ..., contextMix })`.
  - The decode bytes per step read the mix-weighted KV per sequence.
  - Prefill and TTFT stay at `sequenceLength`.
- When `contextMix` is absent, or `isDefaultMix`, the existing code path runs unchanged.

- [ ] **Step 1: Failing tests**
  - Llama 3.3 70B fp16, users 10, mix 90% 4096 / 10% 262144: `kvCache` equals
    0.9 x KV(4096) + 0.1 x KV(262144) for 10 users. Both terms come from
    `calculateKVCacheVRAM`, within 1e-9.
  - The same model with `contextMix: defaultMix(8192)`: every field equals the call without
    `contextMix`.
  - Gemma 3 27B, mix 50% 512 / 50% 32768: the sliding term is weighted per class at
    min(window, tokens).
  - Kimi K3, mix 50/50: `linearState` equals the no-mix value.
  - `estimatePerformance` decode tok/s with the 90/10 mix is greater than at 262144 alone and
    lower than at 4096 alone. `timeToFirstToken` equals the value at `sequenceLength` 262144.
  - Offload: with `offloadingEnabled` and the 90/10 mix, the offloaded KV bytes equal the mix
    KV share (not the 262144 KV).
- [ ] **Step 2:** Run them. Expect FAIL.
- [ ] **Step 3:** Implement. Thread `contextMix` through `calculateMultiGPUVRAM` /
  `calculateMultiNodeVRAM` callers only if they recompute KV. They reuse the single-GPU
  breakdown, so most need nothing.
- [ ] **Step 4:** Run `npx vitest run`. Expect all pass; the existing suite is untouched.
- [ ] **Step 5:** Commit `feat(engine): KV memory and decode follow the context mix`.

### Task 3: TTFT under a burst

**Files:**
- Create: `src/engines/ttft.ts`
- Create: `src/engines/ttft.test.ts`
- Modify: `src/engines/concurrency.ts`. Delete `perUserTimeToFirstToken`: its waves model let B
  prompts finish in the time of one, which contradicts compute-bound prefill. Its only caller,
  `PerformanceSection`, moves to `burstTTFT` in Task 6.

**Interfaces:**
- Consumes: Task 1; `estimatePerformance` (Task 2).
- Produces:
  - `interface ClassPrefill { tokens: number; share: number; expectedTTFT: Decimal; ceilingTTFT: Decimal }`
  - `classPrefills(params: Omit<PerformanceParams, 'batchSize' | 'sequenceLength' | 'contextMix'>, mix: ContextMix): ClassPrefill[]`:
    one `estimatePerformance` call per class at `batchSize: 1` and `sequenceLength: tokens`.
  - `interface BurstTTFT { burst: number; firstExpected: Decimal; firstCeiling: Decimal; lastExpected: Decimal; lastCeiling: Decimal }`
  - `burstTTFT(classes: ClassPrefill[], burst: number): BurstTTFT`:
    - first = the largest class's TTFT;
    - last = burst x the sum over classes of share x TTFT;
    - burst is clamped to at least 1.
  - `maxUsersWithinTarget(classes: ClassPrefill[], targetSeconds: number): number`:
    floor(target / the sum of share x expectedTTFT). 0 when one prompt misses.
  - `meetsTarget(b: BurstTTFT, targetSeconds: number | null): boolean | null`: null when there is
    no target.

- [ ] **Step 1: Failing tests**
  - One class, burst 1: last equals first equals `estimatePerformance(... batchSize 1).expected.timeToFirstToken`.
  - One class, burst 100: last equals 100 x first.
  - Mix 90% 4096 / 10% 262144, burst 10: last equals 10 x (0.9 x t4k + 0.1 x t256k).
  - **Multi-server batch 1** (Llama 3.1 405B fp16, H100 SXM 8 GPUs x 2 nodes): each
    `ClassPrefill` equals `estimatePerformance` at `batchSize: 1`, and the hop term is counted
    once per prompt.
  - **M5 Max case:** Qwen3.6 35B A3B nvfp4, `apple-m5-max` (check the id in `gpus.json`), 262144
    tokens, 100 users, target 10 s:
    - `meetsTarget` is false;
    - `maxUsersWithinTarget` is 0.
  - **Calibration check:** for each TTFT row with c > 1 in
    `docs/calibration/2026-11-11-vllm-efficiency-points.md` that the test can rebuild from
    `models.json` and `gpus.json`, compare measured mean TTFT with
    `burstTTFT(..., c).lastExpected` and record the ratio in the test name. Assert
    last >= measured mean (the safe direction). If no row can be rebuilt, write a single
    `it.todo` naming why.
- [ ] **Step 2:** Run them. Expect FAIL.
- [ ] **Step 3:** Implement. Delete `perUserTimeToFirstToken` and its tests. Keep
  `perUserTokensPerSecond`.
- [ ] **Step 4:** Run `npx vitest run src/engines`. Expect PASS. `PerformanceSection` fails
  `tsc` until Task 6, so this task leaves a temporary local `// TODO(task 6)` call to `burstTTFT`
  there that keeps the build green: one `classPrefills` plus one `burstTTFT` call.
- [ ] **Step 5:** Commit `feat(engine): TTFT under a burst of concurrent prompts`.

### Task 4: Sizing solver with the target

**Files:**
- Modify: `src/engines/types.ts`:
  - `SizingInput` gains `contextMix: ContextMix` and `ttftTargetSeconds: number | null`.
  - `ReplicaCandidate` gains `burstCap: number | null`, `ttft: BurstTTFT`,
    `classes: ClassPrefill[]` and `boundBy: 'memory' | 'ttft'`.
  - `SizingResult` gains `neverMeetsTarget: { largestClassTokens: number } | null`.
- Modify: `src/engines/sizing.ts`: `evaluateReplica`, `pack` and `sizeDeployment`.
- Test: `src/engines/sizing.test.ts`.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces:
  - The solver formulation from the spec:
    - users per replica = min(sessions held or memory sessions, burst cap);
    - burst = users per replica, or the decode batch when a tier is used;
    - replicas = ceil(users / users per replica).
  - Memory guard: a candidate also needs one session at `largestContext(mix)` to fit at 90%
    (an extra `perGPUBreakdown` at that context, 1 session). Otherwise it is not `fits`.
  - A candidate whose burst cap is 0 is invalid. If every candidate is invalid,
    `neverMeetsTarget` is set and `fits` stays a memory statement.
  - The server branch is not extended. It passes `ttftTargetSeconds` through and computes no cap,
    with a comment `// server branch unreachable from the UI (ADR 0010 amendment)`.

- [ ] **Step 1: Failing tests** (rack GPU `nvidia-gb300-nvl72`):
  - **No target:** every existing pin is unchanged. Run the old tests untouched.
  - **Target:** Llama 3.3 70B fp8, 4096 tokens, 2000 users, a target set to 2 x the lone TTFT.
    `boundBy` is `'ttft'`, users per replica is at most 2, and units are at least the no-target
    count.
  - **Generous target** (3600 s): result identical to no target.
  - **Target below a lone prompt** at the largest replica: `neverMeetsTarget.largestClassTokens`
    equals 262144 for the 90/10 mix.
  - **Tier:** host-grace, activeShare 0.1, target set. The burst equals `decodeBatch`, not the
    open sessions.
- [ ] **Step 2:** Run them. Expect FAIL.
- [ ] **Step 3:** Implement. Compute the class prefills once per candidate, on its multi-GPU
  layout at batch 1.
- [ ] **Step 4:** Run `npx vitest run src/engines`. Expect PASS.
- [ ] **Step 5:** Commit `feat(sizing): size to a TTFT target with the context mix`.

### Task 5: Store, rules, URL and worker

**Files:**
- Modify: `src/engines/config-rules.ts`: `RuleConfig` gains `contextMix: ContextMix` and
  `ttftTargetSeconds: number | null`. Default `defaultMix(4096)` and `null`.
- Modify: `src/store/uiStore.ts`: setters `setContextMix` and `setTTFTTarget`, through
  `commit()`.
- Modify: `src/store/urlSerializer.ts`:
  - `cm`, optional: an array of `[tokens, sharePercent]`, written only when the mix is not
    default;
  - `tt`, optional: the target in seconds.
- Modify: `src/workers/calculation.worker.ts`, `src/hooks/useInferenceCalculation.ts` and
  `src/hooks/useSizingCalculation.ts`: carry both values.
- Modify: `src/store/comparisonStore.ts`: the snapshot carries the mix and target.
- Test: `config-rules.test.ts`, `urlSerializer.test.ts`, `uiStore.test.ts`.

**Interfaces:**
- **New rules** in `RULES`, with explicit mode lists:
  - **R16 mix** (inference, sizing): normalizes the mix (`normalizeMix`) and sets
    `sequenceLength = largestContext(mix)`. Moving the sequence slider with a one-class mix
    rewrites that class.
  - **R17 target** (inference, sizing): clamps to [0.1, 3600], or null.
  - Each correction yields one "Adjusted for …" toast. On restore it yields "Shared link
    adjusted".
- **Native context:** `softWarnings` gets a per-class warning when `tokens > model.context_length`,
  naming the class. It never clamps.

- [ ] **Step 1: Failing tests**
  - A link with no `cm` or `tt` restores `defaultMix(sl)` and a null target.
  - A link with shares summing to 120% is corrected to 100% and noticed.
  - A link with `tt: 0` restores 0.1 and is noticed.
  - Setting the mix to [4096 70%, 262144 30%] sets `sequenceLength` to 262144.
  - Training mode leaves the mix and target inert (not corrected).
  - The 256K class on a 128K model gives a soft warning naming 262144.
  - Round-trip: URL, then store, then URL is identical.
- [ ] **Step 2:** Run them. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `npx vitest run`. Expect PASS.
- [ ] **Step 5:** Commit `feat(config): context mix and TTFT target in store, rules and links`.

### Task 6: Inference UI

**Files:**
- Create: `src/components/inputs/ContextMixEditor.tsx` and its test.
- Create: `src/components/inputs/TTFTTargetInput.tsx` and its test.
- Modify: `src/components/layout/InputPanel.tsx`: the mix editor sits under Sequence Length, the
  target under Concurrent Users. Both are hidden in Training.
- Modify: `src/components/outputs/PerformanceSection.tsx`: remove the Task 3 TODO.
- Modify: `src/components/outputs/VerdictBlock.tsx` and/or `FitIndicator.tsx`: the target
  verdict line and the memory label.
- Modify: `src/components/outputs/Recommendations.tsx`.

**Interfaces:**
- Consumes: Tasks 3 and 5.
- **Mix editor:**
  - It shows the classes as rows (preset chips plus a tokens field plus a share %).
  - It can add a class (up to 3) and remove a class (down to 1).
  - The last share is read-only and shows the remainder.
  - It shows "memory sized on the mix average" when the mix is not the default.
- **TTFT line** (exact copy):
  - "TTFT {first} first, {last} last ({N} users at once)";
  - under it, "worst case: all N prompts at once, first come first served, no prefix cache";
  - the ceiling beside each;
  - each class's solo time when there are several classes;
  - with a target: "Target {t}: met" or "Target {t}: missed by {x}";
  - "Max users within target: {n}".
- **Memory guard in Inference:** the verdict is "does not fit" when one session at the largest
  class does not fit at 90%, naming that class. "Max sessions at N tokens" reads "Max sessions
  (mix: 90% 4K, 10% 256K)" when the mix is not the default.
- **Recommendation when missed:** "{k} machines like this one (one replica each)",
  k = ceil(users / max(1, maxUsersWithinTarget)). When the cap is 0: "No number of these
  machines meets {t} at {largest} tokens".

- [ ] **Step 1: Failing component tests**
  - The editor's last share equals the remainder.
  - A fourth class is impossible.
  - The default mix shows no memory label.
  - The TTFT line text matches for 1 class and 100 users.
  - The missed-target recommendation shows k.
  - The M5 Max case shows "Max users within target: 0".
  - Mixed context where the largest class alone does not fit gives "does not fit" naming it.
- [ ] **Step 2:** Run them. Expect FAIL.
- [ ] **Step 3:** Implement with existing input styling and dark mode, using `allowedOptions`
  values.
- [ ] **Step 4:** Run `npx vitest run` and `npx tsc -b --force`. Expect PASS.
- [ ] **Step 5:** Commit `feat(ui): context mix editor, TTFT target and burst TTFT in Inference`.

### Task 7: Sizing results and exports

**Files:**
- Modify: `src/components/outputs/SizingResults.tsx`.
- Modify: `src/utils/exportPdf.ts` (`assumptionRows`).
- Modify: `src/utils/exportPptx.ts`.
- Modify: `src/components/comparison/ComparisonColumn.tsx`.
- Tests alongside.

**Interfaces:**
- Consumes: Task 4's result fields and Task 5's snapshot.
- **Sizing card:**
  - It shows "Sized by {memory | TTFT target}", the TTFT line from Task 6 (one shared
    formatter, DRY) and the memory label.
  - `neverMeetsTarget` renders "No rack replica meets {t} at {tokens} tokens".
- **Exports:**
  - The PDF assumptions add "Context mix" (e.g. "90% 4K, 10% 256K") and "TTFT target".
  - The PPTX adds the same rows plus the TTFT first and last.
  - Comparison adds rows for the mix, the target, the TTFT last and "Sized by".

- [ ] **Step 1: Failing tests**
  - The Sizing card shows "Sized by TTFT target" for the Task 4 target case.
  - `assumptionRows` includes the mix and target rows only when set.
  - The PPTX params include both.
- [ ] **Step 2:** Run them. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `npx vitest run`, `npx tsc -b --force` and `npm run build`. Expect PASS.
- [ ] **Step 5:** Commit `feat(sizing-ui): TTFT target, context mix in results and exports`.

### Task 8: Docs

**Files:**
- Create: `docs/adr/0012-ttft-target-and-context-mix.md`, with a row in `docs/adr/README.md`.
- Modify: `CHANGELOG.md` (Unreleased: Added; Changed: the per-user TTFT is now the burst queue),
  `README.md` ("How to read the figures": the mix and the worst-case TTFT), `GuidePage.tsx`,
  and `CLAUDE.md` (one Key Patterns bullet).

**Interfaces:**
- **ADR 0012** records:
  - the user's decisions;
  - why the burst is a worst case and not "expected";
  - the 0.469 closed-loop correction (no double count);
  - that mix-mean memory deliberately overrides "never too few units" (user, 2026-11-12);
  - the removal of `perUserTimeToFirstToken`.

- [ ] **Step 1:** Write the docs. Every count and figure must match the code.
- [ ] **Step 2:** Run `./node_modules/.bin/biome check .` (read the full output) and
  `npx vitest run`.
- [ ] **Step 3:** Commit `docs: ADR 0012 TTFT target and context mix`.
