# Piece 2: Sizing mode (users in, servers or racks out)

**Date:** 2026-11-11
**Status:** Approved by the user 2026-11-11 (subagent-driven, Sonnet subagents)
**Series:** piece 2 of the rack-sizing effort. Piece 1 (rack data, PR #78) and the calibration piece (PR #79) come before it.
**Branch:** `feat/sizing-mode`, stacked on `feat/calibration-sparse-attention`. The user merges everything at the end (2026-11-11).

## Why

The user is a presales engineer. A customer gives three things: a model, a context size and a number of users. The customer expects a number back: how many servers or how many racks.

Today the tool answers the opposite question: "does this model fit on these GPUs, and how fast is it?" To get a rack count, the user has to work it out by hand, one replica at a time.

Rack GPUs (NVL72) also don't belong in the per-replica calculator. Nobody runs one replica on a whole 72-GPU rack. A rack is bought as a unit and filled with replicas.

## Decisions (by the user, 2026-11-11)

- **Input:** model, context size, number of users.
- **One calculator, a third mode.** "Sizing" sits beside Inference and Training (mockup C). There is no new tab. To compare platforms, the user saves results to the existing Comparison tab.
- **Rack GPUs move to Sizing.** GB200 NVL72, GB300 NVL72 and Vera Rubin NVL72 can be selected only in Sizing. Sizing also offers every server GPU (HGX, MI355X, and so on).
- **Answer:** in large type, "N racks <GPU>" or "N servers <GPU>". Below it: the replica size, sessions per replica and the expected per-user speed, with the ceiling beside it.
- **Unit counts use expected speed,** never the ceiling. This was decided in the calibration piece.
- **Engineering principles:** KISS, DRY, functional programming. Sizing reuses the existing pure engine functions (memory, sessions, expected performance) and adds one small pure solver. There is no second engine.

## How it works

### Solver (pure function, `src/engines/sizing.ts`)

`sizeDeployment(model, gpu, input)` returns a `SizingResult`. The input is quantization, KV precision, context and users.

1. **Candidate replicas.** These are the tensor-parallel degrees vLLM accepts for this model. Use the existing R14 rule (heads % t, the KV-head rule) via `allowedOptions`. Do not write a new list.
   - **Servers:** degrees up to `max_gpus_per_node`. When one server is not enough, the replica spans servers as pipeline stages, through the existing multi-node engine (1-8 nodes).
   - **Racks:** degrees up to 72, inside the NVLink domain. A replica never spans racks.
2. **Fit.** A candidate is valid when `fitsInference` holds at the 90% limit with at least one session at the context.
3. **Sessions per replica.** Use the existing `maxConcurrentSessions` at the context.
   - Per-user speed (shown, not a constraint) = expected decode tok/s at batch = sessions ÷ sessions.
4. **Replicas:** ceil(users / sessions per replica).
5. **Units.** First, replicas per unit:
   - **Server:** 1 when the replica holds `max_gpus_per_node` GPUs; otherwise floor(GPUs per server / replica GPUs).
   - **Rack:** floor(72 / replica GPUs).

   Then units = ceil(replicas / replicas per unit). For a multi-server replica, units = replicas × nodes.
6. **Pick the candidate with the fewest units.** On a tie, take the fewest total GPUs, then the smaller replica (it scales better and is less exposed to failures). _Our call._
7. **Never fits:** no candidate fits even with the whole unit. Return a "does not fit" result naming the closest candidate and how much memory it lacks.

All of these are pure functions of their inputs. The UI calls `sizeDeployment` in the same worker that already runs the engine.

### Screen (Sizing mode)

- **Inputs:** model, quantization, KV precision, GPU, context, users.
- **Hidden in Sizing, because the solver chooses them:** GPUs per replica, servers, sharding strategy, batch size, offloading and the KV tier. Config rules handle this per mode, as they already do. A field hidden as inert is never corrected.
- **Results:**
  - The headline: "N racks GB300 NVL72" or "N servers HGX B300".
  - A summary line: replica = T GPUs (tensor parallel, × P servers when it spans servers), R replicas, R per rack, S sessions per replica.
  - Expected per-user tok/s and TTFT, with the theoretical ceiling beside each.
  - The 72-GPU split for racks, for example "9 replicas × 8 GPUs, 0 GPUs idle". Idle GPUs are shown when 72 is not a multiple of the replica size.
- **Save to Compare:** Comparison shows the unit count and the replica summary as new rows. _Our call._
- **Exports:** PDF and PPTX carry the sizing summary. They do not show the per-replica charts. _Our call._

### Rack GPUs leave Inference and Training

- **Mode rule:** a rule in `config-rules.ts` says a GPU with `max_gpus_per_node > 8` is selectable in Sizing only.
  - **Choosing a rack GPU** in Inference or Training switches the mode to Sizing, with a toast ("Adjusted for …"). _Our call:_ it switches the mode rather than refusing the GPU.
  - **A shared link** with a rack GPU in Inference or Training restores in Sizing, with the existing "Shared link adjusted" message. Ids are never renamed.
- **Breaking change:** training on NVL72 is no longer offered. The CHANGELOG says so.

## Gates (from the calibration review; done first in this piece)

1. **fp8_ds_mla KV layout.**
   - Today the engine charges 576 B per token per layer for FP8 MLA KV. The final review said vLLM's `fp8_ds_mla` layout is 656 B.
   - Verify this in the vLLM source (Context7 or GitHub, cite the file).
   - If it is confirmed, charge 656 at FP8 KV on MLA models. Re-pin each test with a dated reason.
   - The safe direction applies: if it cannot be confirmed, charge 656 anyway and say so in the ADR.
2. **TP degrees vLLM refuses.**
   - The solver takes candidates from `allowedOptions` only.
   - The old Recommendations panel ("Use 3x" for a model with 64 heads) also switches to the same source. That is a one-line DRY fix.
3. **Batch-dependent decode efficiency.** Use the load bands from the user's answers below.

## Goals

1. Model + context + users in; servers or racks out, in one step.
2. Every count uses the calibrated expected speed and the 90% fit rule.
3. Only TP degrees vLLM accepts.
4. One engine. The solver is small and pure, and it is tested on its own.

## Non-goals

- Speculative decoding. It is not modelled; maybe later. _Our call._
- Several models in one rack, and mixed GPU types.
- Price, power and floor-space figures.
- Prefill/decode disaggregation and expert-parallel replica layouts across a rack. Replicas are TP within the domain, plus PP across servers for server GPUs.
- Training in Sizing.

## Anti-goals

- A unit count built on the ceiling.
- A replica layout vLLM would refuse.
- A second copy of the memory or speed logic.
- Hiding that a rack has idle GPUs.

## Success criteria

- `npx vitest run`, `tsc -b --force`, `biome check .` and `npm run build` are green.
- **Solver tests (pure, on real models from `models.json`):**
  - A small model on HGX gives several replicas per server.
  - A model larger than one HGX server gives a multi-server replica.
  - The rack split is right, including idle GPUs.
  - Refused TP degrees are never chosen.
  - Doubling the users doubles the replicas (up to rounding).
  - "Does not fit" names the closest candidate.
- **Mode rule tests:** a rack GPU switches to Sizing; a shared link with a rack GPU restores in Sizing.
- **Gate tests:** the fp8_ds_mla re-pins; Recommendations never advise a refused TP degree.
- Docs: ADR 0010, CHANGELOG, README, CLAUDE.md (the Sizing pattern), and the in-app guide.

## Answers from the user (2026-11-11)

1. **Decode efficiency by load band**, in every mode (one engine). The band is chosen by the decode batch (concurrent sequences per replica):
   - up to 32: 0.47;
   - 33 to 128: 0.425;
   - above 128: 0.25, labelled "provisional (1 point)".

   Bands come from the committed calibration points. Median per concurrency band: 2-8 0.481 (n=203), 9-32 0.472 (n=188), 33-128 0.425 (n=103), >128 0.252 (n=1).
   - This replaces the flat `DECODE_EFFICIENCY`. Inference speeds at high batch drop too; list that as a breaking change.
   - Re-pin the expected-speed tests with a dated reason.
   - The ceiling is unchanged. Prefill keeps 0.469.
2. **Users = sessions active at the same time.** No concurrency-rate input.
3. **No per-user speed floor.** Show the per-user speed obtained; it does not constrain the solver.

## Calls made after the builder check (_our call_, 2026-11-11)

- **Multi-server replicas (server GPUs only).**
  - The candidate grid is TP t in the R14-valid degrees ≤ `max_gpus_per_node`, with nodes = 1.
  - Plus TP = `max_gpus_per_node` (a full server) × nodes 2..8 as pipeline stages.
  - TP is never partial across servers, so no GPU sits idle across nodes. vLLM practice is TP inside a server and PP across servers.
  - A multi-server replica's units = replicas × nodes.
- **Rack replicas.**
  - TP only, inside the 72-GPU domain, nodes = 1, no PP.
  - Candidates are every R14-valid degree ≤ 72.
  - `recommendedMaxTPDegree` and the soft warnings flag a candidate; they never exclude it.
  - Ties at equal units go to the fewest GPUs, then the smaller replica. In practice that is usually 8.
- **Mode and store.**
  - `ConfigMode` becomes `'inference' | 'training' | 'sizing'`, and the URL `m` accepts `'sizing'`.
  - Sizing reuses `concurrentUsers` (users) and `sequenceLength` (context).
  - The solver output lives in the results only. It never writes `numGPUs` or `numNodes`.
  - Every rule in `RULES` gets an explicit mode list. Sizing runs the model, GPU, quantization and KV-precision rules; the rules for the fields Sizing hides do not run there.
- **Rack GPUs outside Sizing.**
  - **Selector:** the GPU selector in Inference and Training lists only `allowedOptions`, so rack GPUs are absent.
  - **Shared link:** a link with a rack GPU in Inference or Training restores with `mode: 'sizing'`. The only rule that writes `mode` runs on restore, before the mode-gated rules.
  - **Switching back:** going from Sizing back to Inference or Training with a rack GPU selected clears the GPU, with a toast: "<GPU> is sized as a rack; pick a server GPU".
  - **Recommendations** never suggest a rack GPU.
- **fp8_ds_mla.** The layout is an engine rule keyed on `use_mla` × FP8 KV (a vLLM layout property, not model data), and it applies in every mode. The vLLM source must confirm it (gate 1).
- **Comparison.** New optional fields, blank for Inference items. `comparisonStore` is not persisted, so there is no schema version.

## Left to the builder

- The exact field layout of the Sizing results card. It follows the existing results style.
- How the worker message carries the sizing input.
- The ADR 0010 text.
