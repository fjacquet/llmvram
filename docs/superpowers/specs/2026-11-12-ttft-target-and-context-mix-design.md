# TTFT target and context mix

**Date:** 2026-11-12
**Status:** Draft, for the user's review (builder check done)
**Branch:** to create from `main` (v2.1.0 and later)

## Why

A presales user looked at Qwen3.6 35B A3B on an Apple M5 Max at a 256K context. The tool
said: time to first token 29.0 min, theoretical ceiling 13.6 min. That is unusable. Today the
tool cannot say whether a setup meets a response-time promise, and it cannot size for one.

Two assumptions make the figure misleading:

- **Every prompt fills the whole context.** `performance.ts` prices TTFT as one prompt of
  `sequenceLength` tokens. Real users mix short chats with a few long documents.
- **One request at a time.** TTFT is a lone request. With N users sending prompts at once, the
  last one waits for the others: prefill is compute-bound, so batching prompts does not speed
  them up.

The user wants to show customers that consumer hardware does not scale. A TTFT under load,
against a target, shows it.

## Decisions (by the user, 2026-11-12)

- **A TTFT target, in both modes.**
  - Inference says whether the setup meets it.
  - Sizing sizes to meet it: more replicas, or bigger replicas, until the target holds.
- **Load = the users entered, all concurrent.** There is no active-share guess. The number of
  concurrent users is shown beside the TTFT.
- **Contexts differ per user: a mix that sums to 100%.**
- **KV memory follows the mix, as vLLM allocates it.** It is not the maximum for everyone.
- **Headline TTFT stays first served** (a lone request, today's figure). The last served sits
  under it and carries the target. With the default mix no existing figure changes, in results,
  Sizing, Comparison, PDF and PPTX.
- **With a KV tier, the burst is the active sessions**: ceil(open sessions x active share), the
  assumption the tier already shows. A resumed session reloads its KV and does not prefill. Without
  a tier, the burst is all users.
- **One target, global worst case.** The target applies to the last prompt served in the burst, all classes together, in a first-come-first-served queue. A short chat prompt can wait behind long documents, and the result says so.

## How it works

### Context mix (input, both modes)

- **Classes:** one to three classes, each a context size and a share of users. The shares
  always sum to 100%. The last class takes the remainder, so the total cannot be wrong.
- **Presets** (sizes editable, _our call_): Chat 4K, RAG 32K, Long documents 256K.
- **Default:** one class, 100% at the chosen context. With the default, every figure is
  unchanged from today.
- **URL and store:** the mix is part of the shared link. Rules declare it in `RULES` (mode
  lists, `allowedOptions` for the sizes, the 100% invariant). A shared link whose shares do not
  sum to 100% is corrected and noticed, as other rules do.

### Memory with a mix

- **Per-session KV is the mix-weighted sum.** For example, 90% of users at 4K and 10% at 256K
  reserve 0.9 x KV(4K) + 0.1 x KV(256K) per session on average. vLLM's paged KV allocates by
  tokens actually present.
- **The longest class must still fit.** One session at the largest context must fit at the 90%
  limit. Otherwise the result is "does not fit", naming that class.
- **Sessions held and Max sessions** use the mix-weighted KV per session.
- **Sliding-window, linear-state and sparse-attention models** keep their existing per-context
  rules. Each class is priced by the existing engine at its own context and then weighted.
  There is no second formula (DRY).

### TTFT under load

- **Per-class prefill.** The prefill time of one prompt at each class's context comes from the
  existing engine (`performance.ts`, the expected figure = ceiling / `PREFILL_EFFICIENCY`).
- **Queue.** N concurrent users send a prompt at once, the burst being the users of one replica.
  Prompts are processed back to back at the machine's prefill rate.
  - **First served:** the prefill of one prompt.
  - **Last served:** the sum of the prefill times of all prompts in the burst, with classes in
    their shares.
  - **Per class, worst case:** a prompt of that class served last.
- **Shown:**
  - "TTFT 4.2 s first, 3.1 min last (100 users at once)".
  - Each class's own prompt time.
  - The ceiling beside each, as today.
- **The target applies to "last served"** (the safe direction). Met or missed, with by how
  much.
- **Decode with a mix** reads each session's KV at its own context. Decode bytes per step use
  the mix-weighted KV per session.

### Inference mode

- **New optional input "TTFT target" (seconds).** Empty = no target, nothing changes.
- **Verdict line beside the memory verdict:** "TTFT target met", or "missed: 3.1 min for the
  last of 100 users, target 10 s".
- **Recommendations:** when missed, advise more GPUs or servers, or fewer users per replica,
  and say when no machine size of this GPU can meet it.

### Sizing mode

- **The target becomes a constraint of the solver.** A replica size is valid only if its
  last-served TTFT, with the users per replica, meets the target.
- **Smaller bursts:** more replicas mean fewer users per replica and a shorter queue.
- **Faster prefill:** bigger replicas (more GPUs per replica, among the valid TP degrees) speed
  up the prefill.
- **Units still count by memory too:** the answer is the larger of the memory count and the
  TTFT count. The result says which one decided.
- **Never meets:** if even one user per replica at the largest replica misses the target, the
  result says so and names the class that drives it.

## Goals

1. A TTFT target in Inference and Sizing, met or missed, in plain words.
2. A TTFT under load that shows consumer hardware does not scale.
3. A context mix that makes memory and TTFT realistic, defaulting to today's figures.
4. One engine: classes reuse the existing per-context prefill, KV and decode functions.

## Non-goals

- Arrival-rate (Poisson) queueing, p50/p99 percentiles. The burst of all users at once is the
  model. _Our call:_ it is simple and safe.
- Prefix caching and prefill/decode disaggregation.
- Speculative decoding.
- Per-class decode-speed targets.

## Anti-goals

- A TTFT that ignores the queue when users are concurrent.
- An average prompt length used for TTFT. The quadratic cost makes that unsafe.
- A mix that changes today's figures when left at its default.
- A second copy of the prefill, KV or decode logic.

## Success criteria

- `npx vitest run`, `tsc -b --force`, `./node_modules/.bin/biome check .` (full output read)
  and `npm run build` are green.
- **Engine tests on real models:**
  - The default mix gives today's figures.
  - With a mix, KV per session equals the weighted sum.
  - Last-served TTFT for B identical prompts is B x one prompt.
  - Missing the target adds replicas in Sizing.
  - The M5 Max case (Qwen3.6 35B A3B, 256K, N users) misses a 10 s target, and the result says
    why.
- **Rule tests:** shares sum to 100%; a shared link is corrected; the default mix restores old
  links unchanged.
- **Docs:** an ADR (0012), CHANGELOG, README "How to read the figures", the Guide, and PDF/PPTX
  carrying the mix, the target and the TTFT first/last.

## Calls made after the builder check (_our call_, 2026-11-12)

- **Burst and prefill are per replica.**
  - Inference models one replica (`numGPUs` x `numNodes`), so the burst is the concurrent users
    entered.
  - Sizing spreads users evenly over replicas, so the burst is ceil(users / replicas).
  - Prefill runs on the replica's own GPUs; replicas do not share a queue.
- **Fractional users are expected values.** Class counts are users x share, not rounded. The
  queue sums each class's prefill times its expected count. With one class at 100%, the sum is
  exactly B x one prompt, so today's figures are reproduced.
- **Memory uses the mix mean.** KV per session is the weighted mean (the user's decision). One
  extra guard: one session at the largest class must fit at the 90% limit. There is no "all users
  in the largest class" worst case.
- **TTFT target input.**
  - Optional, in seconds, 0.1 to 3600, empty by default.
  - Shown in Inference and Sizing, hidden in Training. Declared in `RULES` with its mode list,
    and carried in the shared link.
- **Sizing search.**
  - Sizing lists racks only. Replicas are the R14-valid TP degrees up to 72, inside one rack, as
    today.
  - Tie-break unchanged: fewest units, then fewest GPUs, then the smaller replica.
  - A candidate is valid when it meets both memory and the TTFT target.
- **The M5 Max test** uses 100 users. A single user at 256K already misses 10 s.

## Calls made after the advisor review (_our call_, 2026-11-12)

- **"Last served" is a worst case, not an "expected" figure.** It is labelled "worst case: all N
  prompts at once, first come first served, no prefix cache". ADR 0009 keeps "expected" for medians
  of published points.
  - The prefill efficiency 0.469 already removes queueing: it was fitted against mean TTFT with a
    closed-loop correction 0.9 + 0.1 x (c + 1) / 2 (`docs/calibration/2026-11-11-vllm-efficiency-points.md`).
    Each prompt's expected prefill (ceiling / 0.469) times the queue therefore does not count the
    queue twice.
  - Where the calibration has TTFT rows at several concurrencies, a test checks the queue model
    against them and records the gap.
- **Each per-class engine call runs at batch 1.** `performance.ts` divides hop and pipeline-fill
  costs by the batch (ADR 0007), so the queue multiplies a batch-1 prefill. A multi-server
  Inference test pins this.
- **One solver formulation.**
  - Burst cap = the largest B whose last-served TTFT meets the target.
  - Users per replica = min(memory sessions, burst cap).
  - Replicas = ceil(users / users per replica).
  - The result says which bound decided: memory or TTFT.
- **Inference shows "max users within target"** (the burst cap). This is the "consumer hardware
  does not scale" figure. Inference cannot add replicas, and more servers there means deeper
  pipeline stages, not a shorter queue. So when the target is missed the recommendation reads
  "N machines like this one (one replica each)", not "add servers".
- **No per-class worst case.** In a first-come-first-served queue the total is the same whichever
  class is last. Each class shows its own solo prompt time.
- **`sequenceLength` with a mix = the largest class.** Consumers that stay on it, listed so none is
  missed:
  - the native-context warning, checked per class (a 256K preset on a 128K model warns);
  - the "Max sessions at N tokens" label, which states the mix;
  - the KV tier resume/recompute and tier traffic, priced on the mix mean KV;
  - offloading and multi-node hop bytes, on the mix;
  - "Reduce context" recommendations, which act on the largest class;
  - `assumptionRows` (PDF), the PPTX slide and the Comparison snapshot, which list the mix and
    the target.
- **The server branch of `sizing.ts` is not extended.** It is unreachable from the UI (ADR 0010
  amendment).
- **ADR 0012 records that mix-mean memory overrides "never too few units".** This is the user's
  decision, 2026-11-12. The result reads "memory sized on the mix average".

## Left to the builder

- The exact layout of the mix editor and the TTFT line.
- How the worker message carries the mix.
- The ADR 0012 text.
