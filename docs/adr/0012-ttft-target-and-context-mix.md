# 0012. TTFT target and context mix

**Status:** Accepted, 2026-11-12
**Spec:** `docs/superpowers/specs/2026-11-12-ttft-target-and-context-mix-design.md`

## Context

Qwen3.6 35B A3B on an Apple M5 Max at 256K read "time to first token 29.0 min". The tool
could not say whether a setup keeps a response-time promise, and it priced every user as a lone
prompt filling the whole context. Two assumptions misled: every prompt is as long as the
context, and prompts never wait for each other. Prefill is compute-bound, so a burst queues.

## Decision

- **D1 User decisions (2026-11-12).**
  - A TTFT target (0.1 to 3600 s, optional) in Inference and Sizing; Inference says met or
    missed, Sizing sizes to it.
  - Load is the users entered, all concurrent. No active-share guess in Inference, which keeps
    all concurrent users in the burst (the safe direction, TTFT overstated with a tier). In
    Sizing with a KV tier the burst is the active sessions, as the tier already assumes.
  - Contexts differ per user: a mix of one to three classes, shares summing to 100% (presets
    Chat 4K, RAG 32K, Long documents 256K, sizes editable). The default is one class at the
    chosen context, so no existing figure changes.
  - KV memory and decode follow the mix; `sequenceLength` is the largest class.
  - The headline TTFT stays first served: in Inference a lone prompt at the largest class; in
    Sizing the figure the card always showed, priced at the replica's real decode batch and
    fabric hops. The last served sits under it and carries the target.
- **D2 The burst is a worst case, not "expected".** Last served = one largest-class prompt
  plus (B - 1) x the share-weighted per-class TTFT at batch 1: a burst holds at least one prompt
  of the largest class, so last is never below first (a plain B x mean fell below it with a
  skewed mix, e.g. 99% 4K / 1% 128K at burst 1). One class is unchanged (t + (B - 1)t = Bt).
  It is labelled "worst case: all N prompts at once, first come first served, no prefix
  cache". ADR 0009 keeps "expected" for medians of published points. No
  Poisson arrivals, no percentiles, no per-class worst case (in a first-come-first-served queue
  the total is the same whichever class is last).
- **D3 No double count of queueing.** The 0.469 prefill efficiency was fitted against mean TTFT
  with a closed-loop correction, so it already removes queueing, per the calibration note
  (`docs/calibration/2026-11-11-vllm-efficiency-points.md`). Each prompt's expected prefill
  times the queue length counts the queue once.
- **D4 Batch 1 per class.** `performance.ts` divides hop and pipeline-fill costs by the batch
  (ADR 0007); the queue multiplies a batch-1 prefill, so multi-server cost is not divided twice.
- **D5 Sizing.** Burst cap = the largest B whose last-served TTFT meets the target: 0 if the
  largest class alone misses, else floor((T - largest) / mean) + 1 (one class: floor(T / t)).
  Users per replica = min(memory sessions, burst cap). Replicas = ceil(users / users per replica). The
  result reads "Sized by memory" or "Sized by TTFT target". If one user per replica on every
  replica shape still misses, `neverMeetsTarget` names the largest class. With a KV tier the
  burst is the active sessions (Sizing only; see D1).
- **D6 Inference.** It models one replica, so more servers mean deeper pipelines, not a shorter
  queue. With a target set, it shows "Max users within target"; a miss also
  advises "{k} machines like this one (one replica each)".
- **D7 Memory guard.** One session at the largest class must fit at the 90% limit; otherwise
  "does not fit", naming that class.
- **D8 Mix-mean memory overrides "never too few units"** (user decision, 2026-11-12). Per-session
  KV is the share-weighted mean, so a rack sized on it can run short when many users land in the
  long class at once. The result says "memory sized on the mix average".
- **D9 Rules.** R16 (mix: 1 to 3 classes, shares sum to 100%, context range; `sequenceLength`
  follows the largest class), R17 (target 0.1 to 3600 s), W10 (per-class native-context warning).
  URL keys `cm` (mix) and `tt` (target).
- **D10 `perUserTimeToFirstToken` is removed.** It was a waves model (users / sessions batches);
  the Inference multi-user TTFT is now the queue of D2.

## Consequences

- The default mix and no target reproduce every earlier figure, in results, Sizing, Comparison,
  PDF and PPTX.
- A mix lowers KV per session and raises sessions held; the largest-class guard keeps the long
  prompt honest.
- Multi-user TTFT in Inference changes (waves to queue), a visible change for shared links with
  several users.
- Accepted gap (multi-server): the Sizing headline "first" is priced at the replica's real batch
  while the "last" line starts from the batch-1 largest class, so the two close figures can differ
  slightly; one-server figures are identical.
- Not modelled: prefix caching, prefill/decode disaggregation, speculative decoding.
