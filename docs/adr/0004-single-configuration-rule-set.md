# 0004. One configuration rule set, enforced in the store, every correction shown

**Status:** Accepted, 2026-09-27 (design approved; rule list finalized in the spec after a third verification pass)

## Context

An audit found input dependency rules enforced in 9 places (5 store setters,
2 engine modules, the URL schema, and ad-hoc hiding in ~6 components), each
covering some entry paths and not others. Reproduced consequences:
expert-parallel restored from a shared link onto a dense model (engine
error); training mode keeping an inference-only framework preset; CPU-RAM
offload and PCIe KV tiers offered on unified-memory GPUs; offloaded layers
beyond the model's layer count; training settings written into shared links
but never restored. The Grace KV-tier bug fixed earlier had the same cause.

## Decision

- One pure module (`src/engines/config-rules.ts`) holds a rule table. Each
  rule declares when it applies, the allowed values, the correction and a
  short message.
- `normalizeConfig(config, model, gpu)` returns a valid configuration plus
  the list of corrections; it is idempotent and independent of the order in
  which fields were set. `allowedOptions(...)` tells the UI what to offer,
  so impossible combinations are not selectable.
- The store applies `normalizeConfig` after every change; a shared link is
  restored in one step and normalized once. Engine throws remain as a last
  backstop only.
- Every correction is shown to the user, on interactive changes and on
  shared-link restore (product owner: "always warn the user"), grouped into
  one notice per action.
- Soft conditions stay warnings, never corrections: context beyond the
  model's native length, TP above the recommended degree, multi-node Apple
  or DGX Spark clusters, custom GPU without FLOPS, format vs GPU generation.
- Inputs that do not apply (for example multi-node in training mode) are
  hidden and ignored, not reset, so shared links are not damaged.

## Consequences

- One place to read, test and extend rules; each rule is tested on three
  paths (change the field, change its dependency, load a link) plus two
  properties (idempotent, order-independent).
- A shared link that encodes an impossible combination opens corrected, with
  a notice explaining why the numbers differ from the sender's.
- Every future input must declare its rules in the table.

## Sources

Three verification passes (code reproduction and vendor/documentation
sources), recorded in the spec.
