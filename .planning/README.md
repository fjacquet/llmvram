# .planning

What remains here is **research**, not status.

The status tracking this directory once held — `STATE.md`, `ROADMAP.md`,
`MILESTONES.md`, and the per-phase `PLAN` / `SUMMARY` / `VERIFICATION`
files — went stale several releases ago and was removed. Treat
`CHANGELOG.md` and the git history as the record of what shipped.

The files kept are the derivations behind the calculation engines:

- `research/PITFALLS.md` — the domain pitfalls reference cited from
  `CLAUDE.md` and from `src/engines/`.
- `research/` — stack, architecture, feature and summary research.
- `phases/*/NN-RESEARCH.md` — per-area research for the inference,
  fine-tuning, optimization and framework-preset engines.

These are cited by `Reference:` comments throughout `src/engines/`, which
is why they survive: they explain why constants are what they are (FP32
optimizer states, quantization overhead multipliers, activation memory in
training vs inference). Deleting one orphans those comments — grep for its
path before removing anything here.
