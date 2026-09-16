# .planning

What remains here is **research**, not status.

The status tracking this directory once held — `STATE.md`, `ROADMAP.md`,
`MILESTONES.md`, and the per-phase `PLAN` / `SUMMARY` / `VERIFICATION`
files — went stale several releases ago and was removed. Treat
`CHANGELOG.md` and the git history as the record of what shipped.

The files kept are the research behind the implementation:

- `research/` — stack, architecture, feature and summary research, plus
  `PITFALLS.md`, the domain pitfalls reference cited from `CLAUDE.md`.
- `phases/*/NN-RESEARCH.md` — per-area research, all ten kept.

Four of those phase files (`02`, `06`, `08`, `10`) plus `PITFALLS.md` are
cited by `Reference:` comments in `src/engines/`, where they explain why
constants are what they are: FP32 optimizer states, quantization overhead
multipliers, activation memory in training versus inference. Deleting one
of those orphans the comments that cite it.

The other six phase files are uncited but kept deliberately — they are the
research for areas whose code carries no `Reference:` comments, not dead
weight. Being uncited is not a reason to remove one.
