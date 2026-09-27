# 0005. Presales/datacenter audience: essential vs advanced inputs, feature freeze

**Status:** Accepted, 2026-09-27

## Context

The interface grew fast: 8 always-visible inputs, 9 conditional groups and
up to 12 result blocks in inference mode, with no collapsible grouping. The
main answer ("does it fit, how fast") was getting lost. The product owner
asked whether we were over-complicating it ("le mieux est l'ennemi du bien").

## Decision

- Primary audience: presales engineers sizing datacenter configurations
  (HGX, NVL72), not the general public. Advanced settings stay available,
  collapsed.
- Essential inputs stay visible (mode, model, GPU with GPUs per server and
  servers, format, context, concurrent users). Advanced inputs (batch, KV
  precision, strategy, fabric, interconnect variant, offloading, KV tier)
  move into a collapsed section that opens automatically when any of them
  differs from its default.
- Results lead with a verdict block (fits, speed, first-token delay,
  sessions); warnings stay visible; the detail is collapsed. The PDF export
  expands everything before capture.
- Feature freeze: no new inputs or options until this simplification ships.
  UI additions that only add information (for example the MoE split label)
  are dropped.
- Exception to the freeze: two reset buttons ("Reset advanced settings" in
  the Advanced section, "Reset all" in the header). They add no calculation
  and return to a known-good configuration (the defaults, always valid under
  ADR 0004).

## Consequences

- The UI depends on ADR 0004 (`allowedOptions`) so collapsed inputs can
  never hold an impossible value unseen.
- Calculations, store fields and shared-link format are unchanged by the
  layout work.
