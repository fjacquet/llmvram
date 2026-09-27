/**
 * Label of the first-token figure. For B > 1 the engine's figure is the prefill of a
 * burst of B prompts divided by B (ADR 0007), not one request's wait: in a real burst
 * the last request waits about B times longer.
 */
export function firstTokenLabel(batchSize: number): string {
  return batchSize > 1
    ? `Prefill per request (amortized over batch ${batchSize})`
    : 'Time to first token'
}
