/** GB below 1000, TB (one decimal) from there: "1.2 TB", "640 GB" */
export function formatCapacity(gb: number): string {
  return gb >= 1000 ? `${(gb / 1000).toFixed(1)} TB` : `${Math.round(gb)} GB`
}

/** Capacity wording across a machine: "total 1.2 TB of 2.1 TB across 8 GPUs" */
export function totalCapacityLine(perGPUGB: number, perGPUCapacityGB: number, n: number): string {
  return `total ${formatCapacity(perGPUGB * n)} of ${formatCapacity(perGPUCapacityGB * n)} across ${n} GPUs`
}
