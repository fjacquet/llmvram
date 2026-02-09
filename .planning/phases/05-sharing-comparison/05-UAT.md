---
status: testing
phase: full-project
source: 03-01-SUMMARY.md, 03-02-SUMMARY.md, 03-03-SUMMARY.md, 03-04-SUMMARY.md, 04-01-SUMMARY.md, 04-02-SUMMARY.md, 04-03-SUMMARY.md, 05-01-SUMMARY.md, 05-02-SUMMARY.md, 05-03-SUMMARY.md
started: 2026-02-09T21:50:00Z
updated: 2026-02-09T21:50:00Z
---

## Current Test

number: 1
name: Model Selection & Search
expected: |
  Open http://localhost:5177. In the model selector, type "llama" — filtered list shows LLaMA models. Select "Llama 3.1 70B". Also try selecting "Custom..." and entering custom model parameters.
awaiting: user response

## Tests

### 1. Model Selection & Search
expected: Search/filter across 34 curated models; select a model; try "Custom..." option with manual parameter entry
result: [pending]

### 2. GPU Selection & Search
expected: Search GPUs by name; select "H100 80GB SXM"; try "Custom..." with manual VRAM/bandwidth entry
result: [pending]

### 3. Quantization Format Selection
expected: See 22 quantization formats in 5 groups (Float, NVIDIA FP, Integer, GPTQ/AWQ, GGUF); select GPTQ; VRAM results update
result: [pending]

### 4. Parameter Sliders
expected: Adjust sequence length slider (log-scale, 512 to 131K); adjust batch size (1-64); change KV cache quantization; results update in real-time
result: [pending]

### 5. VRAM Breakdown Chart & Table
expected: Donut/pie chart shows weights, KV cache, activations, overhead with colors and labels; table below shows GB values and percentages
result: [pending]

### 6. Fit Indicator
expected: Color-coded progress bar (green ≤80%, yellow 81-95%, red >95%); shows "X.XX GB / Y GB (Z%)" with clear fit/doesn't fit status
result: [pending]

### 7. Recommendations Panel
expected: When model doesn't fit (e.g., Llama 3.1 405B on RTX 4090 FP16), shows actionable suggestions with estimated savings
result: [pending]

### 8. Dark Mode Toggle
expected: Click moon/sun icon; entire UI switches theme; preference persists after page refresh
result: [pending]

### 9. Multi-GPU Configuration
expected: Set GPUs to 4; choose Tensor Parallel; see per-GPU VRAM in stacked bar chart; interconnect badge shows NVLink/PCIe
result: [pending]

### 10. Offloading Controls
expected: Enable offloading toggle; select CPU/RAM target; adjust percentage slider; see offloading summary in results with performance impact
result: [pending]

### 11. Performance Estimate
expected: Below VRAM section: decode speed (tokens/sec), TTFT (ms), bottleneck (color-coded: balanced/memory/compute)
result: [pending]

### 12. URL Hash Sharing
expected: Select model/GPU/params; URL hash updates; copy URL to new tab; exact configuration restored; dark mode NOT transferred
result: [pending]

### 13. Share Button
expected: Click share/link icon in header; toast says "Link copied to clipboard"; pasted URL is valid with hash
result: [pending]

### 14. Comparison Flow
expected: Save 2-3 configs via "Save to Compare"; switch to Comparison tab; see side-by-side cards with diff highlighting (amber for differences)
result: [pending]

### 15. Responsive Layout
expected: On mobile width: cards stack vertically; on desktop: side-by-side with sticky input panel; comparison adapts columns
result: [pending]

## Summary

total: 15
passed: 0
issues: 0
pending: 15
skipped: 0

## Gaps

[none yet]
