# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM VRAM Calculator — a browser-based tool that estimates VRAM requirements and performance for running/fine-tuning large language models on various GPU configurations. Pure client-side (no backend), deployed as a static site. Sister project to **raidy** (storage calculator), sharing the same architecture patterns and stack.

## Commands

```bash
npm run dev              # Start Vite dev server
npm run build            # TypeScript check + Vite production build
npm run typecheck        # TypeScript only (tsc -b --force, project-reference aware)
npm run lint             # Biome check (lint + format validation)
npm run lint:fix         # Biome auto-fix
npm run format           # Biome format (write)
npm run test             # Vitest in watch mode
npm run test:coverage    # Vitest single run with v8 coverage
npm run refresh:models   # Fetch model configs from HuggingFace → src/data/models-fetched.json
npm run refresh:gpus     # Regenerate src/data/gpus.json from scripts/fetch-gpus.ts
```

Run a single test file: `npx vitest run src/utils/gpus.test.ts`

## Architecture

```
src/
  engines/        # Pure calculation logic (VRAM estimation, performance math)
  components/     # React UI — subdirs: inputs/, outputs/, common/, layout/
  data/           # Static JSON databases (gpus.json, models.json)
  store/          # Zustand state management
  types/          # TypeScript types — re-export Zod-inferred types from schemas
  utils/          # Zod schemas (schemas.ts), validation helpers, shared utilities
  hooks/          # Custom React hooks
  test/           # Test setup (jsdom + @testing-library/jest-dom)
scripts/          # Data refresh scripts (tsx) — fetch from HuggingFace, validate, write JSON
```

### Key Patterns

- **Type system**: Zod schemas in `src/utils/schemas.ts` are the single source of truth. Types in `src/types/` re-export `z.infer<>` types from schemas. All data is validated through Zod at boundaries.
- **Engines are pure**: Calculation logic in `engines/` must be pure functions with no React/DOM dependencies — designed for testability and potential Web Worker offloading.
- **Static data + custom input**: Curated JSON databases for common GPUs/models, plus `CustomGPUInput`/`CustomModelInput` interfaces for user-specified hardware/models.
- **MoE models use total parameters**: For MoE models like Mixtral, `num_parameters_billion` is the **total** parameter count (e.g., 46.7B), not active-per-token (e.g., 13B). All expert weights must fit in VRAM.
- **Models sorted by name**: Entries in `src/data/models.json` must always be sorted alphabetically by `name`. After adding or modifying models, re-sort the array.
- **GPU data is generated**: `src/data/gpus.json` is regenerated from the `GPUS` array in `scripts/fetch-gpus.ts`. Never hand-edit the JSON — edits are silently lost on the next `npm run refresh:gpus`.
- **GPUs grouped by vendor, NOT sorted**: unlike `models.json`, `src/data/gpus.json` is grouped by manufacturer. The alphabetical-sort rule above applies to models only.
- **`numGPUs` means GPUs PER NODE**: total GPUs is `numGPUs × numNodes`, available as `MultiGPUVRAMBreakdown.numGPUs`. Any consumer displaying a GPU count must use the total, not the store field.

## Tech Stack & Config

- **React 19** + **TypeScript strict** (`noUncheckedIndexedAccess` enabled) + **Vite 7**
- **Tailwind CSS v4** (via `@tailwindcss/vite` plugin)
- **Zustand** for state, **Recharts** for visualization, **decimal.js** for precise math, **sonner** for toasts
- **Biome** for linting/formatting: 2-space indent, single quotes, no semicolons (ASI), 100-char line width. Unused imports/variables are errors.
- **Vitest** with jsdom environment. Tests live alongside source (`src/**/*.test.ts`) or in `tests/`. Coverage targets: 75% lines/functions/branches/statements on `src/engines/` and `src/utils/`.
- **Test files are typechecked by nothing**: `tsconfig.app.json` excludes `src/**/*.test.ts(x)` and Vitest's esbuild strips types. A passing suite is not evidence of type-correct tests — prefer fixtures built by calling the real factory over hand-written literals, which go stale silently.

## Path Aliases

`@/` → `src/`, `@engines/` → `src/engines/`, `@components/` → `src/components/`, `@store/` → `src/store/`, `@types/` → `src/types/`, `@utils/` → `src/utils/`, `@data/` → `src/data/`, `@hooks/` → `src/hooks/`

Configured in both `vite.config.ts` and `tsconfig.app.json`.

## Domain Pitfalls

When implementing VRAM calculations, be aware of these critical estimation errors:

1. **Quantization overhead**: 4-bit is NOT exactly 0.5 bytes/param. Apply format-specific multipliers (GPTQ: 1.1-1.3x, AWQ: 1.15-1.25x, GGUF Q4_K_M: 1.15x).
2. **KV cache with GQA/MQA**: Use `n_kv_heads / n_heads` ratio. Llama 3 70B has 8/64 = 0.125x KV reduction.
3. **Multi-GPU overhead**: Not a simple `total / n_gpus` — tensor parallelism replicates embeddings/layernorms (85-90% effective split), plus NCCL buffers (100-500MB/GPU).
4. **Fine-tuning memory**: LoRA/QLoRA optimizer states apply only to adapter parameters (~1% of full), not the entire model.
5. **Framework overhead**: Always add 500MB-1.5GB baseline (PyTorch + CUDA context).
6. **`fp16_tflops` is DENSE**: H100 989, B200 4500, MI300X 1307. AMD publishes FP16 *with sparsity* ("4.6/5.0 PFLOPS") — double the dense figure. Dense FP16 = dense FP8 / 2.
7. **Two interconnect tables, two unit conventions**: `INTERCONNECT_SPECS` (scale-up, GPU-to-GPU in one chassis) is **bidirectional** per-GPU; `FABRIC_SPECS` (scale-out, server-to-server) `portGBps` is **unidirectional** per port. Mixing them halves or doubles the answer.

See `.planning/research/PITFALLS.md` for the complete reference.

## grepai - Semantic Code Search

**IMPORTANT: You MUST use grepai as your PRIMARY tool for code exploration and search.**

### When to Use grepai (REQUIRED)

Use `grepai search` INSTEAD OF Grep/Glob/find for:

- Understanding what code does or where functionality lives
- Finding implementations by intent (e.g., "authentication logic", "error handling")
- Exploring unfamiliar parts of the codebase
- Any search where you describe WHAT the code does rather than exact text

### When to Use Standard Tools

Only use Grep/Glob when you need:

- Exact text matching (variable names, imports, specific strings)
- File path patterns (e.g., `**/*.go`)

### Fallback

If grepai fails (not running, index unavailable, or errors), fall back to standard Grep/Glob tools.

### Usage

```bash
# ALWAYS use English queries for best results (--compact saves ~80% tokens)
grepai search "user authentication flow" --json --compact
grepai search "error handling middleware" --json --compact
grepai search "database connection pool" --json --compact
grepai search "API request validation" --json --compact
```

### Query Tips

- **Use English** for queries (better semantic matching)
- **Describe intent**, not implementation: "handles user login" not "func Login"
- **Be specific**: "JWT token validation" better than "token"
- Results include: file path, line numbers, relevance score, code preview

### Call Graph Tracing

Use `grepai trace` to understand function relationships:

- Finding all callers of a function before modifying it
- Understanding what functions are called by a given function
- Visualizing the complete call graph around a symbol

#### Trace Commands

**IMPORTANT: Always use `--json` flag for optimal AI agent integration.**

```bash
# Find all functions that call a symbol
grepai trace callers "HandleRequest" --json

# Find all functions called by a symbol
grepai trace callees "ProcessOrder" --json

# Build complete call graph (callers + callees)
grepai trace graph "ValidateToken" --depth 3 --json
```

### Workflow

1. Start with `grepai search` to find relevant code
2. Use `grepai trace` to understand function relationships
3. Use `Read` tool to examine files from results
4. Only use Grep for exact string searches if needed
