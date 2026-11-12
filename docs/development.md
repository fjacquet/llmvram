# Developer guide

Run locally, test, and find your way around the code. Conventions and domain rules for
contributors (and coding agents) are in [CLAUDE.md](../CLAUDE.md); the design is in
[ARCHITECTURE.md](../ARCHITECTURE.md).

## Run locally

```bash
npm install
npm run dev
```

Open <http://localhost:5173>.

## Commands

```bash
npm run build          # TypeScript check + Vite production build
npm run typecheck      # TypeScript only (tsc -b --force, project-reference aware)
npm run lint           # Biome linting
npm run lint:fix       # Biome auto-fix
npm run format         # Biome format
npm run test           # Vitest watch mode
npm run test:coverage  # Coverage report (75% thresholds on engines/utils)
npm run refresh:all    # Refresh model/GPU data from HuggingFace
```

Or use `make help` for all available targets.

## Technology Stack

- React 19 + Vite 7
- TypeScript (strict mode, `noUncheckedIndexedAccess`)
- Zustand (state management with URL hash persistence)
- Tailwind CSS v4 (dark mode)
- Recharts (visualization)
- Decimal.js (precision arithmetic)
- Zod (schema validation, single source of truth for types)
- Web Workers (non-blocking calculations)
- Biome (linting + formatting)
- Vitest (testing)

## Project Structure

```
src/
├── engines/        # Pure calculation logic (VRAM, performance, multi-GPU, offloading)
├── components/
│   ├── inputs/     # Model/GPU selectors, quantization, parameters
│   ├── outputs/    # Charts, tables, fit indicator, recommendations
│   ├── comparison/ # Side-by-side config comparison
│   ├── common/     # Dark mode toggle, shared components
│   └── layout/     # App layout, input/results panels
├── data/           # Static JSON databases (gpus.json, models.json)
├── store/          # Zustand stores (UI state, comparison, URL serialization)
├── hooks/          # useInferenceCalculation, useDarkMode, useURLSync
├── utils/          # Zod schemas, validation helpers
├── types/          # TypeScript type re-exports from Zod
├── workers/        # Web Worker for calculation offloading
└── test/           # Test setup
scripts/            # Data refresh scripts (HuggingFace fetch)
```

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed technical documentation.
