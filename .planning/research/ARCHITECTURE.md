# Architecture Patterns: LLM VRAM Calculator

**Domain:** Browser-based technical calculator (LLM VRAM estimation)
**Researched:** 2026-02-09
**Baseline:** raidy's architecture pattern
**Confidence:** MEDIUM (based on established React patterns and raidy reference)

## Recommended Architecture

### High-Level Structure

```
llmvram/
├── src/
│   ├── engines/              # Pure calculation logic (framework-agnostic)
│   │   ├── inference.ts      # VRAM for inference calculations
│   │   ├── finetuning.ts     # VRAM for training calculations
│   │   ├── multigpu.ts       # Multi-GPU sharding calculations
│   │   ├── performance.ts    # Performance estimation calculations
│   │   └── shared/           # Shared calculation utilities
│   │       ├── quantization.ts
│   │       ├── kvcache.ts
│   │       └── constants.ts
│   ├── components/           # React UI components
│   │   ├── inputs/           # Input controls
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── GPUSelector.tsx
│   │   │   ├── QuantizationPicker.tsx
│   │   │   └── ParameterInputs.tsx
│   │   ├── outputs/          # Results display
│   │   │   ├── VRAMBreakdown.tsx
│   │   │   ├── PerformanceMetrics.tsx
│   │   │   └── MultiGPUVisualization.tsx
│   │   ├── common/           # Reusable UI components
│   │   │   ├── Slider.tsx
│   │   │   ├── Toggle.tsx
│   │   │   └── InfoTooltip.tsx
│   │   └── layout/           # Layout components
│   │       ├── Header.tsx
│   │       ├── Footer.tsx
│   │       └── Calculator.tsx
│   ├── data/                 # Static data sources
│   │   ├── models.json       # Model specifications
│   │   ├── gpus.json         # GPU specifications
│   │   └── schemas/          # TypeScript schemas for data
│   ├── store/                # State management
│   │   ├── useCalculatorStore.ts
│   │   ├── slices/           # Store slices by domain
│   │   │   ├── inputsSlice.ts
│   │   │   ├── resultsSlice.ts
│   │   │   └── configSlice.ts
│   │   └── middleware/
│   │       └── calculationMiddleware.ts
│   ├── workers/              # Web Workers for heavy calculations
│   │   ├── calculation.worker.ts
│   │   └── workerPool.ts
│   └── hooks/                # Custom React hooks
│       ├── useCalculation.ts
│       ├── useDebounce.ts
│       └── useWorker.ts
```

### Component Boundaries

| Component | Responsibility | Communicates With | Isolation Level |
|-----------|---------------|-------------------|-----------------|
| **Engines** | Pure calculation functions, no side effects | None directly (called by workers/hooks) | Complete (no dependencies) |
| **Workers** | Offload heavy calculations, manage computation | Engines (imports), Main thread (postMessage) | Isolated thread |
| **Store** | Single source of truth for app state | Components (via hooks), Middleware | Singleton |
| **Components** | UI rendering, user interaction | Store (read/write), Hooks | React tree |
| **Data** | Static reference data (models, GPUs) | Engines (import), Components (import) | Read-only |
| **Hooks** | Encapsulate stateful logic, worker communication | Store, Workers, Components | Per-component instance |

### Data Flow

```
User Input
    ↓
Components (inputs/)
    ↓
Store (state update)
    ↓
Middleware (triggers calculation)
    ↓
Worker (via useWorker hook)
    ↓
Engine (pure calculation)
    ↓
Worker (postMessage results)
    ↓
Store (results update)
    ↓
Components (outputs/)
    ↓
UI Update
```

**Key Flow Characteristics:**
- **Unidirectional:** Input → Store → Calculation → Results → Output
- **Async Calculation:** Workers prevent UI blocking
- **Reactive:** Store updates trigger component re-renders
- **Debounced:** Rapid inputs debounced before triggering calculations

### Data Models

#### Input State

```typescript
interface CalculatorInputs {
  // Model selection
  model: ModelSpec | null;
  customParams?: {
    paramCount: number;
    hiddenSize: number;
    numLayers: number;
    numHeads: number;
  };

  // Quantization
  quantization: 'fp32' | 'fp16' | 'int8' | 'int4' | 'nf4';

  // Inference parameters
  sequenceLength: number;
  batchSize: number;
  contextWindow: number;

  // Fine-tuning parameters (optional)
  fineTuning?: {
    enabled: boolean;
    method: 'full' | 'lora' | 'qlora';
    loraRank?: number;
    loraAlpha?: number;
  };

  // Multi-GPU parameters (optional)
  multiGPU?: {
    enabled: boolean;
    numGPUs: number;
    strategy: 'tensor-parallel' | 'pipeline-parallel' | 'hybrid';
  };

  // GPU selection
  gpu: GPUSpec | null;
}
```

#### Result State

```typescript
interface CalculationResults {
  timestamp: number;

  // VRAM breakdown
  vram: {
    model: number;           // Model weights
    kvCache: number;         // KV cache
    activations: number;     // Activation memory
    overhead: number;        // Framework overhead
    total: number;
  };

  // Fine-tuning additions (if enabled)
  fineTuning?: {
    optimizer: number;       // Optimizer states
    gradients: number;       // Gradient memory
    adapters: number;        // LoRA adapters
    total: number;
  };

  // Multi-GPU distribution (if enabled)
  multiGPU?: {
    perGPU: number;
    totalRequired: number;
    efficiency: number;      // Utilization %
  };

  // Performance estimation
  performance: {
    tokensPerSecond: number;
    computeBound: boolean;
    memoryBound: boolean;
    bottleneck: string;
  };

  // Validation
  fits: boolean;             // Does it fit in selected GPU?
  recommendation?: string;   // Suggested optimizations
}
```

#### Data Schemas

```typescript
// models.json
interface ModelSpec {
  id: string;
  name: string;
  family: 'llama' | 'mistral' | 'gpt' | 'other';
  params: number;            // Parameter count
  architecture: {
    hiddenSize: number;
    numLayers: number;
    numHeads: number;
    numKVHeads?: number;     // For GQA
    vocabSize: number;
  };
  contextWindow: number;
  tags: string[];
}

// gpus.json
interface GPUSpec {
  id: string;
  name: string;
  manufacturer: 'nvidia' | 'amd' | 'intel';
  vram: number;              // GB
  vramType: 'HBM2' | 'HBM3' | 'GDDR6' | 'GDDR6X';
  bandwidth: number;         // GB/s
  compute: {
    fp32: number;            // TFLOPS
    fp16: number;
    int8: number;
  };
  architecture: string;      // 'Ada Lovelace', 'RDNA 3', etc.
  releaseYear: number;
}
```

## Patterns to Follow

### Pattern 1: Engine Isolation
**What:** Calculation engines are pure functions with zero dependencies
**When:** All calculation logic
**Why:** Enables testing, worker usage, potential server-side execution
**Example:**
```typescript
// engines/inference.ts
export interface InferenceParams {
  paramCount: number;
  quantization: Quantization;
  sequenceLength: number;
  batchSize: number;
}

export interface InferenceResult {
  modelVRAM: number;
  kvCacheVRAM: number;
  activationVRAM: number;
  overheadVRAM: number;
  totalVRAM: number;
}

export function calculateInferenceVRAM(
  params: InferenceParams
): InferenceResult {
  // Pure calculation - no side effects, no external state
  const bytesPerParam = getQuantizationBytes(params.quantization);
  const modelVRAM = (params.paramCount * bytesPerParam) / (1024 ** 3);

  // ... other calculations

  return {
    modelVRAM,
    kvCacheVRAM,
    activationVRAM,
    overheadVRAM,
    totalVRAM: modelVRAM + kvCacheVRAM + activationVRAM + overheadVRAM,
  };
}
```

### Pattern 2: Worker Pool for Calculations
**What:** Maintain pool of Web Workers for parallel calculations
**When:** User changes inputs, triggering recalculation
**Why:** Prevents UI blocking, enables parallel computation for multi-scenario comparisons
**Example:**
```typescript
// workers/workerPool.ts
class CalculationWorkerPool {
  private workers: Worker[] = [];
  private queue: Task[] = [];

  constructor(size: number = navigator.hardwareConcurrency || 4) {
    for (let i = 0; i < size; i++) {
      this.workers.push(new Worker('./calculation.worker.ts'));
    }
  }

  async calculate(inputs: CalculatorInputs): Promise<CalculationResults> {
    const worker = this.getAvailableWorker();
    return new Promise((resolve, reject) => {
      worker.postMessage({ type: 'CALCULATE', inputs });
      worker.onmessage = (e) => {
        if (e.data.type === 'RESULT') {
          resolve(e.data.results);
        }
      };
    });
  }
}
```

### Pattern 3: Middleware-Triggered Calculations
**What:** Store middleware intercepts state changes and triggers calculations
**When:** Input state changes
**Why:** Separates UI concerns from calculation orchestration
**Example:**
```typescript
// store/middleware/calculationMiddleware.ts
export const calculationMiddleware = (store) => (next) => (action) => {
  const result = next(action);

  if (action.type === 'inputs/update') {
    const state = store.getState();

    // Debounce rapid changes
    clearTimeout(calculationTimer);
    calculationTimer = setTimeout(() => {
      workerPool.calculate(state.inputs).then((results) => {
        store.dispatch({ type: 'results/set', payload: results });
      });
    }, 300);
  }

  return result;
};
```

### Pattern 4: Slice-Based State Organization
**What:** Organize store into logical slices (inputs, results, config)
**When:** State management setup
**Why:** Scales with complexity, clear ownership, selective updates
**Example:**
```typescript
// store/slices/inputsSlice.ts
export const createInputsSlice = (set, get) => ({
  model: null,
  quantization: 'fp16',
  sequenceLength: 2048,
  batchSize: 1,

  setModel: (model) => set({ model }),
  setQuantization: (quant) => set({ quantization: quant }),
  setSequenceLength: (len) => set({ sequenceLength: len }),

  // Batch update to avoid multiple calculations
  updateInputs: (updates) => set(updates),
});
```

### Pattern 5: Smart Component Composition
**What:** Inputs, outputs, and common components are composable and isolated
**When:** Building UI
**Why:** Reusability, testability, clear data flow
**Example:**
```typescript
// components/layout/Calculator.tsx
export function Calculator() {
  return (
    <div className="calculator">
      <InputPanel>
        <ModelSelector />
        <GPUSelector />
        <QuantizationPicker />
        <ParameterInputs />
      </InputPanel>

      <ResultsPanel>
        <VRAMBreakdown />
        <PerformanceMetrics />
        <MultiGPUVisualization />
      </ResultsPanel>
    </div>
  );
}

// components/inputs/ModelSelector.tsx
export function ModelSelector() {
  const { model, setModel } = useCalculatorStore(
    (state) => ({ model: state.model, setModel: state.setModel })
  );

  return (
    <Select value={model?.id} onChange={(id) => setModel(findModel(id))}>
      {/* ... */}
    </Select>
  );
}
```

### Pattern 6: JSON Data with TypeScript Schemas
**What:** Static data in JSON, validated with TypeScript interfaces
**When:** Model and GPU data
**Why:** Easy to update, version control friendly, type-safe at boundaries
**Example:**
```typescript
// data/schemas/model.ts
import modelsData from '../models.json';

export const models: ModelSpec[] = modelsData;

export function findModel(id: string): ModelSpec | undefined {
  return models.find(m => m.id === id);
}

// Type checking happens at build time
const llama3 = findModel('llama-3-70b');
if (llama3) {
  const params = llama3.params; // TypeScript knows this is a number
}
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Calculations in Components
**What:** Performing VRAM calculations directly in React components
**Why bad:**
- UI blocks during heavy calculations
- Difficult to test calculation logic
- Duplicated logic across components
- Can't reuse calculations in workers or server-side
**Instead:** Always delegate to engines via workers/hooks

### Anti-Pattern 2: Direct Worker Access from Components
**What:** Components creating/managing their own workers
**Why bad:**
- Memory leaks (workers not cleaned up)
- Duplicate workers across components
- Complex lifecycle management
**Instead:** Use hook abstraction (useWorker, useCalculation)

### Anti-Pattern 3: Prop Drilling Configuration
**What:** Passing calculation parameters through multiple component layers
**Why bad:**
- Brittle component hierarchy
- Difficult to add new parameters
- Unnecessary re-renders
**Instead:** Use store/context for global calculator state

### Anti-Pattern 4: Mutable Data Schemas
**What:** Allowing runtime modification of models.json or gpus.json data
**Why bad:**
- Lost type safety
- Unpredictable state
- Difficult to debug
**Instead:** Treat data files as immutable constants, use store for user customizations

### Anti-Pattern 5: Monolithic Calculation Function
**What:** Single massive function handling all calculation types
**Why bad:**
- Difficult to test individual components
- Can't parallelize different calculation types
- Hard to maintain and extend
**Instead:** Separate engines per calculation domain, compose in worker

### Anti-Pattern 6: Synchronous Heavy Calculations
**What:** Calculating large model VRAM requirements on main thread
**Why bad:**
- UI freezes during calculation
- Poor user experience
- Blocks input handling
**Instead:** Always use Web Workers for calculations

## Scalability Considerations

| Concern | Current (Launch) | Future (100K+ users) | Enterprise |
|---------|------------------|----------------------|------------|
| **Calculation Speed** | Web Workers pool (4 workers) | WASM calculation engines | Server-side calculation API |
| **Data Size** | ~500 models, ~100 GPUs in JSON | IndexedDB cache, lazy load | Database backend, CDN |
| **State Management** | Zustand (simple store) | Same (Zustand scales well) | Server state sync (tRPC) |
| **Multi-Scenario** | Calculate sequentially | Parallel worker pool | Batch calculation API |
| **Offline Support** | Service Worker for static assets | Full offline mode with sync | N/A (enterprise likely online) |
| **Export/Share** | Local URL params | Backend for share links | User accounts, saved configs |

## Build Order Implications

### Phase 1: Foundation (Week 1-2)
**Dependencies:** None
**Components:**
- Data schemas (TypeScript interfaces)
- Static data files (initial models.json, gpus.json)
- Basic store setup (Zustand with inputs slice)
- Engine: inference.ts (core VRAM calculation)

**Rationale:** Establishes data contracts, enables parallel development

### Phase 2: Core UI (Week 2-3)
**Dependencies:** Phase 1 (schemas, store)
**Components:**
- Layout components (Calculator, Header)
- Basic input components (ModelSelector, GPUSelector)
- Basic output component (VRAMBreakdown)
- Hook: useCalculation (without workers initially)

**Rationale:** Vertical slice proves architecture, enables testing with synchronous calculations

### Phase 3: Worker Infrastructure (Week 3-4)
**Dependencies:** Phase 2 (working UI), Phase 1 (engines)
**Components:**
- calculation.worker.ts
- workerPool.ts
- Updated useCalculation hook (async worker-based)
- Middleware: calculationMiddleware

**Rationale:** Optimization layer, doesn't change UI contracts

### Phase 4: Advanced Calculations (Week 4-6)
**Dependencies:** Phase 3 (worker infrastructure)
**Components:**
- Engine: finetuning.ts
- Engine: multigpu.ts
- Engine: performance.ts
- Input components for advanced features
- Output components for advanced results

**Rationale:** Reuses infrastructure, additive features

### Phase 5: Polish & Optimization (Week 6-7)
**Dependencies:** Phase 4 (all features)
**Components:**
- Common components refinement
- Debouncing, caching optimizations
- Error boundaries, loading states
- Performance monitoring

**Rationale:** Quality layer on complete functionality

## Dependency Graph

```
Data Schemas (interfaces)
    ↓
    ├─→ Static Data (JSON files)
    ├─→ Store Slices
    └─→ Engine Functions
            ↓
            ├─→ Web Workers
            │       ↓
            │   Worker Pool
            │       ↓
            └─→ Hooks (useCalculation, useWorker)
                    ↓
                Components
                    ↓
                Application
```

**Critical Path:** Data Schemas → Engines → Workers → Hooks → Components

**Parallel Tracks:**
- Static Data can be built alongside Engines
- Input Components and Output Components can be built in parallel
- Common Components can be built anytime (no dependencies)

## Technology Integration Points

### TypeScript
- **Strict mode enabled** for data safety
- **Interfaces for all data models** (inputs, results, specs)
- **Type guards** for runtime validation of JSON data

### React
- **Functional components** with hooks (no class components)
- **Strict mode** for detecting side effects
- **Suspense boundaries** for async data loading

### State Management (Zustand)
- **Slice pattern** for organization
- **Middleware** for side effects (calculations)
- **Selectors** for derived state
- **DevTools** integration for debugging

### Web Workers
- **Typed messages** (TypeScript worker interfaces)
- **Pool management** for parallelism
- **Graceful fallback** if workers unavailable (synchronous calculation)

### Build System (Vite assumed)
- **Worker bundling** (Vite's native support)
- **JSON imports** with tree-shaking
- **TypeScript checking** pre-build
- **Code splitting** by route if multi-page

## Validation Strategy

### Data Validation
- **JSON Schema** validation for models.json and gpus.json at build time
- **TypeScript interfaces** enforce structure at compile time
- **Runtime guards** for user input bounds (e.g., sequenceLength > 0)

### Calculation Validation
- **Unit tests** for each engine function (pure functions = easy to test)
- **Known-good test cases** (e.g., Llama 3 70B fp16 = X GB)
- **Boundary testing** (minimum/maximum values)
- **Comparison tests** against reference implementations

### Component Validation
- **React Testing Library** for component behavior
- **Storybook** for visual regression testing
- **Integration tests** for full calculation flow

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Initial load | < 2s | Lighthouse |
| Calculation latency | < 100ms | Performance API |
| UI responsiveness | 60fps | React DevTools Profiler |
| Memory usage | < 100MB | Chrome DevTools Memory |
| Bundle size | < 200KB (gzipped) | Build output |

## Monitoring Points

- **Calculation time** per engine (identify slow functions)
- **Worker pool utilization** (are we CPU-bound?)
- **State update frequency** (excessive re-renders?)
- **Error rates** (calculation failures, invalid inputs)
- **User input patterns** (which features are most used?)

## Future Architecture Considerations

### Server-Side Calculation API
If calculator becomes multi-user or needs backend:
- **API**: POST /api/calculate with CalculatorInputs
- **Response**: CalculationResults (same interface)
- **Migration**: Replace worker.postMessage with fetch, minimal code changes

### Database Backend
If user accounts or saved configurations needed:
- **Schema**: Store CalculatorInputs + metadata (user, timestamp, name)
- **API**: CRUD for saved calculations
- **Sync**: Offline-first with sync when online

### Real-Time Collaboration
If multiple users need to view same calculation:
- **WebSocket** for state synchronization
- **CRDT** for conflict-free input merging
- **Presence** indicators for who's viewing

## Sources

**Confidence:** MEDIUM

This architecture is based on:
- Established React + TypeScript patterns (HIGH confidence from training data)
- Web Worker best practices for heavy computation (HIGH confidence)
- raidy's proven architecture pattern provided as baseline (HIGH confidence)
- State management patterns (Zustand, Redux) widely used in calculator apps (MEDIUM confidence)
- Browser calculator architecture patterns from training data (MEDIUM confidence)

**Limitations:**
- No direct access to raidy's codebase for detailed implementation patterns
- Could not verify current (2026) best practices via web search
- Recommendations based on training knowledge up to January 2025

**Recommended validation:**
- Review raidy's actual implementation for specific patterns
- Check Context7 for React, TypeScript, Zustand current API documentation
- Validate Web Worker patterns with current browser API documentation

