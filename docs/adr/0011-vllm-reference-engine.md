# 0011. vLLM is the reference engine

**Status:** Accepted, 2026-11-11

## Context

Every figure the calculator shows depends on how an inference engine allocates
memory and runs a step. Engines differ: KV layout, memory utilisation, which
tensor-parallel degrees are accepted, how MoE and sparse attention are
executed. A customer who runs another engine will not get the same numbers.

## Decision

vLLM is the reference engine, in every mode. The user confirmed this when
ZML was raised as a question (2026-11).

- **Memory:** fits use vLLM's default `gpu_memory_utilization` of 0.9. TP shards
  embeddings and the LM head, replicates only norms, and splits KV by
  `max(1, kv_heads // tp)`. MLA KV is duplicated on every TP rank.
- **KV layout:** KV sizes are what vLLM allocates (cache specs in the model
  classes, for example `fp8_ds_mla` at 656 B per token per layer for DSA models),
  confirmed by a second source.
- **Parallelism:** only tensor-parallel degrees vLLM accepts are offered; Sizing
  uses tensor parallel inside a server or rack and pipeline stages across servers.
- **Speed:** expected speed is the roofline divided by efficiencies measured on
  published vLLM runs (ADR 0009, ADR 0010).
- **Roster:** a model joins only if vLLM has a model class for it and its KV
  layout can be sourced twice. Otherwise it is skipped and reported.
  Mistral Large 4 is the exception: it entered pending upstream vLLM support (PR #64534). ERNIE 5.5 Flash Base, LongCat 2.5 and GLM 5.3 Flash were skipped under this rule.

## Consequences

- Figures for TensorRT-LLM, SGLang, llama.cpp or other engines are indicative
  only. The UI and the Guide say so.
- A vLLM release that changes a layout (for example the FP8 DSA cache) is a data
  change here, pinned with a dated reason in the tests.
- Where vLLM and a published figure disagree, the larger memory figure wins
  (the safe direction).

## Sources

- vLLM documentation and source: <https://github.com/vllm-project/vllm>
- ADR 0009 and ADR 0010 for the calibration and Sizing details.
