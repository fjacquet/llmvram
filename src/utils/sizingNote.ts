/** vLLM's default max_num_seqs; a larger decode batch needs the flag raised */
export const VLLM_DEFAULT_MAX_NUM_SEQS = 256

/** Soft note for a decode batch above vLLM's default; null when the default suffices */
export function decodeBatchNote(decodeBatch: number): string | null {
  return decodeBatch > VLLM_DEFAULT_MAX_NUM_SEQS
    ? `${decodeBatch} sequences decode at once per replica: needs vLLM \`--max-num-seqs\` ≥ ${decodeBatch} (default is lower); per-user speed beyond 128 rests on 1 measured point`
    : null
}

/** Counts come from memory alone; the speed is shown, never used to size */
export const SIZING_SPEED_NOTE =
  'Counts are set by memory (90% vLLM limit); the speed shown is expected, with the theoretical ceiling beside it.'
