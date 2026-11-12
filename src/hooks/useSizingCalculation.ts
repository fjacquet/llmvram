import type { SizingInput, SizingResult } from '@engines/types'
import { useUIStore } from '@store/uiStore'
import { useEffect, useState } from 'react'
import { reconstructSizingResult } from '../workers/sizing-serialize'

export interface UseSizingCalculationResult {
  result: SizingResult | null
  isCalculating: boolean
  error: string | null
}

/** Sizes the deployment in the calculation worker (synchronously when Workers are unavailable) */
export function useSizingCalculation(): UseSizingCalculationResult {
  const model = useUIStore((s) => s.selectedModel)
  const gpu = useUIStore((s) => s.selectedGPU)
  const quantization = useUIStore((s) => s.quantization)
  const kvQuantization = useUIStore((s) => s.kvQuantization)
  const sequenceLength = useUIStore((s) => s.sequenceLength)
  const users = useUIStore((s) => s.concurrentUsers)
  const interNodeFabric = useUIStore((s) => s.interNodeFabric)
  const customFabric = useUIStore((s) => s.customFabric)
  const kvTier = useUIStore((s) => s.kvTier)

  const [result, setResult] = useState<SizingResult | null>(null)
  const [isCalculating, setIsCalculating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!model || !gpu) {
      setResult(null)
      setIsCalculating(false)
      setError(null)
      return
    }
    setIsCalculating(true)
    setError(null)
    const payload = {
      model,
      gpu,
      quantization,
      kvQuantization,
      sequenceLength,
      users,
      interNodeFabric,
      customFabric,
      kvTier,
    }

    if (typeof Worker !== 'undefined') {
      const worker = new Worker(new URL('../workers/calculation.worker.ts', import.meta.url), {
        type: 'module',
      })
      worker.onmessage = (event) => {
        if (event.data.type === 'SIZING_RESULT') {
          setResult(reconstructSizingResult(event.data.result))
        } else if (event.data.type === 'CALCULATION_ERROR') {
          setResult(null)
          setError(event.data.error)
        }
        setIsCalculating(false)
        worker.terminate()
      }
      worker.onerror = (event) => {
        setResult(null)
        setError(`Worker error: ${event.message}`)
        setIsCalculating(false)
        worker.terminate()
      }
      worker.postMessage({ type: 'CALCULATE_SIZING', payload })
      return () => worker.terminate()
    }

    let stale = false
    Promise.all([
      import('@engines/sizing'),
      import('@engines/fabric'),
      import('@engines/context-mix'),
    ])
      .then(([sizing, fabric, { defaultMix }]) => {
        const { interNodeFabric: f, customFabric: c, ...rest } = payload
        const input: SizingInput = {
          ...rest,
          fabric: fabric.resolveFabricSpec(f, c),
          contextMix: defaultMix(rest.sequenceLength),
          ttftTargetSeconds: null,
        }
        if (!stale) setResult(sizing.sizeDeployment(input))
      })
      .catch((err) => !stale && setError(err instanceof Error ? err.message : String(err)))
      .finally(() => !stale && setIsCalculating(false))
    return () => {
      stale = true
    }
  }, [
    model,
    gpu,
    quantization,
    kvQuantization,
    sequenceLength,
    users,
    interNodeFabric,
    customFabric,
    kvTier,
  ])

  return { result, isCalculating, error }
}
