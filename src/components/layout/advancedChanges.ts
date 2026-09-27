import { DEFAULT_UI_CONFIG, type UIConfig } from '@store/uiStore'

/**
 * Advanced settings that differ from their default. The Advanced section opens
 * itself when this is above 0, so a collapsed section never hides an active
 * non-default setting (ADR 0005). Inputs hidden as inert in training don't count.
 */
export function countAdvancedChanges(config: UIConfig): number {
  const d = DEFAULT_UI_CONFIG
  if (config.mode === 'training') return config.batchSize !== d.batchSize ? 1 : 0
  return [
    config.batchSize !== d.batchSize,
    config.kvQuantization !== d.kvQuantization,
    config.shardingStrategy !== d.shardingStrategy,
    config.numNodes > 1 &&
      (config.interNodeFabric !== d.interNodeFabric || config.customFabric !== d.customFabric),
    config.interconnectOverride !== d.interconnectOverride,
    config.offloadingEnabled !== d.offloadingEnabled,
    config.kvTier.tier !== d.kvTier.tier,
  ].filter(Boolean).length
}
