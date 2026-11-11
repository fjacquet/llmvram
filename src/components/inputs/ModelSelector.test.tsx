import modelsData from '@data/models.json'
import { fireEvent, render, screen } from '@testing-library/react'
import { validateModels } from '@utils/schemas'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createCustomModel } from '@/types/model'

// The real uiStore's `persist` middleware throws in jsdom: use a plain store.
const { useUIStore } = vi.hoisted(() => {
  const { create } = require('zustand') as typeof import('zustand')
  const useUIStore = create<Record<string, unknown>>(() => ({
    selectedModel: null,
    setSelectedModel: () => {},
  }))
  return { useUIStore }
})

vi.mock('@store/uiStore', () => ({ useUIStore }))

import { ModelSelector } from './ModelSelector'

function openOptions(): HTMLElement[] {
  const toggle = screen.getAllByRole('button').find((b) => b.hasAttribute('aria-haspopup'))
  if (!toggle) throw new Error('combobox button not found')
  fireEvent.click(toggle)
  return screen.getAllByRole('option')
}

describe('ModelSelector publisher country', () => {
  beforeEach(() => useUIStore.setState({ selectedModel: null }))

  it('shows the code with the country name on hover', () => {
    render(<ModelSelector />)
    const option = openOptions().find((o) => o.textContent?.includes('Mistral Large 3'))
    const badge = option?.querySelector('[data-testid="origin-country"]')
    expect(badge?.textContent).toBe('FR')
    expect(badge?.getAttribute('title')).toBe('France')
  })

  it('shows no code for a custom model', () => {
    const custom = createCustomModel({ name: 'Mine', num_parameters_billion: 7 })
    useUIStore.setState({ selectedModel: custom })
    const { container } = render(<ModelSelector />)
    expect(container.querySelector('[data-testid="origin-country"]')).toBeNull()
  })

  it('shows the code beside the selected model', () => {
    const model = validateModels(modelsData).find((m) => m.origin_country === 'FR')
    useUIStore.setState({ selectedModel: model })
    const { container } = render(<ModelSelector />)
    expect(container.querySelector('[data-testid="origin-country"]')?.textContent).toBe('FR')
  })
})
