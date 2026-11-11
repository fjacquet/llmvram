import { render, screen } from '@testing-library/react'
import Decimal from 'decimal.js'
import { describe, expect, it } from 'vitest'
import { FitIndicator } from './FitIndicator'

describe('FitIndicator', () => {
  it('should render "Fits Comfortably" when usage <= 80%', () => {
    // 40 GB used out of 80 GB = 50%
    render(<FitIndicator totalVRAM={new Decimal(40)} availableVRAM={80} />)
    expect(screen.getByText('Fits Comfortably')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('should render "Tight Fit" when usage is 81-95%', () => {
    // 72 GB used out of 80 GB = 90%
    render(<FitIndicator totalVRAM={new Decimal(72)} availableVRAM={80} />)
    expect(screen.getByText('Tight Fit')).toBeInTheDocument()
    expect(screen.getByText('90%')).toBeInTheDocument()
  })

  it('should render "Does Not Fit" when usage > 95%', () => {
    // 100 GB used out of 80 GB = 125%
    render(<FitIndicator totalVRAM={new Decimal(100)} availableVRAM={80} />)
    expect(screen.getByText('Does Not Fit')).toBeInTheDocument()
    expect(screen.getByText('125%')).toBeInTheDocument()
  })

  it('should display correct GB usage text', () => {
    render(<FitIndicator totalVRAM={new Decimal(42.5)} availableVRAM={80} />)
    expect(screen.getByText('Using 42.50 GB of 80 GB')).toBeInTheDocument()
  })

  it('should render at the 80% boundary as "Fits Comfortably"', () => {
    // Exactly 80%
    render(<FitIndicator totalVRAM={new Decimal(64)} availableVRAM={80} />)
    expect(screen.getByText('Fits Comfortably')).toBeInTheDocument()
    expect(screen.getByText('80%')).toBeInTheDocument()
  })

  it('inference limit of 90 reads "Does Not Fit" at 92%, training keeps "Tight Fit"', () => {
    const { unmount } = render(
      <FitIndicator totalVRAM={new Decimal(73.6)} availableVRAM={80} limitPercent={90} />,
    )
    expect(screen.getByText('Does Not Fit')).toBeInTheDocument()
    unmount()
    render(<FitIndicator totalVRAM={new Decimal(73.6)} availableVRAM={80} />)
    expect(screen.getByText('Tight Fit')).toBeInTheDocument()
  })

  it('inference decides fit with fitsInference: exactly 90% is a tight fit, above is not (M5)', () => {
    // 2026-11-11, final review M5: 0.9 * 100 is 90.00000000000001 in JS; the decision is now
    // fitsInference, the same rule as the verdict. No RED was observable: Decimal gives exactly 90.
    const { unmount } = render(
      <FitIndicator totalVRAM={new Decimal(72)} availableVRAM={80} limitPercent={90} />,
    )
    expect(screen.getByText('Tight Fit')).toBeInTheDocument()
    unmount()
    render(<FitIndicator totalVRAM={new Decimal(72.01)} availableVRAM={80} limitPercent={90} />)
    expect(screen.getByText('Does Not Fit')).toBeInTheDocument()
  })

  it('draws the 90% marker in inference only (I2)', () => {
    // 2026-11-11, final review I2: spec P4, the 90% line marked where a bar exists
    const { unmount } = render(
      <FitIndicator totalVRAM={new Decimal(40)} availableVRAM={80} limitPercent={90} />,
    )
    expect(screen.getByTestId('limit-marker').style.left).toBe('90%')
    unmount()
    render(<FitIndicator totalVRAM={new Decimal(40)} availableVRAM={80} />)
    expect(screen.queryByTestId('limit-marker')).not.toBeInTheDocument()
  })

  it('training between 90% and 100% of VRAM is unchanged: Tight Fit, 95%, no marker', () => {
    // 2026-11-11, final review training guard: training keeps total <= vram, tiers 80/95
    render(<FitIndicator totalVRAM={new Decimal(76)} availableVRAM={80} />)
    expect(screen.getByText('Tight Fit')).toBeInTheDocument()
    expect(screen.getByText('95%')).toBeInTheDocument()
    expect(screen.getByText('Using 76.00 GB of 80 GB')).toBeInTheDocument()
    expect(screen.queryByTestId('limit-marker')).not.toBeInTheDocument()
  })

  it('states the machine total across the GPUs', () => {
    render(<FitIndicator totalVRAM={new Decimal(150)} availableVRAM={262.5} numGPUs={8} />)
    expect(screen.getByText('total 1.2 TB of 2.1 TB across 8 GPUs')).toBeInTheDocument()
    expect(screen.getByText(/Using 150.00 GB per GPU of 262.5 GB \(8 GPUs\)/)).toBeInTheDocument()
  })

  it('shows no machine total for a single GPU', () => {
    render(<FitIndicator totalVRAM={new Decimal(40)} availableVRAM={80} numGPUs={1} />)
    expect(screen.queryByTestId('fit-machine-total')).toBeNull()
  })
})
