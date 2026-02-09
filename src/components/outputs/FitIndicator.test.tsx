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
})
