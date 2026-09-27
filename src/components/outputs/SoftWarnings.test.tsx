import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SoftWarnings } from './SoftWarnings'

describe('SoftWarnings', () => {
  it('renders each warning with its id', () => {
    render(
      <SoftWarnings
        warnings={[{ id: 'W8', message: "256 experts don't split evenly across 6 GPUs." }]}
      />,
    )
    expect(screen.getByTestId('soft-warning-W8')).toHaveTextContent(
      "256 experts don't split evenly across 6 GPUs.",
    )
  })

  it('renders nothing without warnings', () => {
    const { container } = render(<SoftWarnings warnings={[]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
