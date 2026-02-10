import { Popover, PopoverButton, PopoverPanel } from '@headlessui/react'
import { InformationCircleIcon } from '@heroicons/react/20/solid'

interface InfoTipProps {
  text: string
  className?: string
}

export function InfoTip({ text, className = '' }: InfoTipProps) {
  return (
    <Popover className={`relative inline-flex ${className}`}>
      <PopoverButton className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full">
        <InformationCircleIcon className="h-4 w-4" aria-hidden="true" />
        <span className="sr-only">More info</span>
      </PopoverButton>
      <PopoverPanel
        anchor="bottom"
        className="z-50 w-72 rounded-lg bg-white dark:bg-gray-800 p-3 text-sm text-gray-700 dark:text-gray-300 shadow-lg ring-1 ring-gray-200 dark:ring-gray-700 [--anchor-gap:4px]"
      >
        {text}
      </PopoverPanel>
    </Popover>
  )
}
