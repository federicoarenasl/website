'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'

export function PdfLink() {
  return (
    <Link
      href="/cv.pdf"
      target="_blank"
      className="print-hide p-1 text-neutral-400 hover:text-neutral-600 dark:text-neutral-500 dark:hover:text-neutral-300 transition-colors flex items-center justify-center"
      aria-label="Open CV as PDF"
      title="Open CV as PDF"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M7 7h10v10" />
        <path d="M7 17L17 7" />
      </svg>
    </Link>
  )
}

export function PrintWatermark() {
  const [date, setDate] = useState('')

  useEffect(() => {
    setDate(new Date().toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    }))
  }, [])

  return (
    <div className="print-watermark">
      Printed on: {date}
    </div>
  )
}

