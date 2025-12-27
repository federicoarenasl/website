'use client'

import { useEffect, useRef } from 'react'

export function Giscus() {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!ref.current || ref.current.hasChildNodes()) return

    const script = document.createElement('script')
    script.src = 'https://giscus.app/client.js'
    script.async = true
    script.crossOrigin = 'anonymous'
    script.setAttribute('data-repo', 'federicoarenasl/website')
    script.setAttribute('data-repo-id', '') // Get from giscus.app
    script.setAttribute('data-category', 'Announcements')
    script.setAttribute('data-category-id', '') // Get from giscus.app
    script.setAttribute('data-mapping', 'pathname')
    script.setAttribute('data-strict', '0')
    script.setAttribute('data-reactions-enabled', '1')
    script.setAttribute('data-emit-metadata', '0')
    script.setAttribute('data-input-position', 'bottom')
    script.setAttribute('data-theme', 'preferred_color_scheme')
    script.setAttribute('data-lang', 'en')

    ref.current.appendChild(script)
  }, [])

  return (
    <div className="mt-16">
      <hr className="border-neutral-200 dark:border-neutral-800 mb-12" />
      <div ref={ref} />
    </div>
  )
}

