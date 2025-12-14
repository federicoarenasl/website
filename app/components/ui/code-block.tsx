'use client'

import React, { useState, useEffect } from 'react'
import { codeToHtml } from 'shiki'

/**
 * Props for the CodeBlock component
 */
interface CodeBlockProps {
  /** The code content to display */
  children: string
  /** Programming language for syntax highlighting */
  language?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * A code block component with syntax highlighting using Shiki
 * 
 * Features:
 * - Syntax highlighting using Shiki (VS Code's grammar system)
 * - Copy to clipboard functionality
 * - Line numbers
 * - Language label display
 * - Responsive design
 */
export function CodeBlock({ children, language, className, ...props }: CodeBlockProps) {
  // State to track if code has been copied to clipboard
  const [copied, setCopied] = useState(false)
  const [highlightedCode, setHighlightedCode] = useState<string>('')
  const [isDark, setIsDark] = useState(false)
  
  // Detect and track dark mode preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    setIsDark(mediaQuery.matches)
    
    const handler = (e: MediaQueryListEvent) => setIsDark(e.matches)
    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }, [])
  
  useEffect(() => {
    const highlightCode = async () => {
      try {
        // Normalize indentation: find minimum indentation and remove it from all lines
        const lines = children.split('\n')
        // Remove trailing empty lines
        while (lines.length > 0 && lines[lines.length - 1] === '') {
          lines.pop()
        }
        
        const nonEmptyLines = lines.filter(line => line.trim().length > 0)
        if (nonEmptyLines.length > 0) {
          const minIndent = Math.min(
            ...nonEmptyLines.map(line => {
              const match = line.match(/^(\s*)/)
              return match ? match[1].length : 0
            })
          )
          // Remove the common leading whitespace from all lines
          if (minIndent > 0) {
            for (let i = 0; i < lines.length; i++) {
              if (lines[i].length >= minIndent) {
                lines[i] = lines[i].substring(minIndent)
              }
            }
          }
        }
        
        const normalizedCode = lines.join('\n')
        
        const html = await codeToHtml(normalizedCode, {
          lang: language || 'text',
          theme: isDark ? 'github-dark-high-contrast' : 'github-light-high-contrast',
        })
        
        setHighlightedCode(html)
      } catch (error) {
        console.error('Error highlighting code:', error)
        // Fallback to plain text if highlighting fails
        setHighlightedCode(`<pre><code>${children}</code></pre>`)
      }
    }
    
    highlightCode()
  }, [children, language, isDark])
  
  /**
   * Copies the code content to the user's clipboard
   * Shows a temporary success state for 2 seconds
   */
  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(children)
      setCopied(true)
      // Reset copied state after 2 seconds
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy text: ', err)
    }
  }
  
  return (
    <div className="my-6 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Top bar with language and copy button */}
      <div className="flex items-center justify-between px-4 pt-2 pb-1">
        {/* Language label */}
        <span className="text-xs font-mono text-gray-600 dark:text-gray-400">
          {language || 'text'}
        </span>
        
        {/* Copy to clipboard button */}
        <button
          onClick={copyToClipboard}
          className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors duration-200"
          title="Copy to clipboard"
        >
          {copied ? (
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" strokeWidth="2"/>
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" strokeWidth="2"/>
            </svg>
          )}
        </button>
      </div>
      
      {/* Code content container */}
      <div 
        className="shiki-wrapper overflow-x-auto [&>pre]:!border-0 [&>pre]:!my-0 [&>pre]:!pt-2"
        dangerouslySetInnerHTML={{ __html: highlightedCode }}
      />
    </div>
  )
}

/**
 * Props for the InlineCode component
 */
interface InlineCodeProps {
  /** The code content to display inline */
  children: string
  /** Additional CSS classes */
  className?: string
}

/**
 * An inline code component for displaying small code snippets within text
 * 
 * Features:
 * - Styled background with rounded corners
 * - Monospace font
 * - Dark mode support
 */
export function InlineCode({ children, className, ...props }: InlineCodeProps) {
  return (
    <code 
      className={`bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono ${className || ''}`}
      {...props}
    >
      {children}
    </code>
  )
}
