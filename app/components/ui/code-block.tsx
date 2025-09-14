'use client'

import React, { useState } from 'react'
import { highlight } from 'sugar-high'

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
 * A code block component with syntax highlighting, copy functionality, and comment styling
 * 
 * Features:
 * - Syntax highlighting using sugar-high
 * - Copy to clipboard functionality
 * - Special styling for comments (reduced opacity)
 * - Language label display
 * - Responsive design
 */
export function CodeBlock({ children, language, className, ...props }: CodeBlockProps) {
  // State to track if code has been copied to clipboard
  const [copied, setCopied] = useState(false)
  
  /**
   * Processes code content to apply special styling to comments and line numbers
   * @param code - The code content to process
   * @returns An array of React elements with the processed code and line numbers
   */
  const processCodeWithComments = (code: string) => {
    const lines = code.split('\n')
    // Remove trailing empty lines that result from trailing newlines
    while (lines.length > 0 && lines[lines.length - 1] === '') {
      lines.pop()
    }
    const maxLineNumber = lines.length
    const lineNumberWidth = maxLineNumber.toString().length
    
    return lines.map((line, index) => {
      const lineNumber = index + 1
      const trimmedLine = line.trim()
      
      // Check if line is a comment (supports //, #, <!-- -->, /* */)
      const isComment = trimmedLine.startsWith('//') || 
                       trimmedLine.startsWith('#') ||
                       trimmedLine.startsWith('<!--') ||
                       trimmedLine.startsWith('/*') ||
                       (trimmedLine.startsWith('*') && !trimmedLine.startsWith('*/'))
      
      // Render line with line number
      return (
        <div key={index} className="flex items-start leading-relaxed">
          {/* Line number */}
          <span 
            className="text-gray-400 dark:text-gray-500 text-sm font-mono select-none pr-4 text-right opacity-70 flex-shrink-0"
            style={{ minWidth: `${lineNumberWidth + 1}ch`, lineHeight: 'inherit' }}
          >
            {lineNumber}
          </span>
          {/* Code content */}
          <span className={`flex-1 ${isComment ? 'opacity-60' : ''}`} style={{ lineHeight: 'inherit' }}>
            {isComment ? (
              line
            ) : (
              <span dangerouslySetInnerHTML={{ __html: highlight(line) }} />
            )}
          </span>
        </div>
      )
    })
  }
  
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
    <div className="relative white-code-block">
      {/* Header with language label and copy button */}
      <div className="absolute top-2 right-2 flex items-center gap-2 rounded z-10">
        {/* Language label - only shown if language is provided */}
        {language && (
          <div className="text-xs text-gray-600 dark:text-gray-300 font-mono bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">
            {language}
          </div>
        )}
        {/* Copy to clipboard button with visual feedback */}
        <button
          onClick={copyToClipboard}
          className="text-gray-600 dark:text-gray-300 hover:opacity-70 p-1 transition-opacity duration-200"
          title="Copy to clipboard"
        >
          {/* Show checkmark icon when copied, clipboard icon otherwise */}
          {copied ? (
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          ) : (
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" strokeWidth="2"/>
              <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" strokeWidth="2"/>
            </svg>
          )}
        </button>
      </div>
      {/* Code content container */}
      <pre className="p-4 overflow-x-auto leading-relaxed !bg-white dark:!bg-black !border-gray-200 dark:!border-gray-700 rounded-lg">
        <code 
          className={`text-sm font-mono block text-gray-900 dark:text-gray-100 ${className || ''}`}
          {...props}
        >
          {processCodeWithComments(children)}
        </code>
      </pre>
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
