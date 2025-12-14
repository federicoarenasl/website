import Link from 'next/link'
import Image from 'next/image'
import { MDXRemote } from 'next-mdx-remote/rsc'
import React from 'react'
import { CodeBlock, InlineCode } from '../ui/code-block'
import { FootnoteProvider, FootnoteReference, FootnoteDefinition, FootnoteList, parseFootnotes } from './footnotes'
import { TextWithFootnotes, ProcessedParagraph, extractFootnotesFromContent } from './footnote-utils'

/**
 * Process MDX content to convert ^[1] references to FootnoteReference components
 */
function processMDXFootnotes(content: string, footnotes: Array<{ id: string; content: string }>) {
  // Create a map of footnote IDs to content for quick lookup
  const footnoteMap = new Map(footnotes.map(fn => [fn.id, fn.content]))
  
  // Track which footnotes we've already processed to avoid duplication
  const processedFootnotes = new Set<string>()
  
  // Replace ^[1] patterns with FootnoteReference components
  let processedContent = content.replace(/\^\[(\d+)\]/g, (match, id) => {
    const footnoteContent = footnoteMap.get(id)
    if (footnoteContent) {
      // Only pass content to the first occurrence of each footnote
      // This ensures that each footnote is only defined once
      if (!processedFootnotes.has(id)) {
        processedFootnotes.add(id)
        return `<FootnoteReference id="${id}">${footnoteContent}</FootnoteReference>`
      } else {
        // For subsequent references, don't pass content to avoid duplication
        return `<FootnoteReference id="${id}"></FootnoteReference>`
      }
    }
    return match // Keep original if no footnote found
  })
  
  return processedContent
}

/**
 * Table component for rendering structured data in MDX content
 * @param {Object} data - Table data object containing headers and rows
 * @param {string[]} data.headers - Array of header strings
 * @param {string[][]} data.rows - Array of row arrays, where each row contains cell strings
 * @returns {JSX.Element} Rendered table element
 */
function Table({ data }) {
  // Map headers to table header elements
  let headers = data.headers.map((header, index) => (
    <th key={index}>{header}</th>
  ))
  
  // Map rows to table row elements, with each cell mapped to a table data element
  let rows = data.rows.map((row, index) => (
    <tr key={index}>
      {row.map((cell, cellIndex) => (
        <td key={cellIndex}>{cell}</td>
      ))}
    </tr>
  ))

  return (
    <table>
      <thead>
        <tr>{headers}</tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  )
}

/**
 * Custom link component that handles different types of links appropriately
 * - Internal links (starting with '/') use Next.js Link component
 * - Hash links (starting with '#') use regular anchor tags
 * - External links open in new tab with security attributes
 * @param {Object} props - Link props including href and children
 * @returns {JSX.Element} Appropriate link element based on href type
 */
function CustomLink(props) {
  let href = props.href

  // Internal links - use Next.js Link for client-side navigation
  if (href.startsWith('/')) {
    return (
      <Link href={href} {...props}>
        {props.children}
      </Link>
    )
  }

  // Hash links - use regular anchor tag for same-page navigation
  if (href.startsWith('#')) {
    return <a {...props} />
  }

  // External links - open in new tab with security attributes
  return <a target="_blank" rel="noopener noreferrer" {...props} />
}

/**
 * Image component with rounded corners, border, and optional caption
 * Centers the image and displays the alt text as a caption below
 * Uses object-fit: contain to prevent cropping when aspect ratios don't match
 * @param {Object} props - Image props including alt, src, width, height, etc.
 * @returns {JSX.Element} Styled image container with optional caption
 */
function RoundedImage(props) {
  const { width, height, alt, ...restProps } = props
  
  // If width and height are provided, use a container with aspect ratio
  // Otherwise, use responsive sizing
  if (width && height) {
    return (
      <div className="flex flex-col items-center my-6">
        <div 
          className="relative rounded-lg border border-gray-300 dark:border-gray-600 overflow-hidden"
          style={{ 
            width: '100%', 
            maxWidth: `${width}px`,
            aspectRatio: `${width} / ${height}`
          }}
        >
          <Image 
            alt={alt || ''} 
            fill
            style={{ objectFit: 'contain' }}
            sizes={`(max-width: ${width}px) 100vw, ${width}px`}
            {...restProps} 
          />
        </div>
        {/* Display alt text as caption if provided */}
        {alt && (
          <p className="text-sm italic text-gray-600 dark:text-gray-400 mt-2 text-center">
            {alt}
          </p>
        )}
      </div>
    )
  }
  
  // Fallback for images without explicit dimensions
  return (
    <div className="flex flex-col items-center my-6">
      <Image 
        alt={alt || ''} 
        className="rounded-lg border border-gray-300 dark:border-gray-600" 
        width={width}
        height={height}
        {...restProps} 
      />
      {/* Display alt text as caption if provided */}
      {alt && (
        <p className="text-sm italic text-gray-600 dark:text-gray-400 mt-2 text-center">
          {alt}
        </p>
      )}
    </div>
  )
}

/**
 * Pre-formatted text component that renders code blocks with syntax highlighting
 * Extracts language information from className and passes content to CodeBlock component
 * @param {Object} props - Pre element props
 * @param {React.ReactNode} props.children - Code content, either string or React element
 * @param {string} [props.className] - CSS class that may contain language info (e.g., "language-javascript")
 * @returns {JSX.Element} CodeBlock component with extracted language and content
 */
function Pre({ children, ...props }) {
  // Extract language from className if present (e.g., "language-javascript" -> "javascript")
  // MDX puts the language on the code element, not the pre element
  const language = (children?.props?.className || props.className || '')
    .replace('language-', '')
    .replace(/^lang-/, '')
  
  // Extract code content - handle both string children and code element children
  const codeContent = typeof children === 'string' 
    ? children 
    : children?.props?.children || ''
  
  return <CodeBlock language={language}>{codeContent}</CodeBlock>
}

/**
 * Converts a string to a URL-friendly slug
 * @param {string} str - The string to slugify
 * @returns {string} URL-friendly slug with lowercase letters, hyphens, and no special characters
 */
function slugify(str) {
  return str
    .toString()
    .toLowerCase()
    .trim() // Remove whitespace from both ends of a string
    .replace(/\s+/g, '-') // Replace spaces with -
    .replace(/&/g, '-and-') // Replace & with 'and'
    .replace(/[^\w\-]+/g, '') // Remove all non-word characters except for -
    .replace(/\-\-+/g, '-') // Replace multiple - with single -
}

/**
 * Custom blockquote component that supports special styling for notes and warnings
 * Detects if content starts with "NOTE:", "Warning:", etc. and applies appropriate styling
 * @param {Object} props - Blockquote props including children
 * @returns {JSX.Element} Styled blockquote element
 */
function CustomBlockquote({ children, ...props }) {
  // Extract text content from children to detect quote type
  const getTextContent = (node: React.ReactNode): string => {
    if (typeof node === 'string') return node
    if (React.isValidElement(node) && node.props.children) {
      return getTextContent(node.props.children)
    }
    if (Array.isArray(node)) {
      return node.map(getTextContent).join('')
    }
    return ''
  }

  const textContent = getTextContent(children).trim()
  const lowerContent = textContent.toLowerCase()

  // Detect quote type based on content
  let quoteType = 'default'
  if (lowerContent.startsWith('note:') || lowerContent.startsWith('note ')) {
    quoteType = 'note'
  } else if (lowerContent.startsWith('warning:') || lowerContent.startsWith('warning ')) {
    quoteType = 'warning'
  } else if (lowerContent.startsWith('tip:') || lowerContent.startsWith('tip ')) {
    quoteType = 'tip'
  }

  return (
    <blockquote className={`custom-blockquote ${quoteType}`} {...props}>
      {children}
    </blockquote>
  )
}

/**
 * Factory function that creates heading components with anchor links
 * Each heading gets an id based on its content and includes an anchor link for deep linking
 * @param {number} level - Heading level (1-6)
 * @returns {React.Component} Heading component with anchor link functionality
 */
function createHeading(level) {
  const Heading = ({ children }) => {
    // Generate URL-friendly slug from heading content
    let slug = slugify(children)
    
    // Create heading element with id and anchor link
    return React.createElement(
      `h${level}`,
      { id: slug },
      [
        // Create anchor link for deep linking to this heading
        React.createElement('a', {
          href: `#${slug}`,
          key: `link-${slug}`,
          className: 'anchor',
        }),
      ],
      children
    )
  }

  // Set display name for debugging purposes
  Heading.displayName = `Heading${level}`

  return Heading
}

/**
 * Component mapping object for MDX rendering
 * Maps HTML elements and custom components to their React counterparts
 */
let components = {
  h1: createHeading(1),
  h2: createHeading(2),
  h3: createHeading(3),
  h4: createHeading(4),
  h5: createHeading(5),
  h6: createHeading(6),
  Image: RoundedImage,
  a: CustomLink,
  code: InlineCode,
  pre: Pre,
  blockquote: CustomBlockquote,
  Table,
  FootnoteReference,
  FootnoteDefinition,
  TextWithFootnotes,
  ProcessedParagraph,
}

/**
 * Custom MDX component wrapper that provides enhanced rendering capabilities
 * Combines default components with any additional components passed via props
 * Now includes automatic footnote processing for ^[1] syntax
 * @param {Object} props - MDX props including source content and optional custom components
 * @param {Object} [props.components] - Additional custom components to merge with defaults
 * @param {boolean} [props.autoProcessFootnotes=true] - Whether to automatically process footnote references
 * @returns {JSX.Element} Rendered MDX content with custom components and footnotes
 */
export function CustomMDX(props) {
  // If auto-processing is enabled, extract footnotes from the content
  if (props.autoProcessFootnotes !== false) {
    const { cleanedContent, footnotes } = extractFootnotesFromContent(props.source)
    
    // Process the content to convert ^[1] references to FootnoteReference components
    const processedContent = processMDXFootnotes(cleanedContent, footnotes)

    // Enhanced components that include footnote processing
    const enhancedComponents = {
      ...components,
      ...(props.components || {}),
      FootnoteReference,
      FootnoteDefinition,
      TextWithFootnotes,
      ProcessedParagraph,
    }

    return (
      <FootnoteProvider>
        <MDXRemote
          source={processedContent}
          components={enhancedComponents}
        />
        <FootnoteList />
      </FootnoteProvider>
    )
  }

  // Standard processing without automatic footnote extraction
  return (
    <FootnoteProvider>
      <MDXRemote
        {...props}
        components={{ ...components, ...(props.components || {}) }}
      />
      <FootnoteList />
    </FootnoteProvider>
  )
}

/**
 * Legacy CustomMDX component for backward compatibility
 * Use this if you want to disable automatic footnote processing
 * @param {Object} props - MDX props including source content and optional custom components
 * @returns {JSX.Element} Rendered MDX content with custom components
 */
export function CustomMDXLegacy(props) {
  return (
    <FootnoteProvider>
      <MDXRemote
        {...props}
        components={{ ...components, ...(props.components || {}) }}
      />
      <FootnoteList />
    </FootnoteProvider>
  )
}

/**
 * Utility function to create footnote definitions from a simple object
 * Useful for programmatically creating footnotes
 */
export function createFootnotes(footnoteMap: Record<string, string>) {
  return Object.entries(footnoteMap).map(([id, content]) => ({
    id,
    content
  }))
}

/**
 * Component for rendering a single footnote reference
 * Usage: <Footnote id="1" content="This is a footnote" />
 */
export function Footnote({ id, content }: { id: string; content: string }) {
  return <FootnoteReference id={id}>{content}</FootnoteReference>
}
