import Link from 'next/link'
import Image from 'next/image'
import { MDXRemote } from 'next-mdx-remote/rsc'
import { highlight } from 'sugar-high'
import React from 'react'
import { CodeBlock, InlineCode } from './ui/code-block'

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
 * @param {Object} props - Image props including alt, src, width, height, etc.
 * @returns {JSX.Element} Styled image container with optional caption
 */
function RoundedImage(props) {
  return (
    <div className="flex flex-col items-center my-6">
      <Image 
        alt={props.alt} 
        className="rounded-lg border border-gray-300 dark:border-gray-600" 
        {...props} 
      />
      {/* Display alt text as caption if provided */}
      {props.alt && (
        <p className="text-sm italic text-gray-600 dark:text-gray-400 mt-2 text-center">
          {props.alt}
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
  const language = props.className?.replace('language-', '') || ''
  
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
  Table,
}

/**
 * Custom MDX component wrapper that provides enhanced rendering capabilities
 * Combines default components with any additional components passed via props
 * @param {Object} props - MDX props including source content and optional custom components
 * @param {Object} [props.components] - Additional custom components to merge with defaults
 * @returns {JSX.Element} Rendered MDX content with custom components
 */
export function CustomMDX(props) {
  return (
    <MDXRemote
      {...props}
      components={{ ...components, ...(props.components || {}) }}
    />
  )
}
