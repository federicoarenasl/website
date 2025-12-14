import React from 'react'
import { FootnoteReference } from './footnotes'

/**
 * Component that processes text and converts footnote references like ^[1], ^[2] into clickable footnote links
 * This supports the standard Markdown footnote format
 */
export function TextWithFootnotes({ 
  text, 
  footnotes = [] 
}: { 
  text: string
  footnotes?: Array<{ id: string; content: string }>
}) {
  // Create a map of footnote IDs to content for quick lookup
  const footnoteMap = new Map(footnotes.map(fn => [fn.id, fn.content]))
  
  // Split text by footnote references and process each part
  // Only support ^[1] format (standard Markdown footnotes)
  const parts = text.split(/(\^\[\d+\])/g)
  
  return (
    <>
      {parts.map((part, index) => {
        // Check if this part is a footnote reference like ^[1], ^[2], etc.
        const footnoteMatch = part.match(/^\^\[(\d+)\]$/)
        
        if (footnoteMatch) {
          const [, footnoteId] = footnoteMatch
          const footnoteContent = footnoteMap.get(footnoteId)
          
          if (footnoteContent) {
            return (
              <FootnoteReference key={index} id={footnoteId}>
                {footnoteContent}
              </FootnoteReference>
            )
          } else {
            // If no footnote content found, render as plain text
            return <span key={index}>{part}</span>
          }
        }
        
        // Regular text content
        return <span key={index}>{part}</span>
      })}
    </>
  )
}

/**
 * Utility function to extract footnotes from MDX content
 * Looks for footnote definitions at the end of the content in the format ^[1] content
 */
export function extractFootnotesFromContent(content: string): {
  cleanedContent: string
  footnotes: Array<{ id: string; content: string }>
} {
  const footnotes: Array<{ id: string; content: string }> = []
  
  // Pattern to match footnote definitions at the end of content
  // Matches lines like: ^[1] Insert citation
  const footnoteDefPattern = /^\^\[(\d+)\]\s*(.+)$/gm
  let match
  
  while ((match = footnoteDefPattern.exec(content)) !== null) {
    const [, id, content] = match
    footnotes.push({ id, content: content.trim() })
  }
  
  // Remove footnote definitions from the main content
  const cleanedContent = content.replace(footnoteDefPattern, '').trim()
  
  return { cleanedContent, footnotes }
}

/**
 * Component that can be used in MDX to process a paragraph with footnotes
 * Usage: <ProcessedParagraph text="Some text ^[1] with footnotes ^[2]" footnotes={footnotes} />
 */
export function ProcessedParagraph({ 
  text, 
  footnotes = [] 
}: { 
  text: string
  footnotes?: Array<{ id: string; content: string }>
}) {
  return (
    <p>
      <TextWithFootnotes text={text} footnotes={footnotes} />
    </p>
  )
}
