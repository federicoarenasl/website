'use client'

import React, { createContext, useContext, useState, useRef, useEffect, useCallback } from 'react'

// Types
interface Footnote {
  id: string
  number: number
  content: React.ReactNode
  referenceId: string
}

interface FootnoteContextType {
  footnotes: Footnote[]
  addFootnote: (id: string, content: React.ReactNode) => number
  getFootnoteNumber: (id: string) => number | null
}

// Context
const FootnoteContext = createContext<FootnoteContextType | null>(null)

// Hook to use footnote context
export function useFootnotes() {
  const context = useContext(FootnoteContext)
  if (!context) {
    throw new Error('useFootnotes must be used within a FootnoteProvider')
  }
  return context
}

// Footnote Provider Component
export function FootnoteProvider({ children }: { children: React.ReactNode }) {
  const [footnotes, setFootnotes] = useState<Footnote[]>([])
  const footnoteCounter = useRef(0)
  const footnoteMapRef = useRef<Map<string, number>>(new Map())

  const addFootnote = useCallback((id: string, content: React.ReactNode): number => {
    // Check if we've already added this footnote
    const existingNumber = footnoteMapRef.current.get(id)
    if (existingNumber !== undefined) {
      return existingNumber
    }
    
    // Add new footnote
    footnoteCounter.current += 1
    const number = footnoteCounter.current
    
    const newFootnote: Footnote = {
      id,
      number,
      content,
      referenceId: `fn-ref-${id}`
    }
    
    footnoteMapRef.current.set(id, number)
    setFootnotes(prev => [...prev, newFootnote])
    
    return number
  }, [])

  const getFootnoteNumber = useCallback((id: string): number | null => {
    return footnoteMapRef.current.get(id) ?? null
  }, [])

  return (
    <FootnoteContext.Provider value={{ footnotes, addFootnote, getFootnoteNumber }}>
      {children}
    </FootnoteContext.Provider>
  )
}

// Footnote Reference Component
interface FootnoteReferenceProps {
  id: string
  children?: React.ReactNode
}

export function FootnoteReference({ id, children }: FootnoteReferenceProps) {
  const { addFootnote, getFootnoteNumber } = useFootnotes()
  const [number, setNumber] = useState<number | null>(null)
  const hasProcessed = useRef(false)

  useEffect(() => {
    // Prevent double-execution in strict mode
    if (hasProcessed.current) return
    hasProcessed.current = true

    if (children) {
      const footnoteNumber = addFootnote(id, children)
      setNumber(footnoteNumber)
    } else {
      const existingNumber = getFootnoteNumber(id)
      setNumber(existingNumber)
    }
  }, [id, children, addFootnote, getFootnoteNumber])

  if (number === null) return null

  const scrollToFootnote = () => {
    const footnoteElement = document.getElementById(`fn-${id}`)
    if (footnoteElement) {
      footnoteElement.scrollIntoView({ behavior: 'smooth', block: 'start' })
      // Add a temporary highlight effect
      footnoteElement.classList.add('bg-yellow-100', 'dark:bg-yellow-900')
      setTimeout(() => {
        footnoteElement.classList.remove('bg-yellow-100', 'dark:bg-yellow-900')
      }, 2000)
    }
  }

  return (
    <sup>
      <button
        onClick={scrollToFootnote}
        className="text-gray-900 dark:text-gray-100 hover:text-gray-600 dark:hover:text-gray-400 cursor-pointer font-medium transition-colors duration-200 opacity-80 hover:opacity-100"
        aria-label={`Go to footnote ${number}`}
      >
        [{number}]
      </button>
    </sup>
  )
}

// Footnote Definition Component
interface FootnoteDefinitionProps {
  id: string
  children: React.ReactNode
}

export function FootnoteDefinition({ id, children }: FootnoteDefinitionProps) {
  const { getFootnoteNumber } = useFootnotes()
  const number = getFootnoteNumber(id)

  if (number === null) return null

  const scrollToReference = () => {
    const referenceElement = document.getElementById(`fn-ref-${id}`)
    if (referenceElement) {
      referenceElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
      // Add a temporary highlight effect
      referenceElement.classList.add('bg-yellow-100', 'dark:bg-yellow-900')
      setTimeout(() => {
        referenceElement.classList.remove('bg-yellow-100', 'dark:bg-yellow-900')
      }, 2000)
    }
  }

  return (
    <div id={`fn-${id}`} className="flex gap-2 py-2 border-b border-gray-200 dark:border-gray-700 last:border-b-0">
      <button
        onClick={scrollToReference}
        className="text-gray-900 dark:text-gray-100 hover:text-gray-600 dark:hover:text-gray-400 cursor-pointer font-medium flex-shrink-0 transition-colors duration-200 opacity-80 hover:opacity-100 self-start"
        aria-label={`Go back to reference ${number}`}
      >
        [{number}]
      </button>
      <div className="text-sm text-gray-700 dark:text-gray-300">
        {children}
      </div>
    </div>
  )
}

// Footnote List Component
export function FootnoteList() {
  const { footnotes } = useFootnotes()

  if (footnotes.length === 0) return null

  return (
    <div className="mt-12 pt-6 border-t border-gray-200 dark:border-gray-700">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
        Footnotes
      </h3>
      <div className="space-y-1">
        {footnotes.map((footnote) => (
          <FootnoteDefinition key={footnote.id} id={footnote.id}>
            {footnote.content}
          </FootnoteDefinition>
        ))}
      </div>
    </div>
  )
}

// Utility function to parse footnote references from text
export function parseFootnotes(text: string): { text: string; footnotes: Array<{ id: string; content: string }> } {
  const footnotes: Array<{ id: string; content: string }> = []
  
  // Match patterns like [1], [2], etc. at the end of lines or paragraphs
  const footnotePattern = /\[(\d+)\]/g
  let match
  const footnoteIds = new Set<string>()
  
  while ((match = footnotePattern.exec(text)) !== null) {
    footnoteIds.add(match[1])
  }
  
  // Extract footnote definitions from the end of the text
  const footnoteDefPattern = /^\[(\d+)\]\s*(.+)$/gm
  while ((match = footnoteDefPattern.exec(text)) !== null) {
    const [, id, content] = match
    if (footnoteIds.has(id)) {
      footnotes.push({ id, content: content.trim() })
    }
  }
  
  // Remove footnote definitions from the main text
  const cleanedText = text.replace(footnoteDefPattern, '').trim()
  
  return { text: cleanedText, footnotes }
}
