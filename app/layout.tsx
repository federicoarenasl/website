// Import global CSS styles for the entire application
import './global.css'
import type { Metadata } from 'next'
// Import Geist font families for consistent typography
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
// Import core layout components
import { Navbar } from './components/nav'
import { Analytics } from '@vercel/analytics/react'
import { SpeedInsights } from '@vercel/speed-insights/next'
import Footer from './components/footer'
import { baseUrl } from './sitemap'

/**
 * Metadata configuration for the entire application
 * Defines SEO settings, Open Graph data, and robot directives
 */
export const metadata: Metadata = {
  // Base URL for all relative URLs in metadata
  metadataBase: new URL(baseUrl),
  title: {
    // Default title when no specific page title is provided
    default: "Federico Arenas",
    // Template for page-specific titles (e.g., "Blog Post | Federico Arenas")
    template: '%s | Federico Arenas',
  },
  description: 'These are my thoughts.',
  // Viewport configuration for proper mobile rendering
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 5,
  },
  // Open Graph metadata for social media sharing
  openGraph: {
    title: "Federico Arenas",
    description: 'These are my thoughts.',
    url: baseUrl,
    siteName: "Federico Arenas",
    locale: 'en_US',
    type: 'website',
  },
  // Search engine crawler directives
  robots: {
    index: true, // Allow search engines to index this site
    follow: true, // Allow search engines to follow links
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1, // No limit on video preview length
      'max-image-preview': 'large', // Allow large image previews
      'max-snippet': -1, // No limit on text snippet length
    },
  },
}

/**
 * Utility function to conditionally join CSS class names
 * Filters out falsy values and joins remaining classes with spaces
 * @param classes - Array of class names (can include falsy values)
 * @returns Joined string of valid class names
 */
const cx = (...classes) => classes.filter(Boolean).join(' ')

/**
 * Root layout component that wraps all pages in the application
 * Provides consistent structure, fonts, styling, and analytics
 * @param children - The page content to be rendered within the layout
 * @returns JSX element containing the complete HTML document structure
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="en"
      className={cx(
        // Base color scheme with dark mode support
        'text-black bg-white dark:text-white dark:bg-black',
        // Apply Geist font variables for CSS custom properties
        GeistSans.variable,
        GeistMono.variable
      )}
    >
      <body className="antialiased max-w-xl mx-auto px-4 mt-8 lg:max-w-4xl">
        {/* Main content container with responsive layout */}
        <main className="flex-auto min-w-0 mt-6 flex flex-col px-2 md:px-0">
          {/* Site navigation */}
          <Navbar />
          {/* Dynamic page content */}
          {children}
          {/* Site footer */}
          <Footer />
          {/* Vercel Analytics for tracking page views and performance */}
          <Analytics />
          {/* Vercel Speed Insights for Core Web Vitals monitoring */}
          <SpeedInsights />
        </main>
      </body>
    </html>
  )
}
