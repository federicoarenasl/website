import { readMDXFile } from 'app/utils'
import { CustomMDX } from 'app/components/mdx/mdx'
import path from 'path'
import { PdfLink, PrintWatermark } from './print-button'

const CV_FILES = {
  science: 'cv-science.mdx',
  agents: 'cv-agents.mdx',
} as const

type Focus = keyof typeof CV_FILES

const DEFAULT_FOCUS: Focus = 'science'

function resolveFocus(focus?: string | string[]): Focus {
  let value = Array.isArray(focus) ? focus[0] : focus
  return value && value in CV_FILES ? (value as Focus) : DEFAULT_FOCUS
}

export const metadata = {
  title: 'CV',
  description: 'My CV.',
  robots: {
    index: false,
    follow: false,
    nocache: true,
    googleBot: {
      index: false,
      follow: false,
    },
  },
}

export default function Page({
  searchParams,
}: {
  searchParams: { focus?: string | string[] }
}) {
  let focus = resolveFocus(searchParams.focus)
  let {metadata, content} = readMDXFile(path.join(process.cwd(), 'app', 'cv', CV_FILES[focus]))

  return (
    <section>
      <div className="flex items-center gap-2 mb-8">
        <h1 className="font-semibold text-2xl tracking-tighter">{metadata.title}</h1>
        <PdfLink href={`/cv-${focus}.pdf`} />
      </div>
      <article className="prose">
        <CustomMDX source={content} />
      </article>
      <PrintWatermark />
    </section>
  )
}
