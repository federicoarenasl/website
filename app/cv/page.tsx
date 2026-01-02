import { readMDXFile } from 'app/utils'
import { CustomMDX } from 'app/components/mdx/mdx'
import path from 'path'
import { PrintButton, PrintWatermark } from './print-button'

export const metadata = {
  title: 'CV',
  description: 'My CV.',
}

export default function Page() {
  let {metadata, content} = readMDXFile(path.join(process.cwd(), 'app', 'cv', 'cv.mdx'))

  return (
    <section>
      <div className="flex items-center gap-2 mb-8">
        <h1 className="font-semibold text-2xl tracking-tighter">{metadata.title}</h1>
        <PrintButton />
      </div>
      <article className="prose">
        <CustomMDX source={content} />
      </article>
      <PrintWatermark />
    </section>
  )
}
