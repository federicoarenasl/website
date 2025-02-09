import { readMDXFile } from 'app/utils'
import { CustomMDX } from 'app/components/mdx'
import path from 'path'
export const metadata = {
  title: 'CV',
  description: 'My CV.',
}

export default function Page() {
  let {metadata, content} = readMDXFile(path.join(process.cwd(), 'app', 'cv', 'cv.mdx'))

  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">{metadata.title}</h1>
      <article className="prose">
        <CustomMDX source={content} />
      </article>
    </section>
  )

}
