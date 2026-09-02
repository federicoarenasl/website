import { Posts } from 'app/components/posts'
import { PATH_TO_RESEARCH_MDX } from 'app/sitemap'

export const metadata = {
  title: 'Research',
  description: 'Write-ups of my personal research.',
}

export default function Page() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">Research</h1>
      <Posts mdxPath={PATH_TO_RESEARCH_MDX} urlPath="/research" />
    </section>
  )
}
