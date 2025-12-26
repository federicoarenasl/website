import { Posts } from 'app/components/posts'
import { PATH_TO_BLOG_MDX } from 'app/sitemap'


export const metadata = {
  title: 'Thoughts',
  description: 'Some random access thoughts.',
}

export default function Page() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">Thoughts</h1>
      <Posts mdxPath={PATH_TO_BLOG_MDX} urlPath="/thoughts" />
    </section>
  )
}
