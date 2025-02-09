import { Posts } from 'app/components/posts'

export const PATH_TO_BLOG_MDX = 'app/thoughts/posts'

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
