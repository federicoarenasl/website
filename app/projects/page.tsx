import { Posts } from 'app/components/posts'
import { PATH_TO_PROJECT_MDX } from 'app/sitemap'

export const metadata = {
  title: 'Project',
  description: 'Browse through some of my personal projects.',
}

export default function Page() {
  return (
    <section>
      <h1 className="font-semibold text-2xl mb-8 tracking-tighter">My Projects</h1>
      <Posts mdxPath={PATH_TO_PROJECT_MDX} urlPath="/projects" />
    </section>
  )
}
