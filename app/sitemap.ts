import { collectMDXData } from 'app/utils'
export const baseUrl = 'https://portfolio-blog-starter.vercel.app'
import { PATH_TO_BLOG_MDX } from 'app/components/posts'
import { PATH_TO_PROJECT_MDX } from 'app/components/projects'

export default async function sitemap() {
  let blogs = collectMDXData(PATH_TO_BLOG_MDX).map((post) => ({
    url: `${baseUrl}/thoughts/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  let projects = collectMDXData(PATH_TO_PROJECT_MDX).map((project) => ({
    url: `${baseUrl}/projects/${project.slug}`,
    lastModified: project.metadata.publishedAt,
  }))

  let routes = ['', '/thoughts', '/projects'].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...blogs, ...projects]
}
