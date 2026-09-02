import { collectMDXData } from 'app/utils'
export const baseUrl = 'https://portfolio-blog-starter.vercel.app'

export const PATH_TO_BLOG_MDX = 'app/thoughts/posts'
export const PATH_TO_RESEARCH_MDX = 'app/research/research'

export default async function sitemap() {
  let blogs = collectMDXData(PATH_TO_BLOG_MDX).map((post) => ({
    url: `${baseUrl}/thoughts/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  let research = collectMDXData(PATH_TO_RESEARCH_MDX).map((post) => ({
    url: `${baseUrl}/research/${post.slug}`,
    lastModified: post.metadata.publishedAt,
  }))

  let routes = ['', '/thoughts', '/research'].map((route) => ({
    url: `${baseUrl}${route}`,
    lastModified: new Date().toISOString().split('T')[0],
  }))

  return [...routes, ...blogs, ...research]
}
