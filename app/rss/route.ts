import { baseUrl } from 'app/sitemap'
import { collectMDXData } from 'app/utils'
import { PATH_TO_BLOG_MDX, PATH_TO_RESEARCH_MDX } from 'app/sitemap'

export async function GET() {
  let allBlogs = await collectMDXData(PATH_TO_BLOG_MDX)
  let allResearch = await collectMDXData(PATH_TO_RESEARCH_MDX)

  const blogItemsXml = allBlogs
    .sort((a, b) => {
      if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
        return -1
      }
      return 1
    })
    .map(
      (post) =>
        `<item>
          <title>${post.metadata.title}</title>
          <link>${baseUrl}/thoughts/${post.slug}</link>
          <description>${post.metadata.summary || ''}</description>
          <pubDate>${new Date(
            post.metadata.publishedAt
          ).toUTCString()}</pubDate>
        </item>`
    )
    .join('\n')
  
  const researchItemsXml = allResearch
    .sort((a, b) => {
      if (new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)) {
        return -1
      }
      return 1
    })
    .map(
      (post) =>
        `<item>
          <title>${post.metadata.title}</title>
          <link>${baseUrl}/research/${post.slug}</link>
          <description>${post.metadata.summary || ''}</description>
          <pubDate>${new Date(
            post.metadata.publishedAt
          ).toUTCString()}</pubDate>
        </item>`
    )
    .join('\n')

  const rssFeed = `<?xml version="1.0" encoding="UTF-8" ?>
  <rss version="2.0">
    <channel>
        <title>Federico Arenas</title>
        <link>${baseUrl}</link>
        <description>This is my portfolio RSS feed</description>
        ${blogItemsXml}
        ${researchItemsXml}
    </channel>
  </rss>`

  return new Response(rssFeed, {
    headers: {
      'Content-Type': 'text/xml',
    },
  })
}
