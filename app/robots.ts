import { baseUrl } from 'app/sitemap'

export default function robots() {
  return {
    rules: [
      {
        userAgent: '*',
        disallow: ['/cv', '/cv-science.pdf', '/cv-agents.pdf'],
      },
    ],
    sitemap: `${baseUrl}/sitemap.xml`,
  }
}
