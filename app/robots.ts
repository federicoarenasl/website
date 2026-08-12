import { baseUrl } from 'app/sitemap'

export default function robots() {
  return {
    rules: [
      {
        userAgent: '*',
        disallow: ['/cv', '/cv.pdf'],
      },
    ],
    sitemap: `${baseUrl}/sitemap.xml`,
  }
}
