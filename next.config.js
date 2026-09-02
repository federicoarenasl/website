/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      // The research section used to live at /projects. Keep the old URLs
      // working — they're baked into published CV PDFs and search results.
      {
        source: '/projects',
        destination: '/research',
        permanent: true,
      },
      {
        source: '/projects/:slug',
        destination: '/research/:slug',
        permanent: true,
      },
    ]
  },
}

module.exports = nextConfig
