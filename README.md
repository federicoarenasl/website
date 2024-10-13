# Personal Website
This is my personal website, built with Next.js and Tailwind CSS. It is a portfolio site with a blog, and is optimized for SEO. 

It is hosted using Vercel, and uses Vercel's Speed Insights and Web Analytics to track performance and user engagement.

## Template Details
This is a porfolio site template complete with a blog. Includes:

- MDX and Markdown support
- Optimized for SEO (sitemap, robots, JSON-LD schema)
- RSS Feed
- Dynamic OG images
- Syntax highlighting
- Tailwind v4
- Vercel Speed Insights / Web Analytics
- Geist font

## Configuration
This app uses `Performant npm` to run Next.js, this means that you need to first install `pnpm`:

```bash
npm install -g pnpm
```

Then, install any local dependencies, this will update your `pnpm-lock.yaml` file:

```bash
pnpm install
```

## Running Locally

To run the app locally, use the following command:
```bash
pnpm dev
```
You can then view the app at `http://localhost:3000`.

## Deploy on Vercel

This project is ready to deploy on Vercel. Once your changes have been made, push your PR into develop, and then to main. This should trigger the deployment to your app.s