import Link from 'next/link'
import { formatDate, collectMDXData } from 'app/utils'
import React from 'react'

type PostsProps = {
  mdxPath: string
  urlPath: string
}

export const Posts: React.FC<PostsProps> = ({ mdxPath, urlPath }) => {
  let allBlogs = collectMDXData(mdxPath)

  return (
    <div>
      {allBlogs
        .sort((a, b) => {
          if (
            new Date(a.metadata.publishedAt) > new Date(b.metadata.publishedAt)
          ) {
            return -1
          }
          return 1
        })
        .map((post) => (
          <Link
            key={post.slug}
            className="flex flex-row items-center justify-between mb-4"
            href={`${urlPath}/${post.slug}`}
          >
            <p className="text-neutral-900 dark:text-neutral-100 max-w-[75%]">
              {post.metadata.title}
            </p>
            <p className="text-neutral-500 dark:text-neutral-400 text-sm tabular-nums shrink-0">
              {formatDate(post.metadata.publishedAt)}
            </p>
          </Link>
        ))}
    </div>
  )
}
