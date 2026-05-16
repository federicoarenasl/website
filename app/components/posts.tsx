import Link from 'next/link'
import { formatDate, collectMDXData } from 'app/utils'
import React from 'react'

type PostsProps = {
  mdxPath: string
  urlPath: string
}

export const Posts: React.FC<PostsProps> = ({ mdxPath, urlPath }) => {
  let allBlogs = collectMDXData(mdxPath).filter(
    (post) => !post.metadata.hiddenFromList
  )

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
            className="flex flex-col mb-6 group"
            href={`${urlPath}/${post.slug}`}
          >
            <p className="text-neutral-900 dark:text-neutral-100 group-hover:text-neutral-600 dark:group-hover:text-neutral-300 transition-colors">
              {post.metadata.title}
            </p>
            {post.metadata.summary && (
              <p className="text-neutral-500 dark:text-neutral-400 text-sm mt-1">
                {post.metadata.summary}
              </p>
            )}
          </Link>
        ))}
    </div>
  )
}
