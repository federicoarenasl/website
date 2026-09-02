import fs from 'fs'
import path from 'path'

type Metadata = {
  title: string
  publishedAt: string
  summary: string
  image?: string
  github?: string
  /** When true (or "true" from frontmatter), post is still browsable at /research/[slug] but omitted from the research list page. */
  hiddenFromList?: boolean | string
}

function parseFrontmatter(fileContent: string) {
  let frontmatterRegex = /---\s*([\s\S]*?)\s*---/
  let match = frontmatterRegex.exec(fileContent)
  let frontMatterBlock = match![1]
  let content = fileContent.replace(frontmatterRegex, '').trim()
  let frontMatterLines = frontMatterBlock.trim().split('\n')
  let metadata: Partial<Metadata> = {}

  frontMatterLines.forEach((line) => {
    let [key, ...valueArr] = line.split(': ')
    let value = valueArr.join(': ').trim()
    value = value.replace(/^['"](.*)['"]$/, '$1') // Remove quotes
    metadata[key.trim() as keyof Metadata] = value
  })

  return { metadata: metadata as Metadata, content }
}

function getMDXFiles(dir) {
    return fs.readdirSync(dir).filter((file) => path.extname(file) === '.mdx')
  }



export function formatDate(date: string) {
    if (!date.includes('T')) {
      date = `${date}T00:00:00`
    }
    let targetDate = new Date(date)
    
    return targetDate.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  }
  

export function getMDXData(dir) {
    let mdxFiles = getMDXFiles(dir)
    return mdxFiles.map((file) => {
      let { metadata, content } = readMDXFile(path.join(dir, file))
      let slug = path.basename(file, path.extname(file))
  
      return {
        metadata,
        slug,
        content,
      }
    })
  }

export function readMDXFile(filePath) {
  let rawContent = fs.readFileSync(filePath, 'utf-8')
  return parseFrontmatter(rawContent)
}

export function collectMDXData(dir: string) {
    return getMDXData(path.join(process.cwd(), dir))
  }