
export default function Page() {
  return (
    <section>
      <h1 className="mb-8 text-2xl font-semibold tracking-tighter">
        Hey there, I'm Federico Arenas.
      </h1>
      <p className="mb-4">
        {`Currently Senior AI Engineer at `}
        <a 
          href="https://www.materiom.org"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 dark:text-white underline decoration-white dark:decoration-white hover:text-blue-800 dark:hover:text-white visited:text-purple-600 dark:visited:text-white"
        >
          Materiom
        </a>
        {`. Leading the development of data-intensive applications that utilize AI to accelerate bio-based material innovation.`}
      </p>
      <p className="mb-4">
        {`Previously contributed to early-stage AI startups in the self-driving space, computer vision space, and entreprise data space.`}
      </p>
      <p className="mb-4">
        {`I find balance through running, philosophy, and my guitar.`}
      </p>
      <p className="mb-4">
        {`Based in London. Feel free to reach out.`}
      </p>
    </section>
  )
}
