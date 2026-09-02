
export default function Page() {
  return (
    <section>
      <h1 className="mb-8 text-2xl font-semibold tracking-tighter">
        Hey there, I'm Federico Arenas.
      </h1>
      <p className="mb-4">
        {`AI Engineer at `}
        <a
          href="https://www.materiom.org"
          target="_blank"
          rel="noopener noreferrer"
          className="text-black underline decoration-black hover:text-neutral-600 visited:text-neutral-700"
        >
          Materiom
        </a>
        {`, where I build machine learning and agentic systems to accelerate scientific discovery in the materials sector. My work focuses on agentic scientific orchestration, geometric deep learning, and software engineering.`}
      </p>
      <p className="mb-4">
        {`Previously, I was part of early-stage startups building autonomous vehicle validation platforms, synthetic data engines, and scalable ML infrastructure.`}
      </p>
      <p className="mb-4">
        {`Outside work, I find balance through running, cycling, philosophy, and playing guitar.`}
      </p>
      <p className="mb-4">
        {`Based in London. Feel free to reach out.`}
      </p>
    </section>
  )
}
