
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
        {`, where I build machine learning systems and closed-loop pipelines to accelerate scientific discovery across bio-based materials. My work focuses on bridging high-throughput in-silico simulations, geometric deep learning, and automated physical lab experimentation.`}
      </p>
      <p className="mb-4">
        {`Previously, I built autonomous vehicle validation platforms, synthetic data engines using game engines, and scalable ML infrastructure at early-stage startups.`}
      </p>
      <p className="mb-4">
        {`Outside of engineering, I find balance through running, philosophy, and playing guitar.`}
      </p>
      <p className="mb-4">
        {`Based in London. Feel free to reach out.`}
      </p>
    </section>
  )
}
