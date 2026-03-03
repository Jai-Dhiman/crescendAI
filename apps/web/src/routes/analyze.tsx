import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/analyze')({ component: AnalyzePage })

function AnalyzePage() {
  return (
    <section className="py-24 md:py-40">
      <div className="container-editorial">
        <div className="max-w-2xl mx-auto text-center">
          <h1 className="font-display text-display-md md:text-display-lg text-ink-900 mb-6">
            Analyze
          </h1>
          <p className="text-body-lg text-ink-600 mb-8">
            The analysis experience is moving to our iOS app -- record, listen,
            and get feedback right from your phone. Coming soon.
          </p>
          <a
            href="/"
            className="text-body-sm text-clay-600 underline underline-offset-2 hover:text-clay-700 transition-colors duration-200"
          >
            Back to home
          </a>
        </div>
      </div>
    </section>
  )
}
