import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/analyze')({ component: AnalyzePage })

function AnalyzePage() {
  return (
    <section className="pt-32 pb-24 md:pt-40 md:pb-32">
      <div className="max-w-2xl mx-auto px-6 text-center">
        <h1 className="font-display text-display-md md:text-display-lg text-cream mb-6">
          Analyze
        </h1>
        <p className="text-body-lg text-text-secondary mb-8">
          The analysis experience is moving to our iOS app -- record, listen,
          and get feedback right from your phone. Coming soon.
        </p>
        <a
          href="/"
          className="text-body-sm text-text-secondary underline underline-offset-2 hover:text-cream transition-colors"
        >
          Back to home
        </a>
      </div>
    </section>
  )
}
