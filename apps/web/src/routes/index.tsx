import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: LandingPage })

function LandingPage() {
  return (
    <div>
      <HeroSection />
    </div>
  )
}

function HeroSection() {
  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      {/* Full-bleed background image */}
      <img
        src="/Image1.jpg"
        alt="Grand piano seen from above"
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Gradient overlay for text legibility */}
      <div className="absolute inset-0 bg-gradient-to-t from-espresso/80 via-espresso/30 to-espresso/10" />

      {/* Content */}
      <div className="relative z-10 text-center px-6">
        <h1
          className="font-display text-cream text-balance"
          style={{ fontSize: 'clamp(3rem, 8vw, 7rem)', lineHeight: 1.05, letterSpacing: '-0.03em' }}
        >
          A teacher for every pianist.
        </h1>

        <div className="mt-10">
          <a
            href="/analyze"
            className="bg-cream text-espresso rounded-full px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
          >
            Start Practicing
          </a>
        </div>
      </div>
    </section>
  )
}
