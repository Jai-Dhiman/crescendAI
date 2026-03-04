import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: LandingPage })

function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCardsSection />
      <CascadingQuoteSection />
      <FinalCtaSection />
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
      <div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(to top, #2D2926 0%, #2D2926 5%, rgba(45,41,38,0.7) 30%, rgba(45,41,38,0.2) 60%, rgba(45,41,38,0.05) 100%)',
        }}
      />

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
            href="/app"
            className="bg-cream text-espresso px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
          >
            Start Practicing
          </a>
        </div>
      </div>
    </section>
  )
}

function FeatureCardsSection() {
  const cards = [
    {
      title: 'Your teacher is listening',
      description:
        'Your phone listens while you play. When you pause and ask, your teacher is ready with the one thing that matters most.',
    },
    {
      title: 'Exercises built for you',
      description:
        'Not generic drills. Targeted practice for the specific passage and skill your teacher identified.',
    },
    {
      title: 'See what you hear',
      description:
        'The score lights up on a piano keyboard. See the notes, the fingering, the dynamics -- then play along.',
    },
  ]

  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {cards.map((card) => (
            <div
              key={card.title}
              className="bg-surface border border-border rounded-xl overflow-hidden"
            >
              {/* Placeholder visual area */}
              <div className="aspect-[4/3] bg-surface-2" />

              {/* Text content */}
              <div className="p-6 lg:p-8">
                <h3 className="font-display text-display-sm text-cream mb-3">
                  {card.title}
                </h3>
                <p className="text-body-md text-text-secondary">
                  {card.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function CascadingQuoteSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 lg:grid-cols-[5fr_6fr] gap-12 lg:gap-16 items-center">
          {/* Cascading photos */}
          <div className="relative h-[700px] lg:h-[850px]">
            <img
              src="/Image2.jpg"
              alt="Sheet music resting on piano keys"
              className="absolute top-0 left-0 w-[55%] object-cover"
              style={{ aspectRatio: '4/5' }}
            />
            <img
              src="/Image3.jpg"
              alt="Piano score with dynamic markings"
              className="absolute top-[28%] left-[20%] w-[55%] object-cover"
              style={{ aspectRatio: '4/5' }}
            />
            <img
              src="/Image4.jpg"
              alt="Hands playing piano in warm light"
              className="absolute top-[56%] left-[40%] w-[55%] object-cover"
              style={{ aspectRatio: '4/5' }}
            />
          </div>

          {/* Pull quote */}
          <div>
            <blockquote className="font-display italic text-display-md lg:text-display-lg text-cream leading-snug">
              "What's the one thing that sounds off that I can't hear myself?"
            </blockquote>
          </div>
        </div>
      </div>
    </section>
  )
}

function FinalCtaSection() {
  return (
    <section className="py-32 lg:py-40">
      <div className="max-w-4xl mx-auto px-6 lg:px-12 text-center">
        <h2 className="font-display text-display-md lg:text-display-xl text-cream">
          Start practicing with a teacher who's always listening.
        </h2>

        <div className="mt-10">
          <a
            href="/app"
            className="bg-cream text-espresso px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
          >
            Start Practicing
          </a>
        </div>
      </div>
    </section>
  )
}
