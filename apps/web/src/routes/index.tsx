import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: LandingPage })

function LandingPage() {
  return (
    <div>
      <HeroSection />
      <FeatureCardsSection />
      <CascadingQuoteSection />
      <DeviceMockupSection />
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
            className="bg-accent text-cream px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
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
      id: 'listen',
      title: 'Record yourself playing.',
      description:
        'When you pause, your teacher is ready with the one thing that matters most.',
    },
    {
      id: 'annotate',
      title: 'Exercises built for you.',
      description:
        'Targeted practice for the specific passage and skill your teacher identified.',
    },
    {
      id: 'exercises',
      title: 'A teacher who knows your playing.',
      description:
        'Your teacher remembers what you\'ve been working on, notices when you improve, and adapts.',
    },
  ]

  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {cards.map((card) => (
            <div
              key={card.id}
              className="bg-surface border border-border rounded-xl overflow-hidden"
            >
              {/* Animation area */}
              <div className="aspect-[4/3] bg-surface-2 overflow-hidden">
                <video
                  src={`/anim-${card.id}.mp4`}
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="w-full h-full object-cover"
                />
              </div>

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
          {/* Staircase photos */}
          <div className="flex flex-col">
            <div className="w-[55%] self-start">
              <img
                src="/Image2.jpg"
                alt="Practicing alone -- the struggle of hearing your own mistakes"
                className="w-full object-cover"
                style={{ aspectRatio: '3/2' }}
              />
            </div>
            <div className="w-[55%] self-center">
              <img
                src="/Image3.jpg"
                alt="A moment of guidance -- focused attention on the score"
                className="w-full object-cover"
                style={{ aspectRatio: '3/2' }}
              />
            </div>
            <div className="w-[55%] self-end">
              <img
                src="/Image4.jpg"
                alt="The breakthrough -- playing with confidence"
                className="w-full object-cover"
                style={{ aspectRatio: '3/2' }}
              />
            </div>
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

function DeviceMockupSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="max-w-5xl mx-auto px-6 lg:px-12">
        <div className="relative">
          {/* Laptop */}
          <img
            src="/MacbookMockup.png"
            alt="CrescendAI desktop app showing a practice session with score analysis"
            className="w-full"
          />
          {/* Phone -- overlapping bottom-right */}
          <img
            src="/iphonemockup.png"
            alt="CrescendAI mobile app showing exercise recommendations"
            className="absolute bottom-[-8%] right-[-4%] w-[28%] lg:w-[25%]"
          />
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
          Every pianist deserves a great teacher.
        </h2>

        <div className="mt-10">
          <a
            href="/app"
            className="bg-accent text-cream px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
          >
            Start Practicing
          </a>
        </div>
      </div>
    </section>
  )
}
