import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: LandingPage })

function LandingPage() {
  return (
    <div>
      <HeroSection />
      <hr className="editorial-rule" />
      <SocialProofBar />
      <hr className="editorial-rule" />
      <ProblemSection />
      <hr className="editorial-rule" />
      <HowItWorksSection />
      <hr className="editorial-rule" />
      <FeedbackSection />
      <hr className="editorial-rule" />
      <ResearchSection />
      <FinalCtaSection />
    </div>
  )
}

function HeroSection() {
  return (
    <section>
      <div className="editorial-bleed">
        <div className="editorial-bleed-text lg:py-32">
          <h1 className="font-display text-display-md lg:text-display-lg text-ink-900 mb-6">
            A teacher for every pianist.
          </h1>
          <p className="text-body-lg text-ink-600 mb-8 max-w-md">
            Record yourself playing with your phone and get the feedback a great
            teacher would give you: how to clean up your pedaling, shape your
            dynamics, warm your tone.
          </p>
          <div>
            <a href="/analyze" className="btn-primary text-body-sm">
              Start Free
            </a>
            <p className="text-body-sm text-ink-500 mt-3">
              No account required
            </p>
          </div>
        </div>

        <div className="editorial-bleed-image">
          <img
            src="/Image1.jpg"
            alt="Grand piano seen from above"
            className="w-full object-cover"
            style={{ aspectRatio: '7/8' }}
          />
        </div>
      </div>
    </section>
  )
}

function SocialProofBar() {
  return (
    <section className="py-10 md:py-14">
      <div className="container-editorial">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 text-center">
          <div>
            <p className="font-display text-display-md text-ink-900">55%</p>
            <p className="text-label-sm uppercase tracking-wider text-clay-600 mt-1">
              More accurate
            </p>
          </div>
          <div>
            <p className="font-display text-display-md text-ink-900">30+</p>
            <p className="text-label-sm uppercase tracking-wider text-clay-600 mt-1">
              Educator interviews
            </p>
          </div>
          <div>
            <p className="font-display text-display-md text-ink-900">15s</p>
            <p className="text-label-sm uppercase tracking-wider text-clay-600 mt-1">
              To get feedback
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

function ProblemSection() {
  return (
    <section className="py-16 md:py-24">
      <div className="container-editorial">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="font-display text-display-sm md:text-display-lg text-ink-900 mb-6">
            Any app can check your notes.
          </h2>
          <p className="text-body-lg text-ink-600">
            But that's not what separates good playing from great playing. Your
            tone. Your dynamics. Your phrasing. That's always needed a teacher.
          </p>
        </div>
      </div>
    </section>
  )
}

function HowItWorksSection() {
  return (
    <section id="how-it-works" className="scroll-mt-20 py-16 md:py-24">
      <div className="container-editorial">
        <div className="grid grid-cols-1 lg:grid-cols-[2fr_3fr] gap-8 lg:gap-16 items-start">
          <div>
            <img
              src="/Image2.jpg"
              alt="Sheet music resting on vintage piano keys"
              className="w-full object-cover rounded"
              style={{ aspectRatio: '4/5' }}
            />
          </div>

          <div>
            <h2 className="font-display text-display-sm md:text-display-md text-ink-900 mb-4">
              How It Works
            </h2>
            <p className="text-body-lg text-ink-600 mb-10">
              Under 15 seconds from recording to feedback.
            </p>

            <ol className="space-y-8">
              <li>
                <span className="font-display text-display-xl text-paper-300 block leading-none">
                  01
                </span>
                <p className="text-body-md text-ink-700 mt-1">
                  Record yourself playing
                </p>
              </li>
              <li>
                <span className="font-display text-display-xl text-paper-300 block leading-none">
                  02
                </span>
                <p className="text-body-md text-ink-700 mt-1">
                  Upload your recording
                </p>
              </li>
              <li>
                <span className="font-display text-display-xl text-paper-300 block leading-none">
                  03
                </span>
                <p className="text-body-md text-ink-700 mt-1">
                  Get detailed feedback on what to improve
                </p>
              </li>
            </ol>
          </div>
        </div>
      </div>
    </section>
  )
}

function FeedbackSection() {
  return (
    <section className="py-16 md:py-24">
      <div className="container-editorial">
        <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-8 lg:gap-16 items-start">
          <div className="order-2 lg:order-1">
            <img
              src="/Image3.jpg"
              alt="Close-up of classical piano score with dynamic markings"
              className="w-full object-cover rounded"
              style={{ aspectRatio: '3/2' }}
            />

            <div className="mt-8 space-y-4">
              <div className="border-l-2 border-clay-400 pl-4">
                <p className="text-label-sm font-medium text-clay-700 mb-1">
                  Sound Quality
                </p>
                <p className="text-body-sm text-ink-600 leading-relaxed">
                  Your dynamic range in measures 24-31 stays mostly at
                  mezzo-forte where Chopin's marking calls for a gradual
                  crescendo to fortissimo. Try exaggerating the build -- start
                  softer, arrive louder.
                </p>
              </div>
              <div className="border-l-2 border-clay-400 pl-4">
                <p className="text-label-sm font-medium text-clay-700 mb-1">
                  Technical Control
                </p>
                <p className="text-body-sm text-ink-600 leading-relaxed">
                  Pedal changes in the lyrical section are clean, but running
                  passages in bars 56-64 accumulate harmonic blur. Try
                  half-pedaling through the chromatic descent.
                </p>
              </div>
            </div>
          </div>

          <div className="order-1 lg:order-2">
            <h2 className="font-display text-display-sm md:text-display-md text-ink-900 mb-3">
              Real Feedback
            </h2>
            <p className="text-label-sm uppercase tracking-wider text-clay-600 mb-4">
              Chopin -- Ballade No. 1 in G minor
            </p>
            <p className="text-body-md text-ink-600 max-w-sm">
              Not "good job" or a letter grade. Specific, actionable feedback on
              exactly what to practice.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

function ResearchSection() {
  return (
    <section className="py-16 md:py-24">
      <div className="container-editorial">
        <div className="grid grid-cols-1 lg:grid-cols-[2fr_3fr] gap-8 lg:gap-16 items-start">
          <div>
            <h2 className="font-display text-display-sm md:text-display-md text-ink-900 mb-6">
              Built on Research
            </h2>
            <p className="text-body-lg text-ink-600 mb-8">
              Trained on thousands of hours of professional performances to hear
              what a great teacher hears. Published, peer-reviewed, and validated
              against real educator assessments.
            </p>

            <div className="grid grid-cols-2 gap-6 mb-8">
              <div>
                <p className="font-display text-display-md text-ink-900">55%</p>
                <p className="text-label-sm uppercase tracking-wider text-clay-600 mt-1">
                  More accurate
                </p>
              </div>
              <div>
                <p className="font-display text-display-md text-ink-900">30+</p>
                <p className="text-label-sm uppercase tracking-wider text-clay-600 mt-1">
                  Educator interviews
                </p>
              </div>
            </div>

            <a
              href="https://arxiv.org/abs/2601.19029"
              target="_blank"
              rel="noopener"
              className="inline-flex items-center gap-2 text-body-sm text-clay-600 underline underline-offset-2 hover:text-clay-700"
            >
              Published on arXiv
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </div>

          <div>
            <img
              src="/Image4.jpg"
              alt="Hands playing piano in dramatic light"
              className="w-full object-cover rounded"
              style={{ aspectRatio: '1/1' }}
            />
          </div>
        </div>
      </div>
    </section>
  )
}

function FinalCtaSection() {
  return (
    <section className="bg-clay-800 text-paper-50 py-20 md:py-32">
      <div className="container-editorial text-center">
        <h2 className="font-display text-display-md md:text-display-xl text-paper-50 mb-8">
          Ready to hear what your playing really sounds like?
        </h2>
        <a href="/analyze" className="btn-primary-inverted text-body-sm">
          Start Free
        </a>
      </div>
    </section>
  )
}
