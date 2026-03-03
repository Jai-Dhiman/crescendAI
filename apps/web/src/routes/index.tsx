import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/')({ component: Home })

function Home() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center px-6">
      <div className="max-w-2xl text-center">
        <h1 className="font-display text-5xl md:text-6xl font-semibold tracking-tight mb-6">
          A teacher for every pianist.
        </h1>
        <p className="text-lg md:text-xl text-ink-60 leading-relaxed mb-4">
          CrescendAI listens to your playing and gives personalized feedback
          grounded in decades of piano pedagogy -- not just note accuracy, but
          the musical qualities that matter: dynamics, phrasing, pedaling, and
          interpretation.
        </p>
        <p className="text-lg md:text-xl text-ink-60 leading-relaxed mb-10">
          Practice smarter. Play more expressively.
        </p>
        <p className="text-sm font-medium tracking-wide uppercase text-ink-60">
          Coming soon to iOS
        </p>
      </div>
    </main>
  )
}
