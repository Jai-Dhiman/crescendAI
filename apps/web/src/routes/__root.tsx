import { HeadContent, Outlet, Scripts, createRootRoute } from '@tanstack/react-router'

import appCss from '../styles/app.css?url'

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'Crescend -- A Teacher for Every Pianist' },
      {
        name: 'description',
        content:
          'Record yourself playing piano. Get the feedback a great teacher would give you -- on your tone, your dynamics, your phrasing.',
      },
    ],
    links: [
      { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
      {
        rel: 'preconnect',
        href: 'https://fonts.gstatic.com',
        crossOrigin: 'anonymous',
      },
      {
        rel: 'stylesheet',
        href: 'https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap',
      },
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', type: 'image/png', href: '/crescendai.png' },
    ],
  }),
  component: RootDocument,
})

function RootDocument() {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        <div className="min-h-screen bg-paper-50 flex flex-col texture-grain">
          <Header />
          <main className="flex-1">
            <Outlet />
          </main>
          <Footer />
        </div>
        <Scripts />
      </body>
    </html>
  )
}

function Header() {
  return (
    <header className="py-10 lg:py-16">
      <div className="container-editorial">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-6 lg:gap-8 items-start text-center lg:text-left">
          <nav
            className="flex flex-row lg:flex-col items-center lg:items-start justify-center lg:justify-start gap-4 lg:gap-1"
            role="navigation"
            aria-label="Main navigation"
          >
            <a
              href="/#how-it-works"
              className="text-body-sm text-ink-600 hover:text-ink-900 transition-colors duration-200"
            >
              How It Works
            </a>
            <a
              href="/analyze"
              className="text-body-sm text-ink-600 hover:text-ink-900 transition-colors duration-200"
            >
              Analyze
            </a>
            <a
              href="https://arxiv.org/abs/2601.19029"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-body-sm text-ink-600 hover:text-ink-900 transition-colors duration-200"
            >
              Paper
              <svg
                className="w-3 h-3"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </nav>

          <a
            href="/"
            className="font-display text-display-xl lg:text-display-3xl text-ink-900 tracking-tight"
          >
            Crescend
          </a>

          <p className="text-body-sm text-ink-500 lg:text-right max-w-xs mx-auto lg:mx-0 lg:ml-auto">
            Record yourself playing. Get the feedback a great teacher would give
            you.
          </p>
        </div>
      </div>

      <div className="container-editorial mt-10 lg:mt-16">
        <hr className="editorial-rule" />
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="mt-auto py-12 lg:py-16">
      <div className="container-editorial">
        <hr className="editorial-rule mb-12 lg:mb-16" />

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-8 items-start text-center lg:text-left">
          <a
            href="/"
            className="font-display text-display-md text-ink-900 tracking-tight"
          >
            Crescend
          </a>

          <nav className="flex flex-row lg:flex-col items-center lg:items-start gap-3 justify-center text-body-sm">
            <a
              href="https://arxiv.org/abs/2601.19029"
              target="_blank"
              rel="noopener"
              className="text-body-sm text-ink-600 hover:text-ink-900 transition-colors duration-200"
            >
              Paper
            </a>
            <a
              href="/#how-it-works"
              className="text-body-sm text-ink-600 hover:text-ink-900 transition-colors duration-200"
            >
              How It Works
            </a>
            <a
              href="/analyze"
              className="text-body-sm text-ink-600 hover:text-ink-900 transition-colors duration-200"
            >
              Analyze
            </a>
          </nav>

          <div className="lg:text-right">
            <p className="text-body-xs text-ink-500 mb-1">
              Your recordings are yours. We don't store or train on your data.
            </p>
            <p className="text-label-sm text-ink-400">2026 Crescend</p>
          </div>
        </div>
      </div>
    </footer>
  )
}
