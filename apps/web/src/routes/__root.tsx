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
        href: 'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap',
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
      <body className="bg-espresso text-text-primary font-sans">
        <Header />
        <main>
          <Outlet />
        </main>
        <Footer />
        <Scripts />
      </body>
    </html>
  )
}

function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 backdrop-blur-xl bg-espresso/80">
      <div className="max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between h-16">
        <a href="/" className="font-display text-lg text-cream tracking-tight">
          crescend
        </a>
        <a
          href="/analyze"
          className="bg-cream text-espresso rounded-full px-6 py-2 text-body-sm font-medium hover:brightness-110 transition"
        >
          Start Practicing
        </a>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="py-12 lg:py-16">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-body-xs text-text-tertiary">
          <a href="/" className="font-display text-sm text-cream tracking-tight">
            crescend
          </a>
          <p>
            Built on published research.{' '}
            <a
              href="https://arxiv.org/abs/2601.19029"
              target="_blank"
              rel="noopener"
              className="text-text-secondary underline underline-offset-2 hover:text-cream transition-colors"
            >
              Read the paper
            </a>
          </p>
          <p>2026</p>
        </div>
      </div>
    </footer>
  )
}
