import { useEffect, useRef, useState } from 'react'
import { HeadContent, Outlet, Scripts, createRootRoute, useRouterState } from '@tanstack/react-router'
import { AuthProvider } from '../lib/auth'

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
  const pathname = useRouterState({ select: (s) => s.location.pathname })
  const isAppShell = pathname === '/signin' || pathname.startsWith('/app')

  return (
    <html lang="en">
      <head>
        <HeadContent />
        <script
          type="text/javascript"
          src="https://appleid.cdn-apple.com/appleauth/static/jsapi/appleid/1/en_US/appleid.auth.js"
        />
      </head>
      <body className="bg-espresso text-text-primary font-sans">
        <AuthProvider>
          {!isAppShell && <Header />}
          <main>
            <Outlet />
          </main>
          {!isAppShell && <Footer />}
        </AuthProvider>
        <Scripts />
      </body>
    </html>
  )
}

function Header() {
  const [hidden, setHidden] = useState(false)
  const lastScrollY = useRef(0)

  useEffect(() => {
    function onScroll() {
      const y = window.scrollY
      setHidden(y > 64 && y > lastScrollY.current)
      lastScrollY.current = y
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className="fixed top-0 left-0 right-0 z-50 transition-transform duration-300"
      style={{ transform: hidden ? 'translateY(-100%)' : 'translateY(0)' }}
    >
      <div className="max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between h-16">
        <a href="/" className="font-display text-lg text-cream tracking-tight">
          crescend
        </a>
        <a href="/signin" className="font-display text-body-sm text-cream hover:text-text-secondary transition-colors">
          Sign In
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
