import { HeadContent, Outlet, Scripts, createRootRoute } from '@tanstack/react-router'

import appCss from '../styles/app.css?url'

export const Route = createRootRoute({
  head: () => ({
    meta: [
      {
        charSet: 'utf-8',
      },
      {
        name: 'viewport',
        content: 'width=device-width, initial-scale=1',
      },
      {
        title: 'CrescendAI - A teacher for every pianist',
      },
      {
        name: 'description',
        content:
          'CrescendAI evaluates musical expression from audio, providing personalized feedback grounded in piano pedagogy.',
      },
    ],
    links: [
      {
        rel: 'stylesheet',
        href: appCss,
      },
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
      <body className="bg-cream text-ink antialiased">
        <Outlet />
        <Scripts />
      </body>
    </html>
  )
}
