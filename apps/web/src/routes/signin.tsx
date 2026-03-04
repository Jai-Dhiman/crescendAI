import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { setAuth } from '../lib/auth'

export const Route = createFileRoute('/signin')({ component: SignInPage })

function SignInPage() {
  const navigate = useNavigate()

  function handleSignIn() {
    // Stub: simulate successful Apple Sign In
    setAuth({ name: 'Jai' })
    navigate({ to: '/app' })
  }

  return (
    <div className="relative h-screen w-full overflow-hidden">
      {/* Full-bleed background image */}
      <img
        src="/Image4.jpg"
        alt="Hands playing piano in warm light"
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Radial gradient overlay -- darkens edges, draws focus to center */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(45,41,38,0.4) 0%, rgba(45,41,38,0.85) 100%)',
        }}
      />

      {/* Floating sign-in card */}
      <div className="relative z-10 flex items-center justify-center h-full px-6">
        <div
          className="w-full max-w-sm bg-surface/80 backdrop-blur-xl border border-border px-8 py-14 text-center rounded-2xl"
          style={{ animation: 'fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both' }}
        >
          <h1 className="font-display text-display-sm text-cream">crescend</h1>

          <p className="mt-3 text-body-md text-text-secondary">
            A teacher for every pianist.
          </p>

          <button
            type="button"
            onClick={handleSignIn}
            className="mt-8 w-full bg-white text-black px-6 py-3 text-body-sm font-medium flex items-center justify-center gap-3 hover:bg-white/90 transition rounded-lg"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17.05 20.28c-.98.95-2.05.88-3.08.4-1.09-.5-2.08-.48-3.24 0-1.44.62-2.2.44-3.06-.4C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z" />
            </svg>
            Sign in with Apple
          </button>

          <p className="mt-6 text-body-xs text-text-tertiary">
            By signing in, you agree to our Terms of Service
          </p>
        </div>
      </div>
    </div>
  )
}
