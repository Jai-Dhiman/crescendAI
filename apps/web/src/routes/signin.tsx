import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useState, useEffect } from 'react'
import { useAuth } from '../lib/auth'
import { api } from '../lib/api'

export const Route = createFileRoute('/signin')({ component: SignInPage })

const APPLE_CLIENT_ID = 'ai.crescend.web'
const REDIRECT_URI = import.meta.env.PROD
  ? 'https://crescend.ai/signin'
  : 'http://localhost:3000/signin'

function SignInPage() {
  const navigate = useNavigate()
  const { setUser } = useAuth()
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (window.AppleID) {
      window.AppleID.auth.init({
        clientId: APPLE_CLIENT_ID,
        scope: 'name email',
        redirectURI: REDIRECT_URI,
        usePopup: true,
      })
    }
  }, [])

  async function handleSignIn() {
    if (!window.AppleID) {
      setError('Apple Sign In not available. Please try again.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const appleResponse = await window.AppleID.auth.signIn()
      const idToken = appleResponse.authorization.id_token

      // Decode the JWT to extract the subject (Apple user ID)
      const base64 = idToken.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')
      const payload = JSON.parse(atob(base64))
      const userId = payload.sub

      const email = appleResponse.user?.email ?? undefined
      const firstName = appleResponse.user?.name?.firstName
      const lastName = appleResponse.user?.name?.lastName
      const displayName = firstName ? [firstName, lastName].filter(Boolean).join(' ') : undefined

      const result = await api.auth.apple(idToken, userId, email, displayName)

      setUser({
        student_id: result.student_id,
        email: result.email,
        display_name: result.display_name,
      })

      navigate({ to: '/app' })
    } catch (err) {
      if (err instanceof Error && err.message.includes('popup_closed_by_user')) {
        return
      }
      console.error('Sign in failed:', err)
      setError('Sign in failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative h-screen w-full overflow-hidden">
      <img
        src="/Image4.jpg"
        alt="Hands playing piano in warm light"
        className="absolute inset-0 w-full h-full object-cover"
      />

      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(45,41,38,0.4) 0%, rgba(45,41,38,0.85) 100%)',
        }}
      />

      <div className="relative z-10 flex items-center justify-center h-full px-6">
        <div
          className="w-full max-w-sm bg-surface/80 backdrop-blur-xl border border-border px-8 py-14 text-center rounded-2xl"
          style={{ animation: 'fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both' }}
        >
          <h1 className="font-display text-display-sm text-cream">crescend</h1>

          <p className="mt-3 text-body-md text-text-secondary">
            A teacher for every pianist.
          </p>

          {error && (
            <p className="mt-4 text-body-sm text-red-400">{error}</p>
          )}

          <button
            type="button"
            onClick={handleSignIn}
            disabled={loading}
            className="mt-8 w-full bg-white text-black px-6 py-3 text-body-sm font-medium flex items-center justify-center gap-3 hover:bg-white/90 transition rounded-lg disabled:opacity-50"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M17.05 20.28c-.98.95-2.05.88-3.08.4-1.09-.5-2.08-.48-3.24 0-1.44.62-2.2.44-3.06-.4C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z" />
            </svg>
            {loading ? 'Signing in...' : 'Sign in with Apple'}
          </button>

          <p className="mt-6 text-body-xs text-text-tertiary">
            By signing in, you agree to our Terms of Service
          </p>
        </div>
      </div>
    </div>
  )
}
