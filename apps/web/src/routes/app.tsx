import { useEffect, useRef, useState } from 'react'
import { createFileRoute, useNavigate, redirect } from '@tanstack/react-router'
import { MessageSquare, Mic, Plus } from 'lucide-react'
import { getAuth, clearAuth, isAuthenticated } from '../lib/auth'

export const Route = createFileRoute('/app')({
  beforeLoad: () => {
    if (!isAuthenticated()) {
      throw redirect({ to: '/signin' })
    }
  },
  component: AppPage,
})

function AppPage() {
  const user = getAuth()
  const navigate = useNavigate()
  const [showProfile, setShowProfile] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!showProfile) return
    function handleClick(e: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setShowProfile(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showProfile])

  function handleSignOut() {
    clearAuth()
    navigate({ to: '/' })
  }

  // Time-aware greeting
  const hour = new Date().getHours()
  let greeting = 'Good morning'
  if (hour >= 12 && hour < 17) greeting = 'Good afternoon'
  else if (hour >= 17) greeting = 'Good evening'

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Thin icon sidebar */}
      <aside className="w-12 shrink-0 border-r border-border flex flex-col items-center py-4 gap-1">
        <SidebarButton icon={<Plus size={18} />} label="New Chat" />
        <SidebarButton icon={<MessageSquare size={18} />} label="Chats" />
        <SidebarButton icon={<MetronomeIcon />} label="Metronome" />
      </aside>

      {/* Main content area */}
      <div className="flex-1 relative flex flex-col">
        {/* Profile button -- top right */}
        <div ref={profileRef} className="absolute top-4 right-4 z-20">
          <button
            type="button"
            onClick={() => setShowProfile(!showProfile)}
            className="w-8 h-8 bg-surface border border-border flex items-center justify-center text-body-sm text-cream font-medium hover:bg-surface-2 transition"
          >
            {user?.name?.charAt(0).toUpperCase() ?? '?'}
          </button>

          {showProfile && (
            <div className="absolute right-0 top-10 bg-surface border border-border py-1 min-w-[140px]">
              <button
                type="button"
                onClick={handleSignOut}
                className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition"
              >
                Sign Out
              </button>
            </div>
          )}
        </div>

        {/* Centered home content */}
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-2xl text-center">
            <h1 className="font-display text-display-md text-cream">
              {greeting}, {user?.name ?? 'there'}.
            </h1>

            {/* Input box */}
            <div className="mt-8 bg-surface border border-border flex items-center">
              <input
                type="text"
                placeholder="What are you practicing today?"
                className="flex-1 bg-transparent px-5 py-4 text-body-md text-cream placeholder:text-text-tertiary outline-none"
              />
              <button
                type="button"
                className="px-4 py-4 text-text-secondary hover:text-cream transition"
                aria-label="Start recording"
              >
                <Mic size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function SidebarButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button
      type="button"
      className="w-10 h-10 flex items-center justify-center text-text-secondary hover:text-cream hover:bg-surface transition group relative"
      aria-label={label}
    >
      {icon}
      {/* Tooltip */}
      <span className="absolute left-full ml-2 px-2 py-1 bg-surface-2 text-body-xs text-cream whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
        {label}
      </span>
    </button>
  )
}

function MetronomeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2v10" />
      <path d="M5 21h14" />
      <path d="M7.5 21L10 6h4l2.5 15" />
      <path d="M12 12l4-4" />
    </svg>
  )
}
