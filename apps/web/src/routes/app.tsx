import { useEffect, useRef, useState } from 'react'
import { createFileRoute, useNavigate, redirect } from '@tanstack/react-router'
import { ChevronLeft, ChevronRight, MessageSquare, Mic, Plus } from 'lucide-react'
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
  const [sidebarOpen, setSidebarOpen] = useState(false)
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
      {/* Collapsible sidebar */}
      <aside
        className={`shrink-0 border-r border-border flex flex-col items-center py-4 transition-all duration-200 ${
          sidebarOpen ? 'w-56' : 'w-12'
        }`}
      >
        {/* Toggle button */}
        <SidebarButton
          icon={sidebarOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
          label={sidebarOpen ? 'Collapse' : 'Expand'}
          expanded={sidebarOpen}
          onClick={() => setSidebarOpen(!sidebarOpen)}
        />

        <div className="mt-2 flex flex-col gap-1 w-full items-center">
          <SidebarButton icon={<Plus size={18} />} label="New Chat" expanded={sidebarOpen} />
          <SidebarButton icon={<MessageSquare size={18} />} label="Chats" expanded={sidebarOpen} />
          <SidebarButton icon={<MetronomeIcon />} label="Metronome" expanded={sidebarOpen} />
        </div>
      </aside>

      {/* Main content area */}
      <div className="flex-1 relative flex flex-col">
        {/* Profile button -- top right */}
        <div ref={profileRef} className="absolute top-4 right-4 z-20">
          <button
            type="button"
            onClick={() => setShowProfile(!showProfile)}
            className="w-8 h-8 bg-surface border border-border rounded-full flex items-center justify-center text-body-sm text-cream font-medium hover:bg-surface-2 transition"
          >
            {user?.name?.charAt(0).toUpperCase() ?? '?'}
          </button>

          {showProfile && (
            <div className="absolute right-0 top-10 bg-surface border border-border rounded-lg py-1 min-w-[140px]">
              <button
                type="button"
                onClick={handleSignOut}
                className="w-full text-left px-4 py-2 text-body-sm text-text-secondary hover:text-cream hover:bg-surface-2 transition rounded-lg"
              >
                Sign Out
              </button>
            </div>
          )}
        </div>

        {/* Content positioned upper-third */}
        <div className="flex-1 flex flex-col justify-start pt-[20vh] px-6">
          <div className="w-full max-w-2xl mx-auto text-center">
            <h1 className="font-display text-display-md text-cream">
              {greeting}, {user?.name ?? 'there'}.
            </h1>

            {/* Input box */}
            <div className="mt-8 bg-surface border border-border rounded-2xl flex items-center">
              <input
                type="text"
                placeholder="What are you practicing today?"
                className="flex-1 bg-transparent px-5 py-4 text-body-md text-cream placeholder:text-text-tertiary outline-none"
              />
            </div>

            {/* Action button */}
            <div className="mt-5">
              <button
                type="button"
                className="bg-cream text-espresso px-6 py-3 text-body-sm font-medium rounded-full hover:brightness-110 transition inline-flex items-center gap-2.5"
              >
                <Mic size={16} />
                Start Recording
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function SidebarButton({
  icon,
  label,
  expanded = false,
  onClick,
}: {
  icon: React.ReactNode
  label: string
  expanded?: boolean
  onClick?: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center text-text-secondary hover:text-cream hover:bg-surface transition group relative rounded-lg ${
        expanded ? 'w-[calc(100%-16px)] mx-2 px-3 h-10 gap-3' : 'w-10 h-10 justify-center'
      }`}
      aria-label={label}
    >
      <span className="shrink-0">{icon}</span>
      {expanded && (
        <span className="text-body-sm whitespace-nowrap">{label}</span>
      )}
      {/* Tooltip -- only when collapsed */}
      {!expanded && (
        <span className="absolute left-full ml-2 px-2 py-1 bg-surface-2 rounded text-body-xs text-cream whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
          {label}
        </span>
      )}
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
