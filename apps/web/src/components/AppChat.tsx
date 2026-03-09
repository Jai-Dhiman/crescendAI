import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from '@tanstack/react-router'
import {
  SidebarSimple,
  PlusCircle,
  ChatCircle,
  Trash,
  MagnifyingGlass,
} from '@phosphor-icons/react'
import { useAuth } from '../lib/auth'
import { api } from '../lib/api'
import type { ConversationSummary, MessageRow, ChatStreamEvent } from '../lib/api'
import { ChatMessages } from './ChatMessages'
import { ChatInput } from './ChatInput'
import { RecordingOverlay } from './RecordingOverlay'
import { usePracticeSession } from '../hooks/usePracticeSession'

interface AppChatProps {
  initialConversationId?: string
}

export default function AppChat({ initialConversationId }: AppChatProps) {
  const { user, isLoading, isAuthenticated, signOut } = useAuth()
  const navigate = useNavigate()
  const [showProfile, setShowProfile] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)

  // Chat state
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    initialConversationId ?? null,
  )
  const [messages, setMessages] = useState<MessageRow[]>([])
  const [streamingContent, setStreamingContent] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)

  // Practice recording
  const practice = usePracticeSession()

  function handleRecord() {
    practice.start()
  }

  // When practice summary arrives, post it to chat
  useEffect(() => {
    if (practice.summary) {
      const summaryMsg: MessageRow = {
        id: `practice-${Date.now()}`,
        role: 'assistant',
        content: practice.summary,
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, summaryMsg])
    }
  }, [practice.summary])

  // Redirect if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate({ to: '/signin' })
    }
  }, [isLoading, isAuthenticated, navigate])

  // Load conversations on mount
  useEffect(() => {
    if (!isAuthenticated) return
    api.chat.list().then(({ conversations }) => {
      setConversations(conversations)
    }).catch((e) => {
      console.error('Failed to load conversations:', e)
    })
  }, [isAuthenticated])

  // Load initial conversation from URL param
  useEffect(() => {
    if (initialConversationId && isAuthenticated) {
      loadConversation(initialConversationId)
    }
  }, [initialConversationId, isAuthenticated])

  // Click outside to close profile dropdown
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

  async function handleSignOut() {
    await signOut()
    navigate({ to: '/' })
  }

  const loadConversation = useCallback(async (id: string) => {
    try {
      const data = await api.chat.get(id)
      setActiveConversationId(id)
      setMessages(data.messages)
      navigate({
        to: '/app/c/$conversationId',
        params: { conversationId: id },
        replace: true,
      })
    } catch (e) {
      console.error('Failed to load conversation:', e)
    }
  }, [navigate])

  function handleNewChat() {
    setActiveConversationId(null)
    setMessages([])
    setStreamingContent(null)
    navigate({ to: '/app', replace: true })
  }

  async function handleDeleteConversation(id: string) {
    try {
      await api.chat.delete(id)
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (activeConversationId === id) {
        handleNewChat()
      }
    } catch (e) {
      console.error('Failed to delete conversation:', e)
    }
  }

  async function handleSend(message: string) {
    if (isStreaming) return

    const tempUserMsg: MessageRow = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, tempUserMsg])
    setStreamingContent('')
    setIsStreaming(true)

    let newConversationId: string | null = null

    try {
      await api.chat.send(message, activeConversationId, (event: ChatStreamEvent) => {
        switch (event.type) {
          case 'start':
            if (event.conversation_id && !activeConversationId) {
              newConversationId = event.conversation_id
              setActiveConversationId(event.conversation_id)
            }
            break
          case 'delta':
            if (event.text) {
              setStreamingContent((prev) => (prev ?? '') + event.text)
            }
            break
          case 'done':
            setStreamingContent((current) => {
              if (current !== null) {
                const assistantMsg: MessageRow = {
                  id: event.message_id ?? `msg-${Date.now()}`,
                  role: 'assistant',
                  content: current,
                  created_at: new Date().toISOString(),
                }
                setMessages((prev) => [...prev, assistantMsg])
              }
              return null
            })
            setIsStreaming(false)
            break
        }
      })

      if (newConversationId) {
        navigate({
          to: '/app/c/$conversationId',
          params: { conversationId: newConversationId },
          replace: true,
        })
      }

      const { conversations: updated } = await api.chat.list()
      setConversations(updated)
    } catch (e) {
      console.error('Chat send failed:', e)
      setStreamingContent(null)
      setIsStreaming(false)
    }
  }

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <p className="text-text-secondary text-body-md">Loading...</p>
      </div>
    )
  }

  const hour = new Date().getHours()
  let greeting = 'Good morning'
  if (hour >= 12 && hour < 17) greeting = 'Good afternoon'
  else if (hour >= 17) greeting = 'Good evening'

  const hasMessages = messages.length > 0 || streamingContent !== null
  const userInitial = user?.display_name?.charAt(0).toUpperCase() ?? user?.email?.charAt(0).toUpperCase() ?? '?'

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`shrink-0 border-r border-border flex flex-col py-4 transition-all duration-200 overflow-hidden ${
          sidebarOpen ? 'w-56' : 'w-12'
        }`}
      >
        <div className="flex items-center h-10 px-2 mb-2">
          {sidebarOpen ? (
            <>
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <img src="/icon_nobackground.png" alt="crescend" className="w-7 h-7 shrink-0" />
                <span className="font-display text-body-md text-cream truncate">crescend</span>
              </div>
              <button
                type="button"
                onClick={() => setSidebarOpen(false)}
                className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
                aria-label="Collapse sidebar"
              >
                <SidebarSimple size={18} />
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="w-8 h-8 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition mx-auto"
              aria-label="Expand sidebar"
            >
              <SidebarSimple size={20} />
            </button>
          )}
        </div>

        <div className="flex flex-col items-center">
          <div style={{ width: '100%' }}>
            <SidebarButton
              icon={<PlusCircle size={24} weight="fill" className="text-accent" />}
              label="New Chat"
              expanded={sidebarOpen}
              onClick={handleNewChat}
            />
          </div>
          <div className="w-full">
            <SidebarButton
              icon={<MagnifyingGlass size={20} />}
              label="Search"
              expanded={sidebarOpen}
              onClick={() => {}}
            />
          </div>
        </div>

        {/* Conversation list */}
        {sidebarOpen && (
          <div className="mt-4 flex-1 overflow-y-auto px-2">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group flex items-center gap-2 rounded-lg px-3 py-2 cursor-pointer text-body-sm transition ${
                  conv.id === activeConversationId
                    ? 'bg-surface text-cream'
                    : 'text-text-secondary hover:text-cream hover:bg-surface'
                }`}
                onClick={() => loadConversation(conv.id)}
                onKeyDown={(e) => e.key === 'Enter' && loadConversation(conv.id)}
                role="button"
                tabIndex={0}
              >
                <ChatCircle size={16} className="shrink-0" />
                <span className="flex-1 truncate">
                  {conv.title ?? 'New conversation'}
                </span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteConversation(conv.id)
                  }}
                  className="opacity-0 group-hover:opacity-100 shrink-0 text-text-tertiary hover:text-cream transition"
                  aria-label="Delete conversation"
                >
                  <Trash size={14} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Profile at sidebar bottom */}
        <div ref={profileRef} className="mt-auto pt-2 px-2 relative">
          <button
            type="button"
            onClick={() => setShowProfile(!showProfile)}
            className={`flex items-center gap-3 h-10 rounded-lg transition hover:bg-surface ${
              sidebarOpen ? 'w-full px-2' : 'justify-center mx-auto'
            }`}
          >
            <span className="shrink-0 w-8 h-8 bg-surface border border-border rounded-full flex items-center justify-center text-body-sm text-cream font-medium">
              {userInitial}
            </span>
            {sidebarOpen && (
              <div className="flex flex-col items-start min-w-0">
                <span className="text-body-sm text-cream truncate">
                  {user?.display_name ?? user?.email ?? 'User'}
                </span>
                <span className="text-body-xs text-text-tertiary">Pianist</span>
              </div>
            )}
          </button>

          {showProfile && (
            <div className="absolute left-2 bottom-full mb-2 bg-surface border border-border rounded-lg py-1 min-w-[140px] z-20">
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
      </aside>

      {/* Main content */}
      <div className="flex-1 relative flex flex-col">
        {practice.state !== 'idle' && (
          <RecordingOverlay
            state={practice.state}
            elapsedSeconds={practice.elapsedSeconds}
            observations={practice.observations}
            analyserNode={practice.analyserNode}
            error={practice.error}
            onStop={practice.stop}
          />
        )}
        {!hasMessages ? (
          <div className="flex-1 flex flex-col items-center justify-center px-6 pb-[22vh]">
            <img src="/icon_nobackground.png" alt="" className="w-20 h-20 opacity-50 mb-6" />
            <h1 className="font-display text-display-md text-cream mb-8">
              {greeting}.
            </h1>
            <ChatInput
              onSend={handleSend}
              onRecord={handleRecord}
              disabled={isStreaming || practice.state === 'recording'}
              placeholder="What are you practicing today?"
              centered={true}
            />
          </div>
        ) : (
          <>
            <ChatMessages messages={messages} streamingContent={streamingContent} />
            <ChatInput
              onSend={handleSend}
              onRecord={handleRecord}
              disabled={isStreaming || practice.state === 'recording'}
              placeholder="Message your teacher..."
              centered={false}
            />
          </>
        )}
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
        expanded ? 'w-[calc(100%-16px)] mx-2 px-3 h-10 gap-3' : 'w-10 h-10 justify-center mx-auto'
      }`}
      aria-label={label}
    >
      <span className="shrink-0 w-6 flex items-center justify-center">{icon}</span>
      {expanded && (
        <span className="text-body-sm whitespace-nowrap">{label}</span>
      )}
      {!expanded && (
        <span className="absolute left-full ml-2 px-2 py-1 bg-surface-2 rounded text-body-xs text-cream whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
          {label}
        </span>
      )}
    </button>
  )
}
