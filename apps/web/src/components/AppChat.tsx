import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from '@tanstack/react-router'
import {
  SidebarSimple,
  PlusCircle,
  ChatCircle,
  Trash,
} from '@phosphor-icons/react'
import { useAuth } from '../lib/auth'
import { api } from '../lib/api'
import type { ConversationSummary, MessageRow, ChatStreamEvent } from '../lib/api'
import { ChatMessages } from './ChatMessages'
import { ChatInput } from './ChatInput'

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

    // Optimistically add user message
    const tempUserMsg: MessageRow = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, tempUserMsg])
    setStreamingContent('')
    setIsStreaming(true)

    // Capture the conversation ID locally so we can update URL after streaming
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

      // Update URL after streaming completes (not during)
      if (newConversationId) {
        navigate({
          to: '/app/c/$conversationId',
          params: { conversationId: newConversationId },
          replace: true,
        })
      }

      // Refresh conversation list (to get new/updated titles)
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

  // Time-aware greeting
  const hour = new Date().getHours()
  let greeting = 'Good morning'
  if (hour >= 12 && hour < 17) greeting = 'Good afternoon'
  else if (hour >= 17) greeting = 'Good evening'

  const hasMessages = messages.length > 0 || streamingContent !== null

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`shrink-0 border-r border-border flex flex-col py-4 transition-all duration-200 ${
          sidebarOpen ? 'w-56' : 'w-12'
        }`}
      >
        <div className="flex flex-col items-center">
          <SidebarButton
            icon={<SidebarSimple size={20} />}
            label={sidebarOpen ? 'Collapse' : 'Expand'}
            expanded={sidebarOpen}
            onClick={() => setSidebarOpen(!sidebarOpen)}
          />
          <div className="mt-2 w-full">
            <SidebarButton
              icon={<PlusCircle size={20} />}
              label="New Chat"
              expanded={sidebarOpen}
              onClick={handleNewChat}
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
      </aside>

      {/* Main content */}
      <div className="flex-1 relative flex flex-col">
        {/* Profile button */}
        <div ref={profileRef} className="absolute top-4 right-4 z-20">
          <button
            type="button"
            onClick={() => setShowProfile(!showProfile)}
            className="w-8 h-8 bg-surface border border-border rounded-full flex items-center justify-center text-body-sm text-cream font-medium hover:bg-surface-2 transition"
          >
            {user?.display_name?.charAt(0).toUpperCase() ?? user?.email?.charAt(0).toUpperCase() ?? '?'}
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

        {/* Empty state or chat messages */}
        {!hasMessages ? (
          <div className="flex-1 flex flex-col justify-start pt-[28vh] px-6">
            <div className="w-full max-w-2xl mx-auto text-center">
              <h1 className="font-display text-display-md text-cream">
                {greeting}.
              </h1>
            </div>
          </div>
        ) : (
          <ChatMessages messages={messages} streamingContent={streamingContent} />
        )}

        {/* Input bar */}
        <ChatInput
          onSend={handleSend}
          disabled={isStreaming}
          placeholder={hasMessages ? 'Message your teacher...' : 'What are you practicing today?'}
        />
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
      <span className="shrink-0">{icon}</span>
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
