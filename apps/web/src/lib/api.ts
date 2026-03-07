const API_BASE = import.meta.env.PROD
  ? 'https://api.crescend.ai'
  : 'http://localhost:8787'

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText })) as Record<string, string>
    throw new ApiError(response.status, body.error ?? response.statusText)
  }

  return response.json()
}

export interface AuthUser {
  student_id: string
  email: string | null
  display_name: string | null
}

export interface AuthResult {
  student_id: string
  email: string | null
  display_name: string | null
  is_new_user: boolean
}

// --- Chat types ---

export interface ConversationSummary {
  id: string
  title: string | null
  updated_at: string
}

export interface MessageRow {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export interface ConversationWithMessages {
  conversation: {
    id: string
    title: string | null
    created_at: string
  }
  messages: MessageRow[]
}

export interface ChatStreamEvent {
  type: 'start' | 'delta' | 'done'
  conversation_id?: string
  message_id?: string
  text?: string
}

export const api = {
  auth: {
    apple(identityToken: string, userId: string, email?: string, displayName?: string): Promise<AuthResult> {
      return request('/api/auth/apple', {
        method: 'POST',
        body: JSON.stringify({
          identity_token: identityToken,
          user_id: userId,
          email,
          display_name: displayName,
        }),
      })
    },

    me(): Promise<AuthUser> {
      return request('/api/auth/me')
    },

    signout(): Promise<void> {
      return request('/api/auth/signout', { method: 'POST' })
    },

    debug(): Promise<AuthResult> {
      return request('/api/auth/debug', { method: 'POST' })
    },
  },

  chat: {
    async send(
      message: string,
      conversationId: string | null,
      onEvent: (event: ChatStreamEvent) => void,
    ): Promise<void> {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          message,
        }),
      })

      if (!response.ok) {
        const body = await response.json().catch(() => ({ error: response.statusText })) as Record<string, string>
        throw new ApiError(response.status, body.error ?? response.statusText)
      }

      if (!response.body) throw new Error('Response body is empty')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event: ChatStreamEvent = JSON.parse(line.slice(6))
                onEvent(event)
              } catch {
                // Skip unparseable lines
              }
            }
          }
        }
      } finally {
        reader.releaseLock()
      }
    },

    list(): Promise<{ conversations: ConversationSummary[] }> {
      return request('/api/conversations')
    },

    get(conversationId: string): Promise<ConversationWithMessages> {
      return request(`/api/conversations/${conversationId}`)
    },

    async delete(conversationId: string): Promise<void> {
      const response = await fetch(`${API_BASE}/api/conversations/${conversationId}`, {
        method: 'DELETE',
        credentials: 'include',
      })
      if (!response.ok && response.status !== 204) {
        throw new ApiError(response.status, 'Failed to delete conversation')
      }
    },
  },
}
