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
    const body = await response.json().catch(() => ({ error: response.statusText }))
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

export const api = {
  auth: {
    apple(identityToken: string, userId: string, email?: string): Promise<AuthResult> {
      return request('/api/auth/apple', {
        method: 'POST',
        body: JSON.stringify({
          identity_token: identityToken,
          user_id: userId,
          email,
        }),
      })
    },

    me(): Promise<AuthUser> {
      return request('/api/auth/me')
    },

    signout(): Promise<void> {
      return request('/api/auth/signout', { method: 'POST' })
    },
  },
}
