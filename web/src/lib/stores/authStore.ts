import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { browser } from '$app/environment';
import type { AuthState, AuthTokens, User } from '../types';
import { storage } from './storage';
import { tokenManager, withTokenRefresh } from '../services/tokenManager';
import { apiEndpoints } from '../config/environment';

interface AuthError {
  code: string;
  message: string;
  details?: any;
}

interface AuthStore extends AuthState {
  // Authentication actions
  signIn: (user: User, tokens: AuthTokens) => Promise<void>;
  signOut: () => Promise<void>;
  refreshAuth: () => Promise<boolean>;
  
  // User management
  updateUser: (user: Partial<User>) => void;
  
  // State management
  setLoading: (loading: boolean) => void;
  setError: (error: AuthError | null) => void;
  
  // Token management
  getAccessToken: () => string | null;
  hasValidTokens: () => boolean;
  
  // Additional state
  error: AuthError | null;
  lastRefresh: number | null;
}

/**
 * Create enhanced authentication store with secure token handling
 */
const createAuthStore = () => create<AuthStore>()(
  persist(
    (set, get) => ({
      // Initial state
      user: null,
      tokens: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
      lastRefresh: null,

      /**
       * Sign in user with secure token storage
       */
      signIn: async (user: User, tokens: AuthTokens) => {
        try {
          set({ isLoading: true, error: null });

          // Store tokens securely using TokenManager
          if (tokenManager) {
            await tokenManager.storeTokens(tokens);
          }

          set({
            user,
            tokens,
            isAuthenticated: true,
            isLoading: false,
            error: null,
            lastRefresh: Date.now(),
          });

          // Start listening for token refresh events
          if (browser) {
            window.addEventListener('auth:token-refresh-failed', (event) => {
              const customEvent = event as CustomEvent;
              console.error('Token refresh failed, signing out:', customEvent.detail);
              get().signOut();
            });
          }

        } catch (error) {
          const authError: AuthError = {
            code: 'SIGNIN_FAILED',
            message: error instanceof Error ? error.message : 'Sign in failed',
            details: error,
          };

          set({
            isLoading: false,
            error: authError,
            isAuthenticated: false,
          });

          throw authError;
        }
      },

      /**
       * Sign out user and clear all authentication data
       */
      signOut: async () => {
        try {
          set({ isLoading: true, error: null });

          // Clear tokens from secure storage
          if (tokenManager) {
            tokenManager.clearTokens();
          }

          // Make API call to invalidate tokens on server
          const accessToken = get().getAccessToken();
          if (accessToken) {
            try {
              await fetch(apiEndpoints.auth.logout, {
                method: 'POST',
                headers: {
                  'Authorization': `Bearer ${accessToken}`,
                  'Content-Type': 'application/json',
                },
              });
            } catch (error) {
              console.warn('Server logout failed (continuing with local logout):', error);
            }
          }

          set({
            user: null,
            tokens: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
            lastRefresh: null,
          });

        } catch (error) {
          console.error('Sign out error:', error);
          // Force clear state even if API call fails
          set({
            user: null,
            tokens: null,
            isAuthenticated: false,
            isLoading: false,
            error: null,
            lastRefresh: null,
          });
        }
      },

      /**
       * Refresh authentication tokens
       */
      refreshAuth: async () => {
        try {
          set({ isLoading: true, error: null });

          if (!tokenManager) {
            throw new Error('Token manager not available');
          }

          const newTokens = await tokenManager.refreshTokens();
          if (!newTokens) {
            throw new Error('Token refresh failed');
          }

          set({
            tokens: newTokens,
            isLoading: false,
            lastRefresh: Date.now(),
          });

          return true;
        } catch (error) {
          const authError: AuthError = {
            code: 'REFRESH_FAILED',
            message: error instanceof Error ? error.message : 'Token refresh failed',
            details: error,
          };

          set({
            error: authError,
            isLoading: false,
          });

          // If refresh fails, sign out
          await get().signOut();
          return false;
        }
      },

      /**
       * Update user information
       */
      updateUser: (updates: Partial<User>) => {
        const currentUser = get().user;
        if (!currentUser) return;

        const updatedUser = { ...currentUser, ...updates };
        set({ user: updatedUser });
      },

      /**
       * Set loading state
       */
      setLoading: (isLoading: boolean) => {
        set({ isLoading });
      },

      /**
       * Set error state
       */
      setError: (error: AuthError | null) => {
        set({ error });
      },

      /**
       * Get current access token
       */
      getAccessToken: () => {
        return tokenManager?.getAccessToken() || null;
      },

      /**
       * Check if user has valid tokens
       */
      hasValidTokens: () => {
        return tokenManager?.hasValidTokens() || false;
      },
    }),
    {
      name: 'auth-user-data', // Only store user data, not tokens
      storage: {
        getItem: (name) => {
          const str = storage.getString(name);
          return str ? JSON.parse(str) : null;
        },
        setItem: (name, value) => {
          // Only persist user data, not sensitive tokens
          const { tokens, ...safeToPersist } = value;
          storage.setString(name, JSON.stringify(safeToPersist));
        },
        removeItem: (name) => {
          storage.delete(name);
        },
      },
      // Define what to persist (exclude sensitive data)
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
        lastRefresh: state.lastRefresh,
        // Explicitly exclude tokens and errors from persistence
      }),
    }
  )
);

export const useAuthStore = createAuthStore();

// Initialize authentication on app load
if (browser) {
  const store = useAuthStore.getState();
  
  // Check if we have stored tokens and restore authentication state
  if (tokenManager?.hasValidTokens()) {
    const currentUser = store.user;
    if (currentUser) {
      // Restore authentication state if we have both user and valid tokens
      store.setLoading(false);
      useAuthStore.setState({ isAuthenticated: true });
      
      console.info('✅ Authentication restored from secure storage');
    }
  } else if (store.isAuthenticated) {
    // Clear authentication state if tokens are invalid
    console.warn('⚠️ Invalid tokens found, clearing authentication state');
    store.signOut();
  }
}

/**
 * Hook for making authenticated API requests
 */
export async function authenticatedRequest<T>(
  requestFn: (token: string) => Promise<T>
): Promise<T> {
  const store = useAuthStore.getState();
  
  if (!store.isAuthenticated) {
    throw new Error('User not authenticated');
  }

  if (!tokenManager) {
    throw new Error('Token manager not available');
  }

  return withTokenRefresh(requestFn);
}

/**
 * Type export for external use
 */
export type { AuthError };