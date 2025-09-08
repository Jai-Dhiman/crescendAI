import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { createMMKVStorage } from './mmkv';
import type { User, AuthTokens, AuthState } from '../types';

interface AuthStore extends AuthState {
  // Actions
  setUser: (user: User) => void;
  setTokens: (tokens: AuthTokens) => void;
  signIn: (user: User, tokens: AuthTokens) => void;
  signOut: () => void;
  setLoading: (isLoading: boolean) => void;
  updateUser: (updates: Partial<User>) => void;
  isTokenExpired: () => boolean;
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      tokens: null,
      isAuthenticated: false,
      isLoading: false,

      setUser: (user: User) => 
        set((state) => ({ 
          ...state, 
          user,
          isAuthenticated: !!user 
        })),

      setTokens: (tokens: AuthTokens) => 
        set((state) => ({ 
          ...state, 
          tokens 
        })),

      signIn: (user: User, tokens: AuthTokens) => 
        set(() => ({
          user,
          tokens,
          isAuthenticated: true,
          isLoading: false,
        })),

      signOut: () => 
        set(() => ({
          user: null,
          tokens: null,
          isAuthenticated: false,
          isLoading: false,
        })),

      setLoading: (isLoading: boolean) => 
        set((state) => ({ 
          ...state, 
          isLoading 
        })),

      updateUser: (updates: Partial<User>) => 
        set((state) => ({
          ...state,
          user: state.user ? { ...state.user, ...updates } : null,
        })),

      isTokenExpired: () => {
        const { tokens } = get();
        if (!tokens) return true;
        return new Date(tokens.expiresAt) <= new Date();
      },
    }),
    {
      name: 'auth-store',
      storage: createMMKVStorage(),
      partialize: (state) => ({
        user: state.user,
        tokens: state.tokens,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);