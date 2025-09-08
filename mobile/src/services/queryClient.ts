import { QueryClient } from '@tanstack/react-query';

// Create a client
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
      retryDelay: 1000,
    },
  },
});

// Query keys factory
export const queryKeys = {
  // Auth
  auth: ['auth'] as const,
  user: (id: string) => ['user', id] as const,
  
  // Recordings
  recordings: ['recordings'] as const,
  recordingsList: (userId: string) => [...queryKeys.recordings, 'list', userId] as const,
  recording: (id: string) => [...queryKeys.recordings, 'detail', id] as const,
  
  // Analysis
  analysis: ['analysis'] as const,
  analysisResults: (recordingId: string) => [...queryKeys.analysis, recordingId] as const,
  
  // Progress
  progress: ['progress'] as const,
  userProgress: (userId: string) => [...queryKeys.progress, userId] as const,
  
  // Practice Sessions
  practiceSessions: ['practiceSessions'] as const,
  userSessions: (userId: string) => [...queryKeys.practiceSessions, userId] as const,
} as const;