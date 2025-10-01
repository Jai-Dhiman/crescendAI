import type {
  Analysis,
  ApiResponse,
  PaginatedResponse,
  PracticeSession,
  Progress,
  Recording,
  User,
  AuthTokens,
} from '../types';

import { clientConfig, apiEndpoints } from '../config/environment';

// API Configuration using secure environment management
export const API_CONFIG = {
  BASE_URL: clientConfig.apiUrl,
  VERSION: "v1",
  get FULL_URL() {
    return apiEndpoints.v1;
  }
} as const;

// Abstract API client interface that can be implemented by platform-specific clients
export interface ApiClientInterface {
  // Auth endpoints
  authenticateWithGoogle(token: string): Promise<ApiResponse<{ user: User; tokens: AuthTokens }>>;
  refreshToken(refreshToken: string): Promise<ApiResponse<{ tokens: AuthTokens }>>;
  signOut(): Promise<ApiResponse<{}>>;

  // User endpoints
  getUserProfile(userId: string): Promise<ApiResponse<User>>;
  updateUserProfile(userId: string, updates: Partial<User>): Promise<ApiResponse<User>>;

  // Recordings endpoints
  getUserRecordings(userId: string, page?: number, limit?: number): Promise<PaginatedResponse<Recording>>;
  getRecording(recordingId: string): Promise<ApiResponse<Recording>>;
  createRecording(recordingData: Omit<Recording, "id" | "createdAt" | "updatedAt">): Promise<ApiResponse<Recording>>;
  updateRecording(recordingId: string, updates: Partial<Recording>): Promise<ApiResponse<Recording>>;
  deleteRecording(recordingId: string): Promise<ApiResponse<{}>>;
  uploadRecording(recordingId: string, audioFile: File | Blob): Promise<ApiResponse<{ uploadUrl: string }>>;

  // Analysis endpoints
  getAnalysis(recordingId: string): Promise<ApiResponse<Analysis>>;
  requestAnalysis(recordingId: string): Promise<ApiResponse<Analysis>>;

  // Progress endpoints
  getUserProgress(userId: string): Promise<ApiResponse<Progress>>;

  // Practice Sessions endpoints
  getUserPracticeSessions(userId: string, page?: number, limit?: number): Promise<PaginatedResponse<PracticeSession>>;
}

// Common API utilities
export const ApiUtils = {
  // Build query string from parameters
  buildQueryString: (params: Record<string, string | number | undefined>): string => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    return queryString ? `?${queryString}` : '';
  },

  // Common headers for API requests
  getCommonHeaders: (accessToken?: string): HeadersInit => {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (accessToken) {
      headers.Authorization = `Bearer ${accessToken}`;
    }

    return headers;
  },

  // Handle API response
  handleApiResponse: async <T>(response: Response): Promise<ApiResponse<T>> => {
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.message || `HTTP ${response.status}: ${response.statusText}`
      );
    }

    const data = await response.json();
    return data;
  },

  // Transform array response to paginated response
  toPaginatedResponse: <T>(
    data: T[],
    page: number,
    limit: number,
    total?: number
  ): PaginatedResponse<T> => ({
    data,
    pagination: {
      page,
      limit,
      total: total ?? data.length,
      hasMore: data.length === limit,
    },
  }),
};

// Mock data utilities for development
export const MockDataUtils = {
  getMockUser: (): User => ({
    id: "dev-user-123",
    email: "developer@pianoanalyzer.dev", 
    name: "Development User",
    avatar: undefined,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }),

  getMockTokens: (): AuthTokens => ({
    accessToken: "dev-access-token-123",
    refreshToken: "dev-refresh-token-123",
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(), // 24 hours from now
  }),

  bypassAuth: () => ({
    user: MockDataUtils.getMockUser(),
    tokens: MockDataUtils.getMockTokens(),
  }),
};
