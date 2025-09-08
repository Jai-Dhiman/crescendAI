import { useAuthStore } from "../stores";
import type {
  Analysis,
  ApiResponse,
  PaginatedResponse,
  PracticeSession,
  Progress,
  Recording,
  User,
} from "../types";

// API Configuration
const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_URL || "https://api.pianoanalyzer.com";
const API_VERSION = "v1";

class ApiClient {
  private baseURL: string;

  constructor() {
    this.baseURL = `${API_BASE_URL}/${API_VERSION}`;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const { tokens } = useAuthStore.getState();

    const defaultHeaders: HeadersInit = {
      "Content-Type": "application/json",
    };

    if (tokens?.accessToken) {
      defaultHeaders.Authorization = `Bearer ${tokens.accessToken}`;
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, config);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.message || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      const data = await response.json();
      return data;
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : "Network error");
    }
  }

  // Auth endpoints
  async authenticateWithGoogle(
    token: string
  ): Promise<ApiResponse<{ user: User; tokens: any }>> {
    return this.makeRequest("/auth/google", {
      method: "POST",
      body: JSON.stringify({ token }),
    });
  }

  async refreshToken(
    refreshToken: string
  ): Promise<ApiResponse<{ tokens: any }>> {
    return this.makeRequest("/auth/refresh", {
      method: "POST",
      body: JSON.stringify({ refreshToken }),
    });
  }

  async signOut(): Promise<ApiResponse<{}>> {
    return this.makeRequest("/auth/signout", {
      method: "POST",
    });
  }

  // User endpoints
  async getUserProfile(userId: string): Promise<ApiResponse<User>> {
    return this.makeRequest(`/users/${userId}`);
  }

  async updateUserProfile(
    userId: string,
    updates: Partial<User>
  ): Promise<ApiResponse<User>> {
    return this.makeRequest(`/users/${userId}`, {
      method: "PATCH",
      body: JSON.stringify(updates),
    });
  }

  // Recordings endpoints
  async getUserRecordings(
    userId: string,
    page = 1,
    limit = 20
  ): Promise<PaginatedResponse<Recording>> {
    const response = await this.makeRequest<Recording[]>(
      `/users/${userId}/recordings?page=${page}&limit=${limit}`
    );
    // Transform the response to match PaginatedResponse structure
    return {
      data: response.data,
      pagination: {
        page,
        limit,
        total: response.data.length,
        hasMore: response.data.length === limit,
      },
    };
  }

  async getRecording(recordingId: string): Promise<ApiResponse<Recording>> {
    return this.makeRequest(`/recordings/${recordingId}`);
  }

  async createRecording(
    recordingData: Omit<Recording, "id" | "createdAt" | "updatedAt">
  ): Promise<ApiResponse<Recording>> {
    return this.makeRequest("/recordings", {
      method: "POST",
      body: JSON.stringify(recordingData),
    });
  }

  async updateRecording(
    recordingId: string,
    updates: Partial<Recording>
  ): Promise<ApiResponse<Recording>> {
    return this.makeRequest(`/recordings/${recordingId}`, {
      method: "PATCH",
      body: JSON.stringify(updates),
    });
  }

  async deleteRecording(recordingId: string): Promise<ApiResponse<{}>> {
    return this.makeRequest(`/recordings/${recordingId}`, {
      method: "DELETE",
    });
  }

  async uploadRecording(
    recordingId: string,
    audioFile: File | Blob
  ): Promise<ApiResponse<{ uploadUrl: string }>> {
    const formData = new FormData();
    formData.append("audio", audioFile);

    return this.makeRequest(`/recordings/${recordingId}/upload`, {
      method: "POST",
      body: formData,
      headers: {}, // Let the browser set Content-Type for FormData
    });
  }

  // Analysis endpoints
  async getAnalysis(recordingId: string): Promise<ApiResponse<Analysis>> {
    return this.makeRequest(`/recordings/${recordingId}/analysis`);
  }

  async requestAnalysis(recordingId: string): Promise<ApiResponse<Analysis>> {
    return this.makeRequest(`/recordings/${recordingId}/analyze`, {
      method: "POST",
    });
  }

  // Progress endpoints
  async getUserProgress(userId: string): Promise<ApiResponse<Progress>> {
    return this.makeRequest(`/users/${userId}/progress`);
  }

  // Practice Sessions endpoints
  async getUserPracticeSessions(
    userId: string,
    page = 1,
    limit = 20
  ): Promise<PaginatedResponse<PracticeSession>> {
    const response = await this.makeRequest<PracticeSession[]>(
      `/users/${userId}/practice-sessions?page=${page}&limit=${limit}`
    );
    // Transform the response to match PaginatedResponse structure
    return {
      data: response.data,
      pagination: {
        page,
        limit,
        total: response.data.length,
        hasMore: response.data.length === limit,
      },
    };
  }

  async createPracticeSession(
    sessionData: Omit<PracticeSession, "id">
  ): Promise<ApiResponse<PracticeSession>> {
    return this.makeRequest("/practice-sessions", {
      method: "POST",
      body: JSON.stringify(sessionData),
    });
  }

  async endPracticeSession(
    sessionId: string,
    endTime: string
  ): Promise<ApiResponse<PracticeSession>> {
    return this.makeRequest(`/practice-sessions/${sessionId}/end`, {
      method: "POST",
      body: JSON.stringify({ endTime }),
    });
  }
}

export const apiClient = new ApiClient();
