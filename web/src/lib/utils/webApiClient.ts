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
import { API_CONFIG, ApiUtils, type ApiClientInterface } from './api';

// Web-specific API client implementation
export class WebApiClient implements ApiClientInterface {
  constructor(private getAccessToken?: () => string | null) {}

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const accessToken = this.getAccessToken?.();
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...ApiUtils.getCommonHeaders(accessToken || undefined),
        ...options.headers,
      },
    };

    const response = await fetch(`${API_CONFIG.FULL_URL}${endpoint}`, config);
    return ApiUtils.handleApiResponse<T>(response);
  }

  // Auth endpoints
  async authenticateWithGoogle(token: string): Promise<ApiResponse<{ user: User; tokens: AuthTokens }>> {
    return this.makeRequest('/auth/google', {
      method: 'POST',
      body: JSON.stringify({ token }),
    });
  }

  async refreshToken(refreshToken: string): Promise<ApiResponse<{ tokens: AuthTokens }>> {
    return this.makeRequest('/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refreshToken }),
    });
  }

  async signOut(): Promise<ApiResponse<{}>> {
    return this.makeRequest('/auth/signout', {
      method: 'POST',
    });
  }

  // User endpoints
  async getUserProfile(userId: string): Promise<ApiResponse<User>> {
    return this.makeRequest(`/users/${userId}`);
  }

  async updateUserProfile(userId: string, updates: Partial<User>): Promise<ApiResponse<User>> {
    return this.makeRequest(`/users/${userId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  }

  // Recordings endpoints
  async getUserRecordings(userId: string, page = 1, limit = 20): Promise<PaginatedResponse<Recording>> {
    const queryString = ApiUtils.buildQueryString({ page, limit });
    const response = await this.makeRequest<Recording[]>(`/users/${userId}/recordings${queryString}`);
    
    return ApiUtils.toPaginatedResponse(response.data, page, limit);
  }

  async getRecording(recordingId: string): Promise<ApiResponse<Recording>> {
    return this.makeRequest(`/recordings/${recordingId}`);
  }

  async createRecording(recordingData: Omit<Recording, 'id' | 'createdAt' | 'updatedAt'>): Promise<ApiResponse<Recording>> {
    return this.makeRequest('/recordings', {
      method: 'POST',
      body: JSON.stringify(recordingData),
    });
  }

  async updateRecording(recordingId: string, updates: Partial<Recording>): Promise<ApiResponse<Recording>> {
    return this.makeRequest(`/recordings/${recordingId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  }

  async deleteRecording(recordingId: string): Promise<ApiResponse<{}>> {
    return this.makeRequest(`/recordings/${recordingId}`, {
      method: 'DELETE',
    });
  }

  async uploadRecording(recordingId: string, audioFile: File | Blob): Promise<ApiResponse<{ uploadUrl: string }>> {
    const formData = new FormData();
    formData.append('audio', audioFile);

    return this.makeRequest(`/recordings/${recordingId}/upload`, {
      method: 'POST',
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
      method: 'POST',
    });
  }

  // Progress endpoints
  async getUserProgress(userId: string): Promise<ApiResponse<Progress>> {
    return this.makeRequest(`/users/${userId}/progress`);
  }

  // Practice Sessions endpoints
  async getUserPracticeSessions(userId: string, page = 1, limit = 20): Promise<PaginatedResponse<PracticeSession>> {
    const queryString = ApiUtils.buildQueryString({ page, limit });
    const response = await this.makeRequest<PracticeSession[]>(`/users/${userId}/practice-sessions${queryString}`);
    
    return ApiUtils.toPaginatedResponse(response.data, page, limit);
  }
}

// Create singleton instance for web
export const createWebApiClient = (getAccessToken?: () => string | null) => 
  new WebApiClient(getAccessToken);
