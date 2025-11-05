import {
  validateUploadResponse,
  validateAnalysisResponse,
  validateJobStatus,
  validateAnalysisResult,
  validateHealthResponse,
  type UploadResponse,
  type AnalysisRequest,
  type AnalysisResponse,
  type JobStatus,
  type AnalysisResult,
  type HealthResponse,
  ApiErrorSchema
} from '../schemas/api.js';

import type {
  ChatSession,
  ChatSessionWithMessages,
  Message,
  ListChatSessionsResponse,
  CreateChatSessionRequest,
  SendMessageRequest
} from '../types/chat.js';

import { clientConfig, apiEndpoints } from '../config/environment';
import { secureApiClient } from '../security/apiSecurity';
import { prepareSecureUpload } from '../security/fileValidation';

// API Configuration using secure environment management
const API_CONFIG = {
  BASE_URL: clientConfig.apiUrl,
  VERSION: "v1",
  get FULL_URL() {
    return apiEndpoints.v1;
  }
} as const;

// Custom API Error class
export class CrescendApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string
  ) {
    super(message);
    this.name = 'CrescendApiError';
  }
}

// CrescendAI API Client
export class CrescendApiClient {
  
  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {},
    validator?: (data: unknown) => T
  ): Promise<T> {
    try {
      // TODO: Add authentication headers when proper auth is implemented
      const response = await fetch(`${API_CONFIG.FULL_URL}${endpoint}`, {
        ...options,
        headers: {
          ...options.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        
        // Try to parse with our error schema
        const parsedError = ApiErrorSchema.safeParse(errorData);
        if (parsedError.success) {
          throw new CrescendApiError(
            parsedError.data.message,
            response.status,
            parsedError.data.code?.toString()
          );
        }
        
        throw new CrescendApiError(
          `HTTP ${response.status}: ${response.statusText}`,
          response.status
        );
      }

      const data = await response.json();
      
      // Validate response if validator provided
      if (validator) {
        return validator(data);
      }
      
      return data;
    } catch (error) {
      if (error instanceof CrescendApiError) {
        throw error;
      }
      
      // Handle network errors and other issues
      throw new CrescendApiError(
        `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Upload audio file to the server with security validation
   */
  async uploadAudio(file: File): Promise<UploadResponse> {
    // Validate and secure the file upload
    const secureUpload = await prepareSecureUpload(file);
    
    if (!secureUpload.isValid) {
      throw new CrescendApiError(
        `File validation failed: ${secureUpload.errors.join(', ')}`,
        400,
        'INVALID_FILE'
      );
    }

    // Log any warnings
    if (secureUpload.warnings.length > 0) {
      console.warn('File upload warnings:', secureUpload.warnings);
    }

    const formData = new FormData();
    formData.append('audio', secureUpload.file!, secureUpload.metadata?.sanitizedName);
    
    // Add metadata for server processing
    if (secureUpload.metadata) {
      formData.append('metadata', JSON.stringify({
        originalName: secureUpload.metadata.originalName,
        size: secureUpload.metadata.size,
        type: secureUpload.metadata.type,
        hash: secureUpload.metadata.hash,
      }));
    }

    return this.makeRequest(
      '/upload',
      {
        method: 'POST',
        body: formData,
        headers: {
          // Don't set Content-Type for FormData - let browser set it with boundary
        },
      },
      validateUploadResponse
    );
  }

  /**
   * Start analysis for an uploaded file
   */
  async startAnalysis(fileId: string): Promise<AnalysisResponse> {
    const request: AnalysisRequest = { id: fileId };
    
    return this.makeRequest(
      '/analyze',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      },
      validateAnalysisResponse
    );
  }

  /**
   * Get job status by job ID
   */
  async getJobStatus(jobId: string): Promise<JobStatus> {
    return this.makeRequest(
      `/job/${jobId}`,
      {
        method: 'GET',
      },
      validateJobStatus
    );
  }

  /**
   * Get analysis result by result ID
   */
  async getAnalysisResult(resultId: string): Promise<AnalysisResult> {
    return this.makeRequest(
      `/result/${resultId}`,
      {
        method: 'GET',
      },
      validateAnalysisResult
    );
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.makeRequest(
      '/health',
      {
        method: 'GET',
      },
      validateHealthResponse
    );
  }

  /**
   * Start model comparison for an uploaded file
   */
  async startComparison(fileId: string, modelA?: string, modelB?: string): Promise<any> {
    const request = { 
      id: fileId,
      model_a: modelA || 'hybrid_ast',
      model_b: modelB || 'ultra_small_ast'
    };
    
    return this.makeRequest(
      '/compare',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      }
      // TODO: Add comparison response validator
    );
  }

  /**
   * Get comparison result by comparison ID
   */
  async getComparisonResult(comparisonId: string): Promise<any> {
    return this.makeRequest(
      `/comparison/${comparisonId}`,
      {
        method: 'GET',
      }
      // TODO: Add comparison result validator
    );
  }

  /**
   * Save user preference for A/B test
   */
  async saveUserPreference(comparisonId: string, preferredModel: 'model_a' | 'model_b', feedback?: string): Promise<any> {
    const request = {
      comparison_id: comparisonId,
      preferred_model: preferredModel,
      feedback
    };
    
    return this.makeRequest(
      '/preference',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      }
    );
  }

  /**
   * Complete upload and analysis workflow
   * Returns the final analysis result
   */
  async uploadAndAnalyze(
    file: File,
    onProgress?: (stage: 'uploading' | 'analyzing' | 'processing', progress?: number) => void
  ): Promise<AnalysisResult> {
    
    // Step 1: Upload file
    onProgress?.('uploading');
    const uploadResponse = await this.uploadAudio(file);
    
    // Step 2: Start analysis
    onProgress?.('analyzing');
    const analysisResponse = await this.startAnalysis(uploadResponse.id);
    
    // Step 3: Poll for completion
    onProgress?.('processing');
    const result = await this.pollForResult(analysisResponse.job_id, (progress) => {
      onProgress?.('processing', progress);
    });
    
    return result;
  }

  /**
   * Complete upload and comparison workflow
   * Returns the final comparison result
   */
  async uploadAndCompare(
    file: File,
    modelA?: string,
    modelB?: string,
    onProgress?: (stage: 'uploading' | 'analyzing' | 'processing', progress?: number) => void
  ): Promise<any> {
    
    // Step 1: Upload file
    onProgress?.('uploading');
    const uploadResponse = await this.uploadAudio(file);
    
    // Step 2: Start comparison
    onProgress?.('analyzing');
    const comparisonResponse = await this.startComparison(uploadResponse.id, modelA, modelB);
    
    // Step 3: Poll for completion
    onProgress?.('processing');
    const result = await this.pollForComparisonResult(comparisonResponse.comparison_id, (progress) => {
      onProgress?.('processing', progress);
    });
    
    return result;
  }

  /**
   * Poll job status until completion
   */
  private async pollForResult(
    jobId: string,
    onProgress?: (progress: number) => void,
    maxAttempts: number = 60, // 5 minutes with 5-second intervals
    interval: number = 5000
  ): Promise<AnalysisResult> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const status = await this.getJobStatus(jobId);
      
      if (status.progress !== undefined) {
        onProgress?.(status.progress);
      }
      
      if (status.status === 'completed') {
        // Job completed, get the result
        return await this.getAnalysisResult(jobId);
      }
      
      if (status.status === 'failed') {
        throw new CrescendApiError(
          status.error || 'Analysis failed',
          500,
          'ANALYSIS_FAILED'
        );
      }
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    throw new CrescendApiError(
      'Analysis timed out',
      408,
      'ANALYSIS_TIMEOUT'
    );
  }

  /**
   * Poll comparison status until completion
   */
  private async pollForComparisonResult(
    comparisonId: string,
    onProgress?: (progress: number) => void,
    maxAttempts: number = 60, // 5 minutes with 5-second intervals
    interval: number = 5000
  ): Promise<any> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const status = await this.getJobStatus(comparisonId);
      
      if (status.progress !== undefined) {
        onProgress?.(status.progress);
      }
      
      if (status.status === 'completed') {
        // Comparison completed, get the result
        return await this.getComparisonResult(comparisonId);
      }
      
      if (status.status === 'failed') {
        throw new CrescendApiError(
          status.error || 'Model comparison failed',
          500,
          'COMPARISON_FAILED'
        );
      }
      
      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    throw new CrescendApiError(
      'Model comparison timed out',
      408,
      'COMPARISON_TIMEOUT'
    );
  }

  /**
   * Create a new chat session
   */
  async createChatSession(title: string = 'New Chat'): Promise<ChatSession> {
    const request: CreateChatSessionRequest = { title };

    return this.makeRequest(
      '/chat/sessions',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      }
    );
  }

  /**
   * Get a chat session with its message history
   */
  async getChatSession(sessionId: string): Promise<ChatSessionWithMessages> {
    return this.makeRequest(
      `/chat/sessions/${sessionId}`,
      {
        method: 'GET',
      }
    );
  }

  /**
   * List all chat sessions with pagination
   */
  async listChatSessions(page: number = 1, limit: number = 20): Promise<ListChatSessionsResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });

    return this.makeRequest(
      `/chat/sessions?${params.toString()}`,
      {
        method: 'GET',
      }
    );
  }

  /**
   * Delete a chat session
   */
  async deleteChatSession(sessionId: string): Promise<void> {
    return this.makeRequest(
      `/chat/sessions/${sessionId}`,
      {
        method: 'DELETE',
      }
    );
  }

  /**
   * Send a message and receive streaming response via SSE
   */
  async sendChatMessage(
    sessionId: string,
    content: string,
    onToken?: (token: string) => void,
    onToolCall?: (tool: string, args: any) => void
  ): Promise<Message> {
    const request: SendMessageRequest = {
      session_id: sessionId,
      content,
    };

    return new Promise(async (resolve, reject) => {
      try {
        const response = await fetch(`${API_CONFIG.FULL_URL}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const parsedError = ApiErrorSchema.safeParse(errorData);
          if (parsedError.success) {
            reject(new CrescendApiError(
              parsedError.data.message,
              response.status,
              parsedError.data.code?.toString()
            ));
          } else {
            reject(new CrescendApiError(
              `HTTP ${response.status}: ${response.statusText}`,
              response.status
            ));
          }
          return;
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          reject(new CrescendApiError('No response body'));
          return;
        }

        let fullContent = '';
        let messageId = '';
        let createdAt = '';
        const toolCalls: any[] = [];

        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);

              if (data === '[DONE]') {
                // Stream complete
                resolve({
                  id: messageId,
                  role: 'assistant',
                  content: fullContent,
                  created_at: createdAt,
                  tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
                });
                return;
              }

              try {
                const parsed = JSON.parse(data);

                if (parsed.type === 'token') {
                  fullContent += parsed.content;
                  onToken?.(parsed.content);
                } else if (parsed.type === 'tool_call') {
                  toolCalls.push({
                    tool: parsed.tool,
                    arguments: parsed.arguments,
                  });
                  onToolCall?.(parsed.tool, parsed.arguments);
                } else if (parsed.type === 'message') {
                  messageId = parsed.id;
                  createdAt = parsed.created_at;
                }
              } catch (e) {
                console.warn('Failed to parse SSE data:', data);
              }
            }
          }
        }

        // If we get here without a [DONE] message, return what we have
        if (messageId) {
          resolve({
            id: messageId,
            role: 'assistant',
            content: fullContent,
            created_at: createdAt,
            tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          });
        } else {
          reject(new CrescendApiError('Incomplete message received'));
        }
      } catch (error) {
        if (error instanceof CrescendApiError) {
          reject(error);
        } else {
          reject(new CrescendApiError(
            `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`
          ));
        }
      }
    });
  }
}

// Create singleton instance
export const crescendApi = new CrescendApiClient();

// Export for direct use
export default crescendApi;