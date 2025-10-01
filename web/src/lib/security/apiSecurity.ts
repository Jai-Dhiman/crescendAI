/**
 * API Security Layer for CrescendAI
 * 
 * Provides comprehensive security features for API interactions including
 * rate limiting, request validation, response sanitization, and attack prevention.
 */

import { browser } from '$app/environment';
import { clientConfig } from '../config/environment';
import { sanitizeObject, sanitizeInput } from './xss';
import { tokenManager } from '../services/tokenManager';

interface RateLimitConfig {
  maxRequests: number;
  timeWindow: number; // in milliseconds
  endpoint?: string;
}

interface SecurityHeaders {
  [key: string]: string;
}

interface ApiSecurityOptions {
  requireAuth?: boolean;
  rateLimit?: RateLimitConfig;
  sanitizeRequest?: boolean;
  sanitizeResponse?: boolean;
  validateResponse?: boolean;
  timeout?: number;
  retries?: number;
}

interface RequestAttempt {
  timestamp: number;
  endpoint: string;
  success: boolean;
}

interface SecurityValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  sanitizedData?: any;
}

/**
 * API Rate Limiter
 * Implements token bucket algorithm for rate limiting
 */
class ApiRateLimiter {
  private buckets = new Map<string, { tokens: number; lastRefill: number }>();
  private attempts = new Map<string, RequestAttempt[]>();

  constructor(
    private maxTokens: number = clientConfig.rateLimitMaxRequests,
    private refillRate: number = clientConfig.rateLimitMaxRequests, // tokens per window
    private timeWindow: number = clientConfig.rateLimitWindow
  ) {}

  /**
   * Check if request is allowed under rate limit
   */
  canMakeRequest(key: string, endpoint: string = 'default'): { allowed: boolean; error?: string; waitTime?: number } {
    const bucketKey = `${key}:${endpoint}`;
    const now = Date.now();
    
    // Get or create bucket
    let bucket = this.buckets.get(bucketKey);
    if (!bucket) {
      bucket = { tokens: this.maxTokens, lastRefill: now };
      this.buckets.set(bucketKey, bucket);
    }

    // Refill tokens based on time elapsed
    const timeSinceRefill = now - bucket.lastRefill;
    if (timeSinceRefill >= this.timeWindow) {
      const tokensToAdd = Math.floor(timeSinceRefill / this.timeWindow) * this.refillRate;
      bucket.tokens = Math.min(this.maxTokens, bucket.tokens + tokensToAdd);
      bucket.lastRefill = now;
    }

    // Check if we have tokens available
    if (bucket.tokens <= 0) {
      const waitTime = Math.ceil((this.timeWindow - (now - bucket.lastRefill)) / 1000);
      return {
        allowed: false,
        error: `Rate limit exceeded for ${endpoint}. Please wait ${waitTime} seconds.`,
        waitTime: waitTime * 1000,
      };
    }

    // Consume a token
    bucket.tokens -= 1;
    
    // Record this attempt
    this.recordAttempt(key, endpoint, true);

    return { allowed: true };
  }

  /**
   * Record API attempt for monitoring
   */
  private recordAttempt(key: string, endpoint: string, success: boolean): void {
    const attempts = this.attempts.get(key) || [];
    attempts.push({
      timestamp: Date.now(),
      endpoint,
      success,
    });

    // Keep only recent attempts
    const cutoff = Date.now() - this.timeWindow * 3; // Keep 3 time windows worth
    this.attempts.set(key, attempts.filter(attempt => attempt.timestamp > cutoff));
  }

  /**
   * Get rate limit status
   */
  getStatus(key: string, endpoint: string = 'default'): {
    tokensRemaining: number;
    nextRefill: number;
    recentAttempts: number;
  } {
    const bucketKey = `${key}:${endpoint}`;
    const bucket = this.buckets.get(bucketKey) || { tokens: this.maxTokens, lastRefill: Date.now() };
    const attempts = this.attempts.get(key) || [];
    
    const recentAttempts = attempts.filter(
      attempt => Date.now() - attempt.timestamp < this.timeWindow
    ).length;

    return {
      tokensRemaining: bucket.tokens,
      nextRefill: bucket.lastRefill + this.timeWindow,
      recentAttempts,
    };
  }

  /**
   * Reset rate limit for a key
   */
  reset(key: string, endpoint?: string): void {
    if (endpoint) {
      const bucketKey = `${key}:${endpoint}`;
      this.buckets.delete(bucketKey);
    } else {
      // Reset all buckets for this key
      for (const bucketKey of this.buckets.keys()) {
        if (bucketKey.startsWith(`${key}:`)) {
          this.buckets.delete(bucketKey);
        }
      }
      this.attempts.delete(key);
    }
  }
}

/**
 * Security Headers Generator
 */
function generateSecurityHeaders(options: ApiSecurityOptions = {}): SecurityHeaders {
  const headers: SecurityHeaders = {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
    'X-Content-Type-Options': 'nosniff',
    'Cache-Control': 'no-cache, no-store, must-revalidate',
  };

  // Add authentication header if required
  if (options.requireAuth) {
    const authHeader = tokenManager?.getAuthorizationHeader();
    if (authHeader) {
      headers['Authorization'] = authHeader;
    }
  }

  // Add client identification
  headers['X-Client-Version'] = clientConfig.appVersion;
  headers['X-Client-Name'] = clientConfig.appName;

  return headers;
}

/**
 * Request Validation and Sanitization
 */
function validateAndSanitizeRequest(data: any, options: ApiSecurityOptions): SecurityValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  let sanitizedData = data;

  try {
    // Sanitize request data if enabled
    if (options.sanitizeRequest && data) {
      if (typeof data === 'object') {
        sanitizedData = sanitizeObject(data, { allowHtml: false });
      } else if (typeof data === 'string') {
        const result = sanitizeInput(data, { allowHtml: false });
        sanitizedData = result.sanitized;
        warnings.push(...result.warnings);
      }
    }

    // Validate request size
    const requestSize = JSON.stringify(sanitizedData).length;
    if (requestSize > 1024 * 1024) { // 1MB limit
      errors.push('Request payload too large');
    }

    // Validate required authentication
    if (options.requireAuth && !tokenManager?.hasValidTokens()) {
      errors.push('Authentication required');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      sanitizedData,
    };
  } catch (error) {
    return {
      isValid: false,
      errors: [`Request validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`],
      warnings,
    };
  }
}

/**
 * Response Validation and Sanitization
 */
function validateAndSanitizeResponse(response: any, options: ApiSecurityOptions): SecurityValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  let sanitizedData = response;

  try {
    // Basic response validation
    if (!response) {
      return { isValid: true, errors: [], warnings: [] };
    }

    // Sanitize response data if enabled
    if (options.sanitizeResponse) {
      if (typeof response === 'object') {
        sanitizedData = sanitizeObject(response, { allowHtml: false });
      } else if (typeof response === 'string') {
        const result = sanitizeInput(response, { allowHtml: false });
        sanitizedData = result.sanitized;
        warnings.push(...result.warnings);
      }
    }

    // Validate response structure for common API patterns
    if (options.validateResponse && typeof response === 'object') {
      // Check for suspicious properties
      const suspiciousKeys = ['__proto__', 'constructor', 'prototype'];
      for (const key of suspiciousKeys) {
        if (key in response) {
          errors.push(`Response contains suspicious property: ${key}`);
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      sanitizedData,
    };
  } catch (error) {
    return {
      isValid: false,
      errors: [`Response validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`],
      warnings,
    };
  }
}

/**
 * Secure Fetch Wrapper
 */
class SecureApiClient {
  private rateLimiter: ApiRateLimiter;
  private sessionKey: string;

  constructor() {
    this.rateLimiter = new ApiRateLimiter();
    this.sessionKey = browser ? this.generateSessionKey() : 'server';
  }

  /**
   * Generate a unique session key for rate limiting
   */
  private generateSessionKey(): string {
    let key = sessionStorage.getItem('_crescend_session_key');
    if (!key) {
      key = Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
      sessionStorage.setItem('_crescend_session_key', key);
    }
    return key;
  }

  /**
   * Extract endpoint name from URL for rate limiting
   */
  private getEndpointKey(url: string): string {
    try {
      const urlObj = new URL(url);
      const pathParts = urlObj.pathname.split('/').filter(Boolean);
      return pathParts.slice(-2).join('/'); // Use last two path segments
    } catch {
      return 'unknown';
    }
  }

  /**
   * Secure API request with comprehensive security checks
   */
  async secureRequest<T = any>(
    url: string,
    options: RequestInit & { security?: ApiSecurityOptions } = {}
  ): Promise<T> {
    const securityOptions: ApiSecurityOptions = {
      requireAuth: false,
      sanitizeRequest: true,
      sanitizeResponse: true,
      validateResponse: true,
      timeout: 30000,
      retries: 2,
      ...options.security,
    };

    const endpoint = this.getEndpointKey(url);

    // Rate limiting check
    const rateLimitCheck = this.rateLimiter.canMakeRequest(this.sessionKey, endpoint);
    if (!rateLimitCheck.allowed) {
      throw new Error(rateLimitCheck.error);
    }

    // Validate and sanitize request
    const requestValidation = validateAndSanitizeRequest(
      options.body ? JSON.parse(options.body as string) : null,
      securityOptions
    );

    if (!requestValidation.isValid) {
      throw new Error(`Request validation failed: ${requestValidation.errors.join(', ')}`);
    }

    // Generate security headers
    const securityHeaders = generateSecurityHeaders(securityOptions);
    
    // Prepare request options
    const requestOptions: RequestInit = {
      ...options,
      headers: {
        ...securityHeaders,
        ...options.headers,
      },
      body: requestValidation.sanitizedData 
        ? JSON.stringify(requestValidation.sanitizedData)
        : options.body,
    };

    // Add request timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), securityOptions.timeout);
    requestOptions.signal = controller.signal;

    try {
      let lastError: Error | null = null;
      let attempt = 0;
      const maxAttempts = securityOptions.retries! + 1;

      while (attempt < maxAttempts) {
        try {
          const response = await fetch(url, requestOptions);
          clearTimeout(timeoutId);

          // Check for HTTP errors
          if (!response.ok) {
            const errorText = await response.text().catch(() => 'Unknown error');
            throw new Error(`HTTP ${response.status}: ${errorText}`);
          }

          // Parse response
          const responseData = await response.json().catch(() => null);

          // Validate and sanitize response
          const responseValidation = validateAndSanitizeResponse(responseData, securityOptions);
          
          if (!responseValidation.isValid) {
            throw new Error(`Response validation failed: ${responseValidation.errors.join(', ')}`);
          }

          // Log warnings if any
          if (responseValidation.warnings.length > 0) {
            console.warn('API Response Warnings:', responseValidation.warnings);
          }

          return responseValidation.sanitizedData;

        } catch (error) {
          lastError = error instanceof Error ? error : new Error('Unknown error');
          attempt++;

          // Don't retry on authentication errors or validation errors
          if (lastError.message.includes('401') || 
              lastError.message.includes('403') ||
              lastError.message.includes('validation failed')) {
            break;
          }

          // Wait before retry (exponential backoff)
          if (attempt < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
          }
        }
      }

      throw lastError || new Error('Request failed after all retries');

    } catch (error) {
      clearTimeout(timeoutId);
      
      // Record failed attempt
      this.rateLimiter['recordAttempt'](this.sessionKey, endpoint, false);
      
      if (error instanceof Error) {
        throw new Error(`Secure API request failed: ${error.message}`);
      }
      throw error;
    }
  }

  /**
   * GET request with security
   */
  async get<T = any>(url: string, security?: ApiSecurityOptions): Promise<T> {
    return this.secureRequest<T>(url, {
      method: 'GET',
      security,
    });
  }

  /**
   * POST request with security
   */
  async post<T = any>(url: string, data?: any, security?: ApiSecurityOptions): Promise<T> {
    return this.secureRequest<T>(url, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
      security,
    });
  }

  /**
   * PUT request with security
   */
  async put<T = any>(url: string, data?: any, security?: ApiSecurityOptions): Promise<T> {
    return this.secureRequest<T>(url, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
      security,
    });
  }

  /**
   * DELETE request with security
   */
  async delete<T = any>(url: string, security?: ApiSecurityOptions): Promise<T> {
    return this.secureRequest<T>(url, {
      method: 'DELETE',
      security,
    });
  }

  /**
   * Get rate limit status
   */
  getRateLimitStatus(endpoint?: string): ReturnType<ApiRateLimiter['getStatus']> {
    return this.rateLimiter.getStatus(this.sessionKey, endpoint);
  }

  /**
   * Reset rate limits
   */
  resetRateLimit(endpoint?: string): void {
    this.rateLimiter.reset(this.sessionKey, endpoint);
  }
}

// Export singleton instance
export const secureApiClient = new SecureApiClient();

/**
 * CSRF Protection utilities
 */
export const CSRFProtection = {
  /**
   * Generate CSRF token
   */
  generateToken(): string {
    if (browser && crypto && crypto.getRandomValues) {
      const array = new Uint8Array(32);
      crypto.getRandomValues(array);
      return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }
    return Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
  },

  /**
   * Store CSRF token
   */
  setToken(token: string): void {
    if (browser) {
      sessionStorage.setItem('_csrf_token', token);
      // Also set as meta tag for forms
      let metaTag = document.querySelector('meta[name="csrf-token"]') as HTMLMetaElement;
      if (!metaTag) {
        metaTag = document.createElement('meta');
        metaTag.name = 'csrf-token';
        document.head.appendChild(metaTag);
      }
      metaTag.content = token;
    }
  },

  /**
   * Get CSRF token
   */
  getToken(): string | null {
    if (browser) {
      return sessionStorage.getItem('_csrf_token');
    }
    return null;
  },

  /**
   * Validate CSRF token
   */
  validateToken(token: string): boolean {
    const storedToken = this.getToken();
    return storedToken !== null && storedToken === token;
  },
} as const;

// Initialize CSRF token on load
if (browser) {
  const existingToken = CSRFProtection.getToken();
  if (!existingToken) {
    const newToken = CSRFProtection.generateToken();
    CSRFProtection.setToken(newToken);
  }
}

/**
 * Export types and interfaces
 */
export type { ApiSecurityOptions, RateLimitConfig, SecurityValidationResult };