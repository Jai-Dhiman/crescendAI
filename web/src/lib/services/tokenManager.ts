/**
 * JWT Token Management for CrescendAI
 * 
 * Handles secure token storage, validation, and automatic refresh
 * with comprehensive security features.
 */

import { browser } from '$app/environment';
import { storage } from '../stores/storage';
import { clientConfig } from '../config/environment';
import type { AuthTokens } from '../types';

interface TokenPayload {
  sub: string; // Subject (user ID)
  iat: number; // Issued at
  exp: number; // Expires at
  aud: string; // Audience
  iss: string; // Issuer
}

interface TokenValidation {
  isValid: boolean;
  isExpired: boolean;
  expiresIn: number; // milliseconds until expiration
  payload?: TokenPayload;
  error?: string;
}

/**
 * JWT Token Manager
 * Provides secure token handling with automatic refresh
 */
export class TokenManager {
  private static instance: TokenManager | null = null;
  private refreshPromise: Promise<AuthTokens | null> | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;

  // Storage keys (will be encrypted automatically)
  private static readonly ACCESS_TOKEN_KEY = 'access-token';
  private static readonly REFRESH_TOKEN_KEY = 'refresh-token';
  private static readonly TOKEN_METADATA_KEY = 'token-metadata';

  private constructor() {
    if (!browser) {
      throw new Error('TokenManager can only be used in browser context');
    }
    
    // Start automatic refresh monitoring
    this.startRefreshMonitoring();
  }

  /**
   * Get singleton instance
   */
  static getInstance(): TokenManager {
    if (!TokenManager.instance && browser) {
      TokenManager.instance = new TokenManager();
    }
    return TokenManager.instance!;
  }

  /**
   * Parse JWT token payload without verification
   * (Client-side parsing for expiration checking only)
   */
  private parseTokenPayload(token: string): TokenPayload | null {
    try {
      const parts = token.split('.');
      if (parts.length !== 3) {
        throw new Error('Invalid JWT format');
      }

      // Decode the payload (second part)
      const payload = parts[1];
      // Add padding if needed for base64 decoding
      const paddedPayload = payload + '='.repeat((4 - payload.length % 4) % 4);
      const decoded = atob(paddedPayload.replace(/-/g, '+').replace(/_/g, '/'));
      
      return JSON.parse(decoded);
    } catch (error) {
      console.warn('Failed to parse token payload:', error);
      return null;
    }
  }

  /**
   * Validate JWT token
   */
  private validateToken(token: string): TokenValidation {
    if (!token) {
      return {
        isValid: false,
        isExpired: true,
        expiresIn: 0,
        error: 'Token is empty',
      };
    }

    const payload = this.parseTokenPayload(token);
    if (!payload) {
      return {
        isValid: false,
        isExpired: true,
        expiresIn: 0,
        error: 'Invalid token format',
      };
    }

    const now = Date.now() / 1000; // Current time in seconds
    const isExpired = payload.exp <= now;
    const expiresIn = Math.max(0, (payload.exp - now) * 1000); // Convert to milliseconds

    // Additional validation
    const hasRequiredClaims = payload.sub && payload.iat && payload.exp;
    
    return {
      isValid: !isExpired && hasRequiredClaims,
      isExpired,
      expiresIn,
      payload,
      error: isExpired ? 'Token expired' : undefined,
    };
  }

  /**
   * Store tokens securely
   */
  async storeTokens(tokens: AuthTokens): Promise<void> {
    try {
      const metadata = {
        storedAt: Date.now(),
        expiresAt: tokens.expiresAt,
        tokenType: 'Bearer',
      };

      // Store tokens using secure storage
      storage.setSecure(TokenManager.ACCESS_TOKEN_KEY, tokens.accessToken);
      storage.setSecure(TokenManager.REFRESH_TOKEN_KEY, tokens.refreshToken);
      storage.setSecure(TokenManager.TOKEN_METADATA_KEY, metadata);

      // Schedule automatic refresh
      this.scheduleTokenRefresh(tokens.accessToken);

      console.info('✅ Tokens stored securely');
    } catch (error) {
      console.error('Failed to store tokens:', error);
      throw new Error('Token storage failed');
    }
  }

  /**
   * Get stored access token
   */
  getAccessToken(): string | null {
    try {
      const token = storage.getSecure<string>(TokenManager.ACCESS_TOKEN_KEY);
      if (!token) return null;

      const validation = this.validateToken(token);
      if (!validation.isValid) {
        console.warn('Stored access token is invalid:', validation.error);
        this.clearTokens(); // Clear invalid tokens
        return null;
      }

      return token;
    } catch (error) {
      console.error('Failed to retrieve access token:', error);
      return null;
    }
  }

  /**
   * Get stored refresh token
   */
  getRefreshToken(): string | null {
    try {
      return storage.getSecure<string>(TokenManager.REFRESH_TOKEN_KEY);
    } catch (error) {
      console.error('Failed to retrieve refresh token:', error);
      return null;
    }
  }

  /**
   * Check if tokens are available and valid
   */
  hasValidTokens(): boolean {
    const accessToken = this.getAccessToken();
    return accessToken !== null;
  }

  /**
   * Get token expiration info
   */
  getTokenExpiration(): { expiresIn: number; isExpired: boolean } | null {
    const token = storage.getSecure<string>(TokenManager.ACCESS_TOKEN_KEY);
    if (!token) return null;

    const validation = this.validateToken(token);
    return {
      expiresIn: validation.expiresIn,
      isExpired: validation.isExpired,
    };
  }

  /**
   * Clear all stored tokens
   */
  clearTokens(): void {
    try {
      storage.deleteSecure(TokenManager.ACCESS_TOKEN_KEY);
      storage.deleteSecure(TokenManager.REFRESH_TOKEN_KEY);
      storage.deleteSecure(TokenManager.TOKEN_METADATA_KEY);

      // Clear refresh timer
      if (this.refreshTimer) {
        clearTimeout(this.refreshTimer);
        this.refreshTimer = null;
      }

      console.info('✅ Tokens cleared');
    } catch (error) {
      console.error('Failed to clear tokens:', error);
    }
  }

  /**
   * Refresh tokens using refresh token
   */
  async refreshTokens(): Promise<AuthTokens | null> {
    // Prevent concurrent refresh requests
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = this.performTokenRefresh();
    const result = await this.refreshPromise;
    this.refreshPromise = null;

    return result;
  }

  /**
   * Perform the actual token refresh
   */
  private async performTokenRefresh(): Promise<AuthTokens | null> {
    try {
      const refreshToken = this.getRefreshToken();
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      // Make API call to refresh tokens
      const response = await fetch(`${clientConfig.apiUrl}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refreshToken }),
      });

      if (!response.ok) {
        throw new Error(`Token refresh failed: ${response.status}`);
      }

      const data = await response.json();
      const newTokens: AuthTokens = data.tokens;

      // Store new tokens
      await this.storeTokens(newTokens);

      console.info('✅ Tokens refreshed successfully');
      return newTokens;
    } catch (error) {
      console.error('Token refresh failed:', error);
      
      // Clear invalid tokens
      this.clearTokens();
      
      // Dispatch custom event for auth state change
      if (browser) {
        window.dispatchEvent(new CustomEvent('auth:token-refresh-failed', {
          detail: { error },
        }));
      }
      
      return null;
    }
  }

  /**
   * Schedule automatic token refresh
   */
  private scheduleTokenRefresh(accessToken: string): void {
    const validation = this.validateToken(accessToken);
    if (!validation.isValid) return;

    // Schedule refresh 5 minutes before expiration
    const refreshIn = Math.max(0, validation.expiresIn - 5 * 60 * 1000);

    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    this.refreshTimer = setTimeout(async () => {
      console.info('🔄 Starting automatic token refresh...');
      await this.refreshTokens();
    }, refreshIn);

    console.info(`⏰ Token refresh scheduled in ${Math.round(refreshIn / 1000)}s`);
  }

  /**
   * Start monitoring for token refresh needs
   */
  private startRefreshMonitoring(): void {
    // Check on page focus
    if (browser) {
      window.addEventListener('focus', () => {
        const expiration = this.getTokenExpiration();
        if (expiration && expiration.expiresIn < 10 * 60 * 1000) { // Less than 10 minutes
          this.refreshTokens();
        }
      });

      // Periodic check every 5 minutes
      setInterval(() => {
        const expiration = this.getTokenExpiration();
        if (expiration && expiration.expiresIn < 5 * 60 * 1000) { // Less than 5 minutes
          this.refreshTokens();
        }
      }, 5 * 60 * 1000);
    }
  }

  /**
   * Get Authorization header value
   */
  getAuthorizationHeader(): string | null {
    const token = this.getAccessToken();
    return token ? `Bearer ${token}` : null;
  }

  /**
   * Check if token refresh is needed
   */
  needsRefresh(): boolean {
    const expiration = this.getTokenExpiration();
    if (!expiration) return true;

    // Refresh if expires in less than 10 minutes
    return expiration.expiresIn < 10 * 60 * 1000;
  }
}

/**
 * Export singleton instance
 */
export const tokenManager = browser ? TokenManager.getInstance() : null;

/**
 * Hook for automatic token refresh in API requests
 */
export async function withTokenRefresh<T>(
  apiCall: (token: string) => Promise<T>
): Promise<T> {
  if (!tokenManager) {
    throw new Error('Token manager not available');
  }

  // Check if refresh is needed
  if (tokenManager.needsRefresh()) {
    await tokenManager.refreshTokens();
  }

  const token = tokenManager.getAccessToken();
  if (!token) {
    throw new Error('No valid token available');
  }

  try {
    return await apiCall(token);
  } catch (error) {
    // If API call fails with 401, try refreshing token once
    if (error instanceof Error && error.message.includes('401')) {
      console.info('🔄 API returned 401, attempting token refresh...');
      await tokenManager.refreshTokens();
      
      const newToken = tokenManager.getAccessToken();
      if (!newToken) {
        throw new Error('Token refresh failed');
      }

      return await apiCall(newToken);
    }

    throw error;
  }
}