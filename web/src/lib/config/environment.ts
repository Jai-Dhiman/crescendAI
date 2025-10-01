import { dev } from '$app/environment';
import { browser } from '$app/environment';

/**
 * Environment Configuration for CrescendAI Web Application
 * 
 * This module provides a secure way to handle environment variables,
 * separating client-safe from server-only configurations.
 * 
 * SECURITY NOTES:
 * - Only PUBLIC_ prefixed variables are available in the browser
 * - Server-only variables must never be exposed to the client
 * - All API URLs and sensitive configs are validated and sanitized
 */

// Type definitions for environment configuration
interface ClientConfig {
  readonly apiUrl: string;
  readonly appName: string;
  readonly appVersion: string;
  readonly enableAnalytics: boolean;
  readonly enableDebugLogging: boolean;
  readonly supportEmail: string;
  readonly maxFileSize: number; // in bytes
  readonly allowedFileTypes: readonly string[];
  readonly rateLimitWindow: number; // in milliseconds
  readonly rateLimitMaxRequests: number;
}

interface ServerConfig extends ClientConfig {
  readonly jwtSecret: string;
  readonly dbConnectionString: string;
  readonly redisUrl: string;
  readonly encryptionKey: string;
  readonly adminEmail: string;
  readonly smtpSettings: {
    readonly host: string;
    readonly port: number;
    readonly secure: boolean;
    readonly user: string;
    readonly password: string;
  };
}

/**
 * Validate and sanitize URL to prevent injection attacks
 */
function validateUrl(url: string | undefined, fallback: string): string {
  if (!url) return fallback;
  
  try {
    const parsed = new URL(url);
    // Only allow HTTP/HTTPS protocols
    if (!['http:', 'https:'].includes(parsed.protocol)) {
      console.warn(`Invalid protocol in URL: ${url}, using fallback`);
      return fallback;
    }
    return url;
  } catch {
    console.warn(`Invalid URL format: ${url}, using fallback`);
    return fallback;
  }
}

/**
 * Validate boolean environment variable
 */
function validateBoolean(value: string | undefined, fallback: boolean): boolean {
  if (!value) return fallback;
  return value.toLowerCase() === 'true';
}

/**
 * Validate number environment variable
 */
function validateNumber(value: string | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Number(value);
  return isNaN(parsed) ? fallback : parsed;
}

/**
 * Client-safe configuration (available in browser)
 * Only includes PUBLIC_ prefixed environment variables
 */
export const clientConfig: ClientConfig = {
  apiUrl: validateUrl(
    import.meta.env.PUBLIC_API_URL,
    dev ? 'http://localhost:8787' : 'https://crescendai-backend.jai-d.workers.dev'
  ),
  appName: import.meta.env.PUBLIC_APP_NAME || 'CrescendAI',
  appVersion: import.meta.env.PUBLIC_APP_VERSION || '1.0.0',
  enableAnalytics: validateBoolean(import.meta.env.PUBLIC_ENABLE_ANALYTICS, !dev),
  enableDebugLogging: validateBoolean(import.meta.env.PUBLIC_DEBUG_LOGGING, dev),
  supportEmail: import.meta.env.PUBLIC_SUPPORT_EMAIL || 'support@crescendai.app',
  
  // File upload limits
  maxFileSize: validateNumber(import.meta.env.PUBLIC_MAX_FILE_SIZE, 50 * 1024 * 1024), // 50MB default
  allowedFileTypes: Object.freeze([
    'audio/mpeg',
    'audio/wav',
    'audio/x-wav', // Support for .wav files that show as audio/x-wav
    'audio/ogg',
    'audio/mp4',
    'audio/aac',
    'audio/flac',
  ]),
  
  // Rate limiting (client-side enforcement)
  rateLimitWindow: validateNumber(import.meta.env.PUBLIC_RATE_LIMIT_WINDOW, 60000), // 1 minute
  rateLimitMaxRequests: validateNumber(import.meta.env.PUBLIC_RATE_LIMIT_MAX, 100),
} as const;

/**
 * Server-only configuration (not available in browser)
 * Includes sensitive configuration that must never be exposed to clients
 */
function createServerConfig(): ServerConfig {
  if (browser) {
    throw new Error('Server configuration accessed in browser context');
  }

  return {
    ...clientConfig,
    
    // Authentication & Security
    jwtSecret: import.meta.env.JWT_SECRET || (dev ? 'dev-jwt-secret-change-in-production' : ''),
    encryptionKey: import.meta.env.ENCRYPTION_KEY || (dev ? 'dev-encryption-key-32-chars-long!' : ''),
    
    // Database
    dbConnectionString: import.meta.env.DATABASE_URL || '',
    redisUrl: import.meta.env.REDIS_URL || '',
    
    // Admin
    adminEmail: import.meta.env.ADMIN_EMAIL || 'admin@crescendai.app',
    
    // Email (SMTP)
    smtpSettings: {
      host: import.meta.env.SMTP_HOST || 'localhost',
      port: validateNumber(import.meta.env.SMTP_PORT, 587),
      secure: validateBoolean(import.meta.env.SMTP_SECURE, true),
      user: import.meta.env.SMTP_USER || '',
      password: import.meta.env.SMTP_PASSWORD || '',
    },
  } as const;
}

// Lazy-load server configuration to prevent accidental browser access
let _serverConfig: ServerConfig | null = null;

/**
 * Get server configuration (server-side only)
 * Throws error if accessed in browser context
 */
export function getServerConfig(): ServerConfig {
  if (browser) {
    throw new Error('Server configuration cannot be accessed in browser');
  }
  
  if (!_serverConfig) {
    _serverConfig = createServerConfig();
  }
  
  return _serverConfig;
}

/**
 * Validate critical environment variables on server startup
 * Logs warnings for missing production variables
 */
export function validateEnvironment(): void {
  if (browser) return; // Skip validation in browser
  
  const errors: string[] = [];
  const warnings: string[] = [];
  
  if (!dev) {
    // Production-only validations
    const serverConfig = getServerConfig();
    
    if (!serverConfig.jwtSecret || serverConfig.jwtSecret.length < 32) {
      errors.push('JWT_SECRET must be at least 32 characters long');
    }
    
    if (!serverConfig.encryptionKey || serverConfig.encryptionKey.length !== 32) {
      errors.push('ENCRYPTION_KEY must be exactly 32 characters long');
    }
    
    if (!serverConfig.dbConnectionString) {
      warnings.push('DATABASE_URL is not set');
    }
    
    if (!serverConfig.smtpSettings.host || serverConfig.smtpSettings.host === 'localhost') {
      warnings.push('SMTP configuration is not properly set');
    }
  }
  
  // Log validation results
  if (errors.length > 0) {
    console.error('❌ Environment validation errors:');
    errors.forEach(error => console.error(`  - ${error}`));
    if (!dev) {
      throw new Error('Critical environment variables are missing or invalid');
    }
  }
  
  if (warnings.length > 0) {
    console.warn('⚠️  Environment validation warnings:');
    warnings.forEach(warning => console.warn(`  - ${warning}`));
  }
  
  if (errors.length === 0 && warnings.length === 0) {
    console.info('✅ Environment validation passed');
  }
}

/**
 * Utility function to check if running in development mode
 */
export const isDev = dev;

/**
 * Utility function to check if running in browser context
 */
export const isBrowser = browser;

/**
 * API endpoint builders using validated URLs
 */
export const apiEndpoints = {
  base: clientConfig.apiUrl,
  v1: `${clientConfig.apiUrl}/api/v1`,
  
  // Auth endpoints
  auth: {
    login: `${clientConfig.apiUrl}/api/v1/auth/login`,
    logout: `${clientConfig.apiUrl}/api/v1/auth/logout`,
    refresh: `${clientConfig.apiUrl}/api/v1/auth/refresh`,
    google: `${clientConfig.apiUrl}/api/v1/auth/google`,
  },
  
  // Upload endpoints
  upload: {
    audio: `${clientConfig.apiUrl}/api/v1/upload`,
    presigned: `${clientConfig.apiUrl}/api/v1/upload/presigned`,
  },
  
  // Analysis endpoints
  analysis: {
    start: `${clientConfig.apiUrl}/api/v1/analyze`,
    status: (jobId: string) => `${clientConfig.apiUrl}/api/v1/job/${jobId}`,
    result: (resultId: string) => `${clientConfig.apiUrl}/api/v1/result/${resultId}`,
  },
  
  // Health check
  health: `${clientConfig.apiUrl}/api/v1/health`,
} as const;

// Run validation on module load (server-side only)
if (!browser) {
  validateEnvironment();
}