import { browser } from '$app/environment';
import { secureStorage, sanitizeForStorage } from '../security/encryption';

// Web storage wrapper with encryption for sensitive data
interface StorageInterface {
  getString: (key: string) => string | null;
  setString: (key: string, value: string) => void;
  getBoolean: (key: string) => boolean | null;
  setBoolean: (key: string, value: boolean) => void;
  getNumber: (key: string) => number | null;
  setNumber: (key: string, value: number) => void;
  delete: (key: string) => void;
  clearAll: () => void;
  // New secure methods
  setSecure: (key: string, value: any) => void;
  getSecure: <T = any>(key: string) => T | null;
  deleteSecure: (key: string) => void;
}

class WebStorage implements StorageInterface {
  private sensitiveKeys = new Set([
    'auth-storage',
    'user-tokens',
    'refresh-token',
    'access-token',
    'user-session',
    'api-keys',
    'credentials'
  ]);

  private isSensitiveKey(key: string): boolean {
    return this.sensitiveKeys.has(key) || 
           key.includes('token') || 
           key.includes('auth') ||
           key.includes('session') ||
           key.includes('credential');
  }

  getString(key: string): string | null {
    try {
      // Use secure storage for sensitive data
      if (this.isSensitiveKey(key) && secureStorage) {
        return secureStorage.getItem(key);
      }
      
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  setString(key: string, value: string): void {
    try {
      // Use secure storage for sensitive data
      if (this.isSensitiveKey(key) && secureStorage) {
        secureStorage.setItem(key, value);
        return;
      }
      
      localStorage.setItem(key, value);
    } catch {
      // Handle storage full or disabled
    }
  }

  getBoolean(key: string): boolean | null {
    const value = this.getString(key);
    if (value === null) return null;
    return value === 'true';
  }

  setBoolean(key: string, value: boolean): void {
    this.setString(key, value.toString());
  }

  getNumber(key: string): number | null {
    const value = this.getString(key);
    if (value === null) return null;
    const parsed = Number(value);
    return isNaN(parsed) ? null : parsed;
  }

  setNumber(key: string, value: number): void {
    this.setString(key, value.toString());
  }

  delete(key: string): void {
    try {
      // Delete from both regular and secure storage
      if (this.isSensitiveKey(key) && secureStorage) {
        secureStorage.removeItem(key);
      }
      localStorage.removeItem(key);
    } catch {
      // Handle error
    }
  }

  clearAll(): void {
    try {
      localStorage.clear();
      if (secureStorage) {
        secureStorage.clear();
      }
    } catch {
      // Handle error
    }
  }

  // New secure storage methods
  setSecure(key: string, value: any): void {
    if (!secureStorage) {
      console.warn('Secure storage not available, falling back to regular storage');
      this.setString(key, JSON.stringify(value));
      return;
    }

    try {
      const sanitizedData = sanitizeForStorage(value);
      secureStorage.setItem(key, sanitizedData);
    } catch (error) {
      console.error('Secure storage failed:', error);
      // Fallback to regular storage for non-sensitive data
      if (!this.isSensitiveKey(key)) {
        this.setString(key, JSON.stringify(value));
      }
    }
  }

  getSecure<T = any>(key: string): T | null {
    if (!secureStorage) {
      const fallback = this.getString(key);
      if (!fallback) return null;
      try {
        return JSON.parse(fallback);
      } catch {
        return fallback as T;
      }
    }

    try {
      const data = secureStorage.getItem(key);
      if (!data) return null;
      return JSON.parse(data);
    } catch (error) {
      console.error('Secure retrieval failed:', error);
      this.deleteSecure(key); // Remove corrupted data
      return null;
    }
  }

  deleteSecure(key: string): void {
    if (secureStorage) {
      secureStorage.removeItem(key);
    }
    this.delete(key); // Also remove from regular storage
  }
}

// Mock storage for SSR/testing
class MockStorage implements StorageInterface {
  private data = new Map<string, string>();

  getString(key: string): string | null {
    return this.data.get(key) ?? null;
  }

  setString(key: string, value: string): void {
    this.data.set(key, value);
  }

  getBoolean(key: string): boolean | null {
    const value = this.getString(key);
    if (value === null) return null;
    return value === 'true';
  }

  setBoolean(key: string, value: boolean): void {
    this.setString(key, value.toString());
  }

  getNumber(key: string): number | null {
    const value = this.getString(key);
    if (value === null) return null;
    const parsed = Number(value);
    return isNaN(parsed) ? null : parsed;
  }

  setNumber(key: string, value: number): void {
    this.setString(key, value.toString());
  }

  delete(key: string): void {
    this.data.delete(key);
  }

  clearAll(): void {
    this.data.clear();
  }
}

// Export storage instance
export const storage: StorageInterface = typeof window !== 'undefined' 
  ? new WebStorage() 
  : new MockStorage();