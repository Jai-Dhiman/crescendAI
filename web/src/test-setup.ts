/**
 * Vitest Setup File
 * Configures the testing environment with necessary polyfills and mocks
 */

import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock browser APIs that might be missing in test environment
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// Mock Web APIs that might not be available in jsdom
Object.defineProperty(window, 'navigator', {
  writable: true,
  value: {
    ...window.navigator,
    clipboard: {
      writeText: vi.fn().mockResolvedValue(undefined),
      readText: vi.fn().mockResolvedValue(''),
    },
  },
});

// Mock crypto.subtle for encryption tests
Object.defineProperty(global, 'crypto', {
  value: {
    getRandomValues: (arr: any) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i] = Math.floor(Math.random() * 256);
      }
      return arr;
    },
    randomUUID: () => {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    },
    subtle: {
      digest: vi.fn().mockResolvedValue(new ArrayBuffer(32)),
      encrypt: vi.fn().mockResolvedValue(new ArrayBuffer(16)),
      decrypt: vi.fn().mockResolvedValue(new ArrayBuffer(16)),
      generateKey: vi.fn().mockResolvedValue({}),
      importKey: vi.fn().mockResolvedValue({}),
      exportKey: vi.fn().mockResolvedValue(new ArrayBuffer(32)),
    },
  },
});

// Mock localStorage and sessionStorage
const createStorageMock = () => {
  let store: Record<string, string> = {};
  
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
    length: 0,
    key: vi.fn(() => null),
  };
};

Object.defineProperty(window, 'localStorage', {
  value: createStorageMock(),
});

Object.defineProperty(window, 'sessionStorage', {
  value: createStorageMock(),
});

// Mock File and FileReader APIs
global.File = class MockFile extends Blob {
  name: string;
  lastModified: number;
  
  constructor(chunks: BlobPart[], filename: string, options?: FilePropertyBag) {
    super(chunks, options);
    this.name = filename;
    this.lastModified = options?.lastModified || Date.now();
  }
} as any;

global.FileReader = class MockFileReader extends EventTarget {
  result: string | ArrayBuffer | null = null;
  error: DOMException | null = null;
  readyState: number = 0;
  
  onload: ((this: FileReader, ev: ProgressEvent<FileReader>) => any) | null = null;
  onerror: ((this: FileReader, ev: ProgressEvent<FileReader>) => any) | null = null;
  onabort: ((this: FileReader, ev: ProgressEvent<FileReader>) => any) | null = null;
  onloadstart: ((this: FileReader, ev: ProgressEvent<FileReader>) => any) | null = null;
  onloadend: ((this: FileReader, ev: ProgressEvent<FileReader>) => any) | null = null;
  onprogress: ((this: FileReader, ev: ProgressEvent<FileReader>) => any) | null = null;
  
  readAsText = vi.fn().mockImplementation(() => {
    setTimeout(() => {
      this.result = 'mock file content';
      this.readyState = 2;
      this.onload?.call(this, new ProgressEvent('load'));
    }, 0);
  });
  
  readAsDataURL = vi.fn().mockImplementation(() => {
    setTimeout(() => {
      this.result = 'data:text/plain;base64,bW9jayBmaWxlIGNvbnRlbnQ=';
      this.readyState = 2;
      this.onload?.call(this, new ProgressEvent('load'));
    }, 0);
  });
  
  readAsArrayBuffer = vi.fn().mockImplementation(() => {
    setTimeout(() => {
      this.result = new ArrayBuffer(16);
      this.readyState = 2;
      this.onload?.call(this, new ProgressEvent('load'));
    }, 0);
  });
  
  abort = vi.fn();
} as any;

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
global.URL.revokeObjectURL = vi.fn();

// Mock fetch for API tests
global.fetch = vi.fn();

// Environment variables for tests
process.env.PUBLIC_API_URL = 'http://localhost:8787';
process.env.PUBLIC_APP_NAME = 'CrescendAI';
process.env.PUBLIC_ENABLE_ANALYTICS = 'false';
process.env.PUBLIC_DEBUG_LOGGING = 'true';

// Console overrides for test environment
const originalConsoleError = console.error;
console.error = (...args) => {
  // Suppress known React/Testing Library warnings that don't affect tests
  const message = args[0];
  if (
    typeof message === 'string' &&
    (message.includes('Warning: ReactDOM.render is no longer supported') ||
     message.includes('Warning: validateDOMNesting'))
  ) {
    return;
  }
  originalConsoleError(...args);
};

// Reset all mocks before each test
beforeEach(() => {
  vi.clearAllMocks();
});

// Global test helpers
export const createMockFile = (
  name: string = 'test.mp3',
  type: string = 'audio/mpeg',
  content: string = 'mock audio content'
): File => {
  return new File([content], name, { type });
};

export const createMockEvent = <T extends Event>(
  type: string,
  eventInitDict?: EventInit
): T => {
  return new Event(type, eventInitDict) as T;
};

export const waitFor = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

// Declare global test utilities
declare global {
  var createMockFile: typeof createMockFile;
  var createMockEvent: typeof createMockEvent;
  var waitFor: typeof waitFor;
}

global.createMockFile = createMockFile;
global.createMockEvent = createMockEvent;
global.waitFor = waitFor;