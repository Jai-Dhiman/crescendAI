/**
 * XSS Protection and Input Sanitization for CrescendAI
 * 
 * Provides comprehensive protection against Cross-Site Scripting (XSS) attacks
 * with input sanitization, output encoding, and CSP integration.
 */

/**
 * HTML entities for encoding
 */
const HTML_ENTITIES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#x27;',
  '/': '&#x2F;',
  '`': '&#x60;',
  '=': '&#x3D;',
} as const;

/**
 * Regular expressions for detecting XSS patterns
 */
const XSS_PATTERNS = [
  // Script tags
  /<script[\s\S]*?>[\s\S]*?<\/script>/gi,
  /<script[\s\S]*?>/gi,
  
  // Event handlers
  /on\w+\s*=\s*["'][^"']*["']/gi,
  /on\w+\s*=\s*[^>\s]+/gi,
  
  // Javascript protocol
  /javascript\s*:/gi,
  /vbscript\s*:/gi,
  
  // Data URLs with scripts
  /data\s*:\s*text\/html/gi,
  /data\s*:\s*application\/javascript/gi,
  
  // HTML comments that might hide scripts
  /<!--[\s\S]*?-->/g,
  
  // Style expressions (IE)
  /expression\s*\(/gi,
  
  // Import statements
  /@import/gi,
  
  // Object/embed tags
  /<(object|embed|applet|iframe)[\s\S]*?>/gi,
  
  // Form elements
  /<form[\s\S]*?>/gi,
  
  // Meta refresh
  /<meta[\s\S]*?http-equiv[\s\S]*?refresh[\s\S]*?>/gi,
] as const;

/**
 * Dangerous HTML tags that should be removed
 */
const DANGEROUS_TAGS = [
  'script', 'iframe', 'object', 'embed', 'applet', 'form', 'input',
  'textarea', 'select', 'button', 'link', 'meta', 'style', 'base'
] as const;

/**
 * Safe HTML tags that are allowed (for rich text)
 */
const SAFE_TAGS = [
  'p', 'br', 'strong', 'b', 'em', 'i', 'u', 'span', 'div',
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
  'blockquote', 'code', 'pre'
] as const;

/**
 * Safe HTML attributes that are allowed
 */
const SAFE_ATTRIBUTES = [
  'class', 'id', 'title', 'lang', 'dir'
] as const;

interface SanitizationOptions {
  allowHtml?: boolean;
  allowedTags?: readonly string[];
  allowedAttributes?: readonly string[];
  maxLength?: number;
  preserveNewlines?: boolean;
}

interface SanitizationResult {
  sanitized: string;
  wasModified: boolean;
  removedContent: string[];
  warnings: string[];
}

/**
 * HTML encode a string to prevent XSS
 */
export function htmlEncode(input: string): string {
  if (typeof input !== 'string') {
    return String(input);
  }
  
  return input.replace(/[&<>"'`=\/]/g, (match) => HTML_ENTITIES[match] || match);
}

/**
 * HTML decode a string
 */
export function htmlDecode(input: string): string {
  if (typeof input !== 'string') {
    return String(input);
  }
  
  const entityMap = Object.fromEntries(
    Object.entries(HTML_ENTITIES).map(([char, entity]) => [entity, char])
  );
  
  return input.replace(/&amp;|&lt;|&gt;|&quot;|&#x27;|&#x2F;|&#x60;|&#x3D;/g, 
    (match) => entityMap[match] || match);
}

/**
 * Detect potential XSS patterns in input
 */
function detectXSS(input: string): string[] {
  const detectedPatterns: string[] = [];
  
  XSS_PATTERNS.forEach((pattern, index) => {
    if (pattern.test(input)) {
      detectedPatterns.push(`Pattern ${index + 1}: ${pattern.source}`);
    }
  });
  
  return detectedPatterns;
}

/**
 * Remove dangerous HTML tags and attributes
 */
function removeDangerousTags(input: string, allowedTags: readonly string[] = SAFE_TAGS): string {
  // Remove dangerous tags entirely
  let cleaned = input;
  
  DANGEROUS_TAGS.forEach(tag => {
    const regex = new RegExp(`<${tag}[^>]*>.*?<\/${tag}>`, 'gi');
    cleaned = cleaned.replace(regex, '');
    
    // Also remove self-closing versions
    const selfClosingRegex = new RegExp(`<${tag}[^>]*\/?>`, 'gi');
    cleaned = cleaned.replace(selfClosingRegex, '');
  });
  
  // Remove any remaining tags that aren't in the allowed list
  cleaned = cleaned.replace(/<\/?([a-zA-Z][a-zA-Z0-9]*)[^>]*>/g, (match, tagName) => {
    if (!allowedTags.includes(tagName.toLowerCase())) {
      return '';
    }
    return match;
  });
  
  return cleaned;
}

/**
 * Remove dangerous attributes from HTML tags
 */
function sanitizeAttributes(input: string, allowedAttributes: readonly string[] = SAFE_ATTRIBUTES): string {
  return input.replace(/<([a-zA-Z][a-zA-Z0-9]*)[^>]*>/g, (match, tagName) => {
    // Extract attributes
    const attributeRegex = /(\w+)\s*=\s*["']([^"']*)["']/g;
    const cleanAttributes: string[] = [];
    let attrMatch;
    
    while ((attrMatch = attributeRegex.exec(match)) !== null) {
      const [, attrName, attrValue] = attrMatch;
      
      // Only allow safe attributes
      if (allowedAttributes.includes(attrName.toLowerCase())) {
        // Sanitize attribute value
        const cleanValue = htmlEncode(attrValue);
        cleanAttributes.push(`${attrName}="${cleanValue}"`);
      }
    }
    
    return cleanAttributes.length > 0 
      ? `<${tagName} ${cleanAttributes.join(' ')}>`
      : `<${tagName}>`;
  });
}

/**
 * Comprehensive input sanitization
 */
export function sanitizeInput(
  input: unknown, 
  options: SanitizationOptions = {}
): SanitizationResult {
  const {
    allowHtml = false,
    allowedTags = SAFE_TAGS,
    allowedAttributes = SAFE_ATTRIBUTES,
    maxLength = 10000,
    preserveNewlines = true,
  } = options;

  // Convert to string
  let sanitized = String(input || '');
  const original = sanitized;
  const removedContent: string[] = [];
  const warnings: string[] = [];

  // Detect XSS patterns
  const xssPatterns = detectXSS(sanitized);
  if (xssPatterns.length > 0) {
    warnings.push(`Potential XSS patterns detected: ${xssPatterns.length}`);
  }

  // Length check
  if (sanitized.length > maxLength) {
    sanitized = sanitized.substring(0, maxLength);
    warnings.push(`Input truncated to ${maxLength} characters`);
  }

  if (allowHtml) {
    // Sanitize HTML while preserving safe tags
    const beforeTags = sanitized;
    sanitized = removeDangerousTags(sanitized, allowedTags);
    
    if (beforeTags !== sanitized) {
      removedContent.push('Dangerous HTML tags');
    }

    const beforeAttrs = sanitized;
    sanitized = sanitizeAttributes(sanitized, allowedAttributes);
    
    if (beforeAttrs !== sanitized) {
      removedContent.push('Dangerous HTML attributes');
    }
  } else {
    // Complete HTML encoding
    sanitized = htmlEncode(sanitized);
  }

  // Preserve newlines if requested
  if (preserveNewlines && !allowHtml) {
    sanitized = sanitized.replace(/\n/g, '<br>');
  }

  // Remove null bytes and other control characters
  const beforeControlChars = sanitized;
  sanitized = sanitized.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
  
  if (beforeControlChars !== sanitized) {
    removedContent.push('Control characters');
  }

  // Normalize whitespace
  sanitized = sanitized.replace(/\s+/g, ' ').trim();

  return {
    sanitized,
    wasModified: original !== sanitized,
    removedContent,
    warnings,
  };
}

/**
 * Sanitize object properties recursively
 */
export function sanitizeObject<T extends Record<string, any>>(
  obj: T,
  options: SanitizationOptions = {}
): T {
  const sanitized = {} as T;
  
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      sanitized[key as keyof T] = sanitizeInput(value, options).sanitized as T[keyof T];
    } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      sanitized[key as keyof T] = sanitizeObject(value, options);
    } else if (Array.isArray(value)) {
      sanitized[key as keyof T] = value.map(item => 
        typeof item === 'string' 
          ? sanitizeInput(item, options).sanitized 
          : typeof item === 'object' && item !== null
            ? sanitizeObject(item, options)
            : item
      ) as T[keyof T];
    } else {
      sanitized[key as keyof T] = value;
    }
  }
  
  return sanitized;
}

/**
 * Create a CSP nonce for inline scripts (server-side)
 */
export function generateCSPNonce(): string {
  // This should be generated server-side
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const array = new Uint8Array(16);
    crypto.getRandomValues(array);
    return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
  }
  
  // Fallback for environments without crypto
  return Math.random().toString(36).substring(2, 15) + 
         Math.random().toString(36).substring(2, 15);
}

/**
 * Validate and sanitize URL to prevent XSS through URLs
 */
export function sanitizeUrl(url: string): string | null {
  if (!url || typeof url !== 'string') {
    return null;
  }

  // Remove any whitespace
  url = url.trim();

  // Block javascript: and data: protocols
  if (/^(javascript|data|vbscript):/i.test(url)) {
    return null;
  }

  // Only allow http, https, mailto, and relative URLs
  if (!/^(https?:\/\/|mailto:|\/|\.\/|#)/.test(url)) {
    return null;
  }

  try {
    // If it's an absolute URL, validate it
    if (/^https?:\/\//.test(url)) {
      const parsed = new URL(url);
      return parsed.toString();
    }
    
    // Return relative URLs as-is after basic sanitization
    return htmlEncode(url);
  } catch {
    return null;
  }
}

/**
 * Sanitize CSS to prevent XSS through style attributes
 */
export function sanitizeCSS(css: string): string {
  if (!css || typeof css !== 'string') {
    return '';
  }

  // Remove dangerous CSS patterns
  const dangerous = [
    /javascript\s*:/gi,
    /expression\s*\(/gi,
    /url\s*\(\s*["']?\s*javascript:/gi,
    /url\s*\(\s*["']?\s*data:/gi,
    /@import/gi,
    /behavior\s*:/gi,
    /-moz-binding/gi,
  ];

  let sanitized = css;
  dangerous.forEach(pattern => {
    sanitized = sanitized.replace(pattern, '');
  });

  return sanitized;
}

/**
 * Rate limiting for form submissions (client-side)
 */
class FormSubmissionRateLimit {
  private submissions = new Map<string, number[]>();
  private readonly maxSubmissions: number;
  private readonly timeWindow: number;

  constructor(maxSubmissions = 5, timeWindowMs = 60000) {
    this.maxSubmissions = maxSubmissions;
    this.timeWindow = timeWindowMs;
  }

  canSubmit(formId: string): { allowed: boolean; error?: string } {
    const now = Date.now();
    const formSubmissions = this.submissions.get(formId) || [];
    
    // Remove old submissions
    const recentSubmissions = formSubmissions.filter(time => now - time < this.timeWindow);
    
    if (recentSubmissions.length >= this.maxSubmissions) {
      const oldestSubmission = Math.min(...recentSubmissions);
      const waitTime = Math.ceil((this.timeWindow - (now - oldestSubmission)) / 1000);
      return {
        allowed: false,
        error: `Form submission rate limit exceeded. Please wait ${waitTime} seconds.`,
      };
    }

    // Record this submission
    recentSubmissions.push(now);
    this.submissions.set(formId, recentSubmissions);
    
    return { allowed: true };
  }

  reset(formId?: string): void {
    if (formId) {
      this.submissions.delete(formId);
    } else {
      this.submissions.clear();
    }
  }
}

// Export singleton rate limiter
export const formRateLimit = new FormSubmissionRateLimit();

/**
 * Secure form validation and sanitization
 */
export interface SecureFormData<T extends Record<string, any>> {
  data: T;
  isValid: boolean;
  errors: Record<keyof T, string[]>;
  warnings: string[];
}

export function validateSecureForm<T extends Record<string, any>>(
  formData: T,
  formId: string,
  options: SanitizationOptions = {}
): SecureFormData<T> {
  // Check rate limiting
  const rateLimitCheck = formRateLimit.canSubmit(formId);
  if (!rateLimitCheck.allowed) {
    return {
      data: formData,
      isValid: false,
      errors: { _form: [rateLimitCheck.error!] } as any,
      warnings: [],
    };
  }

  // Sanitize all form data
  const sanitized = sanitizeObject(formData, options);
  const errors: Record<string, string[]> = {};
  const warnings: string[] = [];

  // Validate each field
  for (const [key, value] of Object.entries(formData)) {
    const fieldErrors: string[] = [];
    
    if (typeof value === 'string') {
      const result = sanitizeInput(value, options);
      if (result.warnings.length > 0) {
        warnings.push(...result.warnings.map(w => `${key}: ${w}`));
      }
      if (result.removedContent.length > 0) {
        fieldErrors.push(`Removed potentially dangerous content: ${result.removedContent.join(', ')}`);
      }
    }
    
    if (fieldErrors.length > 0) {
      errors[key] = fieldErrors;
    }
  }

  return {
    data: sanitized,
    isValid: Object.keys(errors).length === 0,
    errors: errors as Record<keyof T, string[]>,
    warnings,
  };
}

/**
 * Content Security Policy helpers
 */
export const CSPHelpers = {
  /**
   * Generate a secure nonce for inline scripts
   */
  generateNonce: generateCSPNonce,
  
  /**
   * Create CSP-compliant inline script tag
   */
  createSecureScript: (nonce: string, content: string) => {
    const sanitizedContent = sanitizeInput(content, { allowHtml: false }).sanitized;
    return `<script nonce="${nonce}">${sanitizedContent}</script>`;
  },
  
  /**
   * Validate that a string is safe for CSP
   */
  isCSPSafe: (content: string) => {
    const dangerous = [
      /'unsafe-eval'/,
      /'unsafe-inline'/,
      /data:/,
      /javascript:/,
    ];
    
    return !dangerous.some(pattern => pattern.test(content));
  },
} as const;