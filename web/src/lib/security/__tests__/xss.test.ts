/**
 * XSS Protection Tests
 * Tests for input sanitization and XSS prevention utilities
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  htmlEncode,
  htmlDecode,
  sanitizeInput,
  sanitizeObject,
  sanitizeUrl,
  sanitizeCSS,
  validateSecureForm,
  formRateLimit,
} from '../xss';

describe('XSS Protection', () => {
  beforeEach(() => {
    // Reset rate limiter before each test
    formRateLimit.reset();
  });

  describe('htmlEncode', () => {
    it('should encode HTML entities', () => {
      expect(htmlEncode('<script>alert("xss")</script>')).toBe(
        '&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;'
      );
    });

    it('should handle ampersands', () => {
      expect(htmlEncode('Tom & Jerry')).toBe('Tom &amp; Jerry');
    });

    it('should handle quotes and apostrophes', () => {
      expect(htmlEncode(`He said "Hello" and 'Goodbye'`)).toBe(
        'He said &quot;Hello&quot; and &#x27;Goodbye&#x27;'
      );
    });

    it('should handle non-string input', () => {
      expect(htmlEncode(123 as any)).toBe('123');
      expect(htmlEncode(null as any)).toBe('null');
      expect(htmlEncode(undefined as any)).toBe('undefined');
    });
  });

  describe('htmlDecode', () => {
    it('should decode HTML entities', () => {
      expect(htmlDecode('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;')).toBe(
        '<script>alert("xss")</script>'
      );
    });

    it('should handle mixed encoded and unencoded content', () => {
      expect(htmlDecode('Hello &amp; welcome to &lt;CrescendAI&gt;')).toBe(
        'Hello & welcome to <CrescendAI>'
      );
    });
  });

  describe('sanitizeInput', () => {
    it('should sanitize basic XSS attempts', () => {
      const maliciousInput = '<script>alert("xss")</script>Hello World';
      const result = sanitizeInput(maliciousInput);
      
      expect(result.sanitized).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;Hello World');
      expect(result.wasModified).toBe(true);
    });

    it('should preserve safe content when allowHtml is true', () => {
      const safeHtml = '<p>Hello <strong>World</strong></p>';
      const result = sanitizeInput(safeHtml, { allowHtml: true });
      
      expect(result.sanitized).toContain('<p>');
      expect(result.sanitized).toContain('<strong>');
    });

    it('should remove dangerous tags even when allowHtml is true', () => {
      const dangerousHtml = '<p>Safe</p><script>alert("xss")</script>';
      const result = sanitizeInput(dangerousHtml, { allowHtml: true });
      
      expect(result.sanitized).toContain('<p>Safe</p>');
      expect(result.sanitized).not.toContain('<script>');
      expect(result.removedContent).toContain('Dangerous HTML tags');
    });

    it('should enforce maximum length', () => {
      const longInput = 'a'.repeat(1000);
      const result = sanitizeInput(longInput, { maxLength: 100 });
      
      expect(result.sanitized).toHaveLength(100);
      expect(result.warnings).toContain('Input truncated to 100 characters');
    });

    it('should handle newlines correctly', () => {
      const inputWithNewlines = 'Line 1\nLine 2\nLine 3';
      const result = sanitizeInput(inputWithNewlines, { preserveNewlines: true });
      
      expect(result.sanitized).toContain('<br>');
    });

    it('should remove control characters', () => {
      const inputWithControlChars = 'Hello\x00\x01World';
      const result = sanitizeInput(inputWithControlChars);
      
      expect(result.sanitized).toBe('HelloWorld');
      expect(result.removedContent).toContain('Control characters');
    });
  });

  describe('sanitizeObject', () => {
    it('should sanitize all string properties', () => {
      const maliciousObject = {
        name: '<script>alert("xss")</script>John',
        email: 'test@example.com',
        comment: 'Hello <b>World</b>',
        age: 25,
      };

      const result = sanitizeObject(maliciousObject);
      
      expect(result.name).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;John');
      expect(result.email).toBe('test@example.com');
      expect(result.comment).toBe('Hello &lt;b&gt;World&lt;&#x2F;b&gt;');
      expect(result.age).toBe(25);
    });

    it('should handle nested objects', () => {
      const nestedObject = {
        user: {
          profile: {
            bio: '<script>alert("xss")</script>',
          },
        },
      };

      const result = sanitizeObject(nestedObject);
      expect(result.user.profile.bio).toBe('&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;');
    });

    it('should handle arrays', () => {
      const objectWithArray = {
        tags: ['<script>', 'safe-tag', '<img onerror="alert()">'],
      };

      const result = sanitizeObject(objectWithArray);
      expect(result.tags[0]).toBe('&lt;script&gt;');
      expect(result.tags[1]).toBe('safe-tag');
      expect(result.tags[2]).toBe('&lt;img onerror&#x3D;&quot;alert()&quot;&gt;');
    });
  });

  describe('sanitizeUrl', () => {
    it('should allow safe URLs', () => {
      expect(sanitizeUrl('https://example.com')).toBe('https://example.com');
      expect(sanitizeUrl('http://localhost:3000')).toBe('http://localhost:3000');
      expect(sanitizeUrl('/relative/path')).toBe('&#x2F;relative&#x2F;path');
      expect(sanitizeUrl('./relative')).toBe('.&#x2F;relative');
      expect(sanitizeUrl('#anchor')).toBe('#anchor');
      expect(sanitizeUrl('mailto:test@example.com')).toBe('mailto:test@example.com');
    });

    it('should block dangerous protocols', () => {
      expect(sanitizeUrl('javascript:alert("xss")')).toBe(null);
      expect(sanitizeUrl('data:text/html,<script>alert("xss")</script>')).toBe(null);
      expect(sanitizeUrl('vbscript:msgbox("xss")')).toBe(null);
    });

    it('should handle invalid URLs', () => {
      expect(sanitizeUrl('')).toBe(null);
      expect(sanitizeUrl('not-a-url')).toBe(null);
      expect(sanitizeUrl('http://[invalid')).toBe(null);
    });
  });

  describe('sanitizeCSS', () => {
    it('should remove dangerous CSS patterns', () => {
      expect(sanitizeCSS('color: red; background: url(javascript:alert("xss"));')).toBe(
        'color: red; background: url();'
      );
      
      expect(sanitizeCSS('width: expression(alert("xss"));')).toBe('width: ;');
      
      expect(sanitizeCSS('@import "javascript:alert()";')).toBe(' "javascript:alert()";');
    });

    it('should preserve safe CSS', () => {
      const safeCSS = 'color: #ff0000; background-color: blue; margin: 10px;';
      expect(sanitizeCSS(safeCSS)).toBe(safeCSS);
    });
  });

  describe('validateSecureForm', () => {
    it('should validate and sanitize form data', () => {
      const formData = {
        name: 'John <script>alert("xss")</script>',
        email: 'test@example.com',
        message: 'Hello World',
      };

      const result = validateSecureForm(formData, 'test-form');
      
      expect(result.isValid).toBe(true);
      expect(result.data.name).toBe('John &lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;');
      expect(result.data.email).toBe('test@example.com');
      expect(result.data.message).toBe('Hello World');
    });

    it('should enforce rate limiting', () => {
      const formData = { message: 'test' };
      
      // First few submissions should work
      for (let i = 0; i < 5; i++) {
        const result = validateSecureForm(formData, 'rate-limit-test');
        expect(result.isValid).toBe(true);
      }
      
      // Next submission should be rate limited
      const result = validateSecureForm(formData, 'rate-limit-test');
      expect(result.isValid).toBe(false);
      expect(result.errors._form).toContain('Form submission rate limit exceeded');
    });

    it('should detect and report removed content', () => {
      const formData = {
        comment: '<script>alert("xss")</script>Hello <iframe src="evil.com"></iframe>',
      };

      const result = validateSecureForm(formData, 'content-test', { allowHtml: true });
      
      expect(result.isValid).toBe(false);
      expect(result.errors.comment).toContain('Removed potentially dangerous content');
    });
  });

  describe('Edge cases and security', () => {
    it('should handle null and undefined inputs safely', () => {
      expect(sanitizeInput(null).sanitized).toBe('');
      expect(sanitizeInput(undefined).sanitized).toBe('');
      expect(sanitizeInput('').sanitized).toBe('');
    });

    it('should handle deeply nested XSS attempts', () => {
      const deepXSS = '<div><span><script>alert("nested")</script></span></div>';
      const result = sanitizeInput(deepXSS, { allowHtml: true });
      
      expect(result.sanitized).not.toContain('<script>');
    });

    it('should handle unicode and special characters', () => {
      const unicodeInput = '🎵 Music analysis with émotions';
      const result = sanitizeInput(unicodeInput);
      
      expect(result.sanitized).toBe('🎵 Music analysis with émotions');
      expect(result.wasModified).toBe(false);
    });

    it('should handle very large inputs gracefully', () => {
      const largeInput = 'a'.repeat(100000);
      const result = sanitizeInput(largeInput, { maxLength: 1000 });
      
      expect(result.sanitized).toHaveLength(1000);
      expect(result.warnings).toContain('Input truncated to 1000 characters');
    });
  });
});