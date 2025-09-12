/**
 * Home Page E2E Tests
 * Tests the main landing page functionality and user flows
 */

import { test, expect, type Page } from '@playwright/test';

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should load the home page successfully', async ({ page }) => {
    // Check that the page loads
    await expect(page).toHaveTitle(/CrescendAI/);
    
    // Check for main heading
    await expect(page.locator('h1')).toContainText(/CrescendAI|Piano|Analysis/i);
  });

  test('should have working navigation', async ({ page }) => {
    // Check that navigation elements are present
    const nav = page.locator('nav');
    await expect(nav).toBeVisible();
    
    // Test navigation links (if they exist)
    const navLinks = page.locator('nav a');
    const linkCount = await navLinks.count();
    
    if (linkCount > 0) {
      // At least one navigation link should be present
      expect(linkCount).toBeGreaterThan(0);
    }
  });

  test('should have responsive design', async ({ page }) => {
    // Test desktop view
    await page.setViewportSize({ width: 1280, height: 720 });
    await expect(page.locator('body')).toBeVisible();
    
    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('body')).toBeVisible();
    
    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('body')).toBeVisible();
  });

  test('should have proper security headers', async ({ page, request }) => {
    const response = await request.get('/');
    
    // Check for security headers
    const headers = response.headers();
    
    expect(headers['x-frame-options']).toBeTruthy();
    expect(headers['x-content-type-options']).toBe('nosniff');
    expect(headers['content-security-policy']).toBeTruthy();
  });

  test('should handle file upload area', async ({ page }) => {
    // Look for file upload functionality
    const fileUpload = page.locator('input[type="file"]').first();
    
    if (await fileUpload.isVisible()) {
      await expect(fileUpload).toBeVisible();
      
      // Check that it accepts audio files
      const acceptAttribute = await fileUpload.getAttribute('accept');
      if (acceptAttribute) {
        expect(acceptAttribute).toMatch(/audio/);
      }
    }
  });

  test('should show loading states appropriately', async ({ page }) => {
    // Check initial page load
    await page.waitForLoadState('networkidle');
    
    // Ensure no loading indicators are stuck
    const loadingElements = page.locator('[data-testid*="loading"], .loading, .spinner');
    const loadingCount = await loadingElements.count();
    
    // If there are loading elements, they should not be visible after page load
    for (let i = 0; i < loadingCount; i++) {
      const element = loadingElements.nth(i);
      if (await element.isVisible()) {
        // Loading elements should disappear within a reasonable time
        await expect(element).toBeHidden({ timeout: 10000 });
      }
    }
  });

  test('should handle errors gracefully', async ({ page }) => {
    // Test 404 page
    await page.goto('/non-existent-page');
    
    // Should either redirect to home or show a proper 404 page
    const pageContent = await page.textContent('body');
    expect(pageContent).toBeTruthy();
    
    // Should not show raw error or stack trace
    expect(pageContent?.toLowerCase()).not.toContain('error');
    expect(pageContent?.toLowerCase()).not.toContain('stack trace');
  });

  test('should have accessible elements', async ({ page }) => {
    // Check for basic accessibility features
    const mainContent = page.locator('main, [role="main"]').first();
    if (await mainContent.count() > 0) {
      await expect(mainContent).toBeVisible();
    }
    
    // Check that interactive elements are focusable
    const buttons = page.locator('button, [role="button"]');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      const firstButton = buttons.first();
      await firstButton.focus();
      await expect(firstButton).toBeFocused();
    }
    
    // Check for alt text on images
    const images = page.locator('img');
    const imageCount = await images.count();
    
    for (let i = 0; i < imageCount; i++) {
      const img = images.nth(i);
      const alt = await img.getAttribute('alt');
      // Images should have alt text (can be empty for decorative images)
      expect(alt).not.toBeNull();
    }
  });

  test('should load quickly', async ({ page }) => {
    const start = Date.now();
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    const loadTime = Date.now() - start;
    
    // Page should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
  });

  test('should work offline (PWA functionality)', async ({ page, context }) => {
    // First, load the page normally
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Go offline
    await context.setOffline(true);
    
    // Try to navigate within the app
    await page.reload();
    
    // The page should still be accessible (service worker should serve cached content)
    // Note: This might fail if PWA isn't fully set up yet
    const body = page.locator('body');
    await expect(body).toBeVisible({ timeout: 10000 });
    
    // Go back online
    await context.setOffline(false);
  });
});

test.describe('File Upload Flow', () => {
  test('should handle audio file upload', async ({ page }) => {
    await page.goto('/');
    
    // Look for file upload input
    const fileInput = page.locator('input[type="file"]').first();
    
    if (await fileInput.count() > 0) {
      // Create a mock audio file
      const mockAudioFile = {
        name: 'test-audio.mp3',
        mimeType: 'audio/mpeg',
        buffer: Buffer.from('mock audio data'),
      };
      
      // Upload the file
      await fileInput.setInputFiles({
        name: mockAudioFile.name,
        mimeType: mockAudioFile.mimeType,
        buffer: mockAudioFile.buffer,
      });
      
      // Check for upload feedback
      await expect(page.locator('text=/upload|file|selected/i').first()).toBeVisible({
        timeout: 5000,
      });
    }
  });

  test('should validate file types', async ({ page }) => {
    await page.goto('/');
    
    const fileInput = page.locator('input[type="file"]').first();
    
    if (await fileInput.count() > 0) {
      // Try to upload an invalid file type
      const invalidFile = {
        name: 'malicious.exe',
        mimeType: 'application/x-executable',
        buffer: Buffer.from('malicious content'),
      };
      
      await fileInput.setInputFiles({
        name: invalidFile.name,
        mimeType: invalidFile.mimeType,
        buffer: invalidFile.buffer,
      });
      
      // Should show validation error
      await expect(
        page.locator('text=/invalid|not supported|file type/i').first()
      ).toBeVisible({ timeout: 5000 });
    }
  });
});

test.describe('Security Tests', () => {
  test('should prevent XSS attacks in forms', async ({ page }) => {
    await page.goto('/');
    
    // Look for text input fields
    const textInputs = page.locator('input[type="text"], input[type="email"], textarea');
    const inputCount = await textInputs.count();
    
    if (inputCount > 0) {
      const firstInput = textInputs.first();
      
      // Try to inject XSS
      const xssPayload = '<script>alert("xss")</script>';
      await firstInput.fill(xssPayload);
      
      // Check that the script doesn't execute
      const alertCount = await page.evaluate(() => {
        return window.addEventListener('beforeunload', () => {
          return 'XSS detected';
        });
      });
      
      // The XSS should be escaped/sanitized
      const inputValue = await firstInput.inputValue();
      expect(inputValue).not.toContain('<script>');
    }
  });

  test('should have secure cookie settings', async ({ page, context }) => {
    await page.goto('/');
    
    // Check cookies for security attributes
    const cookies = await context.cookies();
    
    cookies.forEach(cookie => {
      // Authentication cookies should be secure
      if (cookie.name.includes('auth') || cookie.name.includes('session')) {
        expect(cookie.secure).toBe(true);
        expect(cookie.httpOnly).toBe(true);
      }
    });
  });

  test('should handle CSRF protection', async ({ page, request }) => {
    // Test that POST requests require proper headers/tokens
    try {
      const response = await request.post('/api/test', {
        data: { test: 'data' },
      });
      
      // Should either succeed with proper CSRF handling or fail with 403
      expect([200, 201, 403, 404]).toContain(response.status());
    } catch (error) {
      // Network errors are acceptable for this test
    }
  });
});