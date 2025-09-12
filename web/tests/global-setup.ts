/**
 * Playwright Global Setup
 * Runs once before all tests to prepare the environment
 */

import type { FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting Playwright global setup...');
  
  // Set up test environment variables
  process.env.NODE_ENV = 'test';
  process.env.PUBLIC_API_URL = 'http://localhost:8787';
  process.env.PUBLIC_ENABLE_ANALYTICS = 'false';
  process.env.PUBLIC_DEBUG_LOGGING = 'true';
  
  // Clean up any previous test artifacts
  console.log('🧹 Cleaning up previous test artifacts...');
  
  // You can add database seeding, API server setup, etc. here
  
  console.log('✅ Playwright global setup completed');
}

export default globalSetup;