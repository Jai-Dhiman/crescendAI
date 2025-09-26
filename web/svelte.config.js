import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://svelte.dev/docs/kit/integrations
  // for more information about preprocessors
  preprocess: vitePreprocess(),

  kit: {
    // Static export for Cloudflare Pages (static hosting)
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html', // enable SPA fallback for dynamic routes
      strict: false
    }),
    prerender: {
      // Export all routable pages for a static site demo
      entries: ['*']
    }
  }
};

export default config;
