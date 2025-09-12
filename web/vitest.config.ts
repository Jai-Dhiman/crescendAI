import { defineConfig } from 'vitest/config';
import { sveltekit } from '@sveltejs/kit/vite';

export default defineConfig({
	plugins: [sveltekit()],
	
	test: {
		// Test environment
		environment: 'jsdom',
		
		// Setup files
		setupFiles: ['./src/test-setup.ts'],
		
		// Include/exclude patterns
		include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
		exclude: ['node_modules', 'dist', '.svelte-kit'],
		
		// Global test configuration
		globals: true,
		
		// Coverage configuration
		coverage: {
			provider: 'v8',
			reporter: ['text', 'json', 'html'],
			reportsDirectory: './coverage',
			exclude: [
				'node_modules/',
				'src/test-setup.ts',
				'src/**/*.d.ts',
				'src/**/*.config.*',
				'dist/',
				'.svelte-kit/',
				'coverage/',
				'playwright/',
				'tests/',
			],
			include: ['src/**/*.{js,ts,svelte}'],
			thresholds: {
				global: {
					branches: 80,
					functions: 80,
					lines: 80,
					statements: 80,
				},
			},
		},
		
		// Watch configuration
		watch: {
			exclude: ['node_modules/**', 'dist/**', '.svelte-kit/**'],
		},
		
		// Performance settings
		testTimeout: 10000,
		hookTimeout: 10000,
		
		// Reporter configuration
		reporter: ['default', 'junit'],
		outputFile: {
			junit: './test-results/junit.xml',
		},
		
		// Benchmark configuration
		benchmark: {
			include: ['src/**/*.bench.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
			exclude: ['node_modules', 'dist', '.svelte-kit'],
		},
	},
	
	// Resolve configuration
	resolve: {
		alias: {
			$lib: './src/lib',
			$app: './.svelte-kit/runtime/app',
		},
	},
	
	// Define global constants for tests
	define: {
		__TEST__: true,
		__BUILD_TIME__: JSON.stringify(new Date().toISOString()),
		__VERSION__: JSON.stringify('test'),
	},
});