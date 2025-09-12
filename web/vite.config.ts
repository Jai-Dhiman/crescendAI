import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, type UserConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';
import { visualizer } from 'rollup-plugin-visualizer';
import { sentryVitePlugin } from '@sentry/vite-plugin';

export default defineConfig(({ mode }): UserConfig => {
	const isDev = mode === 'development';
	const isAnalyze = mode === 'analyze';
	
	const config: UserConfig = {
		plugins: [
			sveltekit(),
			
			// Progressive Web App
			VitePWA({
				registerType: 'autoUpdate',
				workbox: {
					globPatterns: ['**/*.{js,css,html,ico,png,svg,webp,woff,woff2}'],
					runtimeCaching: [
						{
							urlPattern: /^https:\/\/api\.pianoanalyzer\.com\//,
							handler: 'StaleWhileRevalidate',
							options: {
								cacheName: 'api-cache',
								expiration: {
									maxEntries: 100,
									maxAgeSeconds: 60 * 60 * 24, // 24 hours
								},
								cacheKeyWillBeUsed: async ({ request }) => {
									// Cache API responses but not authenticated ones
									const url = new URL(request.url);
									if (url.pathname.includes('/auth/') || 
										url.pathname.includes('/user/')) {
										return null;
									}
									return request.url;
								},
							},
						},
						{
							urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp)$/,
							handler: 'CacheFirst',
							options: {
								cacheName: 'images-cache',
								expiration: {
									maxEntries: 200,
									maxAgeSeconds: 60 * 60 * 24 * 30, // 30 days
								},
							},
						},
						{
							urlPattern: /\.(?:woff|woff2|ttf|eot)$/,
							handler: 'CacheFirst',
							options: {
								cacheName: 'fonts-cache',
								expiration: {
									maxEntries: 50,
									maxAgeSeconds: 60 * 60 * 24 * 365, // 1 year
								},
							},
						},
					],
				},
				manifest: {
					name: 'CrescendAI',
					short_name: 'CrescendAI',
					description: 'AI-powered piano performance analysis',
					theme_color: '#1e40af',
					background_color: '#ffffff',
					display: 'standalone',
					orientation: 'portrait',
					scope: '/',
					start_url: '/',
					icons: [
						{
							src: '/pwa-192x192.png',
							sizes: '192x192',
							type: 'image/png'
						},
						{
							src: '/pwa-512x512.png',
							sizes: '512x512',
							type: 'image/png'
						},
						{
							src: '/pwa-512x512.png',
							sizes: '512x512',
							type: 'image/png',
							purpose: 'any maskable'
						}
					],
				},
				devOptions: {
					enabled: isDev,
					type: 'module',
				},
			}),
			
			// Bundle analyzer
			isAnalyze && visualizer({
				filename: 'dist/stats.html',
				open: true,
				gzipSize: true,
				brotliSize: true,
				template: 'treemap',
			}),
			
			// Sentry for error monitoring in production
			!isDev && sentryVitePlugin({
				org: process.env.SENTRY_ORG,
				project: process.env.SENTRY_PROJECT,
				authToken: process.env.SENTRY_AUTH_TOKEN,
				sourceMaps: {
					assets: './dist/**',
					ignore: ['node_modules/**'],
					filesToDeleteAfterUpload: './dist/**/*.map',
				},
				telemetry: false,
			}),
		].filter(Boolean),
		
		// Development server configuration
		server: {
			port: 5173,
			host: true,
			fs: {
				allow: ['..'],
			},
		},
		
		// Preview server configuration
		preview: {
			port: 4173,
			host: true,
		},
		
		// Build optimizations
		build: {
			target: 'es2020',
			sourcemap: !isDev,
			minify: 'terser',
			cssMinify: true,
			
			// Optimize chunks
			rollupOptions: {
				output: {
					// Optimize chunk naming
					chunkFileNames: (chunkInfo) => {
						const facadeModuleId = chunkInfo.facadeModuleId
							? chunkInfo.facadeModuleId.split('/').pop()?.replace(/\.[^.]+$/, '')
							: 'chunk';
						return `chunks/[name]-[hash].js`;
					},
					entryFileNames: 'entries/[name]-[hash].js',
					assetFileNames: (assetInfo) => {
						const info = assetInfo.name!.split('.');
						const ext = info[info.length - 1];
						if (/\.(png|jpe?g|svg|gif|tiff|bmp|ico)$/i.test(assetInfo.name!)) {
							return 'images/[name]-[hash].[ext]';
						}
						if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name!)) {
							return 'fonts/[name]-[hash].[ext]';
						}
						if (/\.css$/i.test(assetInfo.name!)) {
							return 'styles/[name]-[hash].[ext]';
						}
						return 'assets/[name]-[hash].[ext]';
					},
				},
				
				// External dependencies (don't bundle)
				external: [],
			},
			
			// Terser options for better minification
			terserOptions: {
				compress: {
					drop_console: !isDev,
					drop_debugger: !isDev,
					pure_funcs: !isDev ? ['console.log', 'console.debug'] : [],
				},
				mangle: {
					safari10: true,
				},
				format: {
					safari10: true,
				},
			},
			
			// Asset optimization
			assetsInlineLimit: 4096, // 4kb
			chunkSizeWarningLimit: 500, // 500kb
		},
		
		// CSS optimization
		css: {
			devSourcemap: isDev,
		},
		
		// Dependency optimization
		optimizeDeps: {
			include: [
				'@tanstack/svelte-query',
				'@tanstack/svelte-form',
				'lucide-svelte',
				'zustand',
				'nanoid',
				'zod',
			],
			exclude: [
				'@sveltejs/kit',
			],
		},
		
		// Define global constants
		define: {
			__BUILD_TIME__: JSON.stringify(new Date().toISOString()),
			__VERSION__: JSON.stringify(process.env.npm_package_version || '1.0.0'),
		},
		
		// Vite-specific performance settings
		esbuild: {
			target: 'es2020',
			legalComments: 'none',
			minifyIdentifiers: !isDev,
			minifySyntax: !isDev,
			minifyWhitespace: !isDev,
		},
		
		// Asset handling
		assetsInclude: ['**/*.md'],
		
		// Environment variables
		envPrefix: ['VITE_', 'PUBLIC_'],
	};
	
	return config;
});
