import { type Handle } from "@sveltejs/kit";
import { nanoid } from "nanoid";
import { dev } from "$app/environment";

/**
 * Security Configuration
 * Implements comprehensive security headers for the CrescendAI web application
 */
const allowEval = dev || process.env.ALLOW_UNSAFE_EVAL === "1";

const SECURITY_CONFIG = {
	// Content Security Policy sources
	CSP: {
		// Allow self for basic resources
		default: "'self'",

		// Script sources - tuned for SvelteKit/Svelte 5 runtime and dev tools
		script: [
			"'self'",
			// Allow inline scripts in development
			dev ? "'unsafe-inline'" : "",
			// Allow eval when needed by runtime or dev tools (can be disabled via ALLOW_UNSAFE_EVAL)
			allowEval ? "'unsafe-eval'" : "",
			allowEval ? "'wasm-unsafe-eval'" : "",
			// Google Fonts and analytics if needed
			"https://www.google.com",
			"https://www.gstatic.com",
		].filter(Boolean),

		// Style sources - allow Google Fonts
		style: [
			"'self'",
			"'unsafe-inline'", // Required for Svelte and TailwindCSS
			"https://fonts.googleapis.com",
			"https://fonts.gstatic.com",
		],

		// Font sources
		font: [
			"'self'",
			"https://fonts.googleapis.com",
			"https://fonts.gstatic.com",
			"data:", // Allow data URIs for fonts
		],

		// Image sources
		img: [
			"'self'",
			"data:", // Allow data URIs for images
			"blob:", // Allow blob URLs for uploaded images
			"https:", // Allow HTTPS images (for user avatars, etc.)
		],

		// Media sources for audio files
		media: [
			"'self'",
			"blob:", // Allow blob URLs for audio playback
			"data:", // Allow data URIs
		],

		// Connect sources for API calls
		connect: [
			"'self'",
			// Development API
			dev ? "http://localhost:*" : "",
			// Production API (backend)
			"https://crescendai-backend.jai-d.workers.dev",
			// If/when you add a custom domain for the API, include it here:
			"https://api.crescend.ai",
			// WebSocket connections if needed
			dev ? "ws://localhost:*" : "",
			"wss://*.pianoanalyzer.com",
		].filter(Boolean),

		// Worker sources
		worker: [
			"'self'",
			"blob:", // Allow blob workers
		],

		// Object sources
		object: ["'none'"], // Disable plugins for security

		// Base URI restriction
		base: ["'self'"],

		// Form action restrictions
		formAction: ["'self'"],

		// Frame ancestors (anti-clickjacking)
		frameAncestors: ["'none'"],
	},

	// HSTS (HTTP Strict Transport Security) configuration
	HSTS: {
		maxAge: 31536000, // 1 year
		includeSubDomains: true,
		preload: true,
	},

	// Referrer Policy
	REFERRER_POLICY: "strict-origin-when-cross-origin",

	// Permissions Policy (Feature Policy)
	PERMISSIONS_POLICY: {
		microphone: ["self"], // Allow microphone for audio recording
		camera: ["none"], // Disable camera
		geolocation: ["none"], // Disable geolocation
		payment: ["none"], // Disable payment APIs
		usb: ["none"], // Disable USB access
	},
} as const;

/**
 * Generate CSP header value from configuration
 */
function generateCSP(nonce?: string): string {
	const directives = [
		`default-src ${SECURITY_CONFIG.CSP.default}`,
		`script-src ${SECURITY_CONFIG.CSP.script.join(" ")}${!dev && nonce ? ` 'nonce-${nonce}'` : ""}`,
		`style-src ${SECURITY_CONFIG.CSP.style.join(" ")}`,
		`font-src ${SECURITY_CONFIG.CSP.font.join(" ")}`,
		`img-src ${SECURITY_CONFIG.CSP.img.join(" ")}`,
		`media-src ${SECURITY_CONFIG.CSP.media.join(" ")}`,
		`connect-src ${SECURITY_CONFIG.CSP.connect.join(" ")}`,
		`worker-src ${SECURITY_CONFIG.CSP.worker.join(" ")}`,
		`object-src ${SECURITY_CONFIG.CSP.object.join(" ")}`,
		`base-uri ${SECURITY_CONFIG.CSP.base.join(" ")}`,
		`form-action ${SECURITY_CONFIG.CSP.formAction.join(" ")}`,
		`frame-ancestors ${SECURITY_CONFIG.CSP.frameAncestors.join(" ")}`,
		// Additional security directives
		"upgrade-insecure-requests",
	];

	return directives.join("; ");
}

/**
 * Generate Permissions Policy header value
 */
function generatePermissionsPolicy(): string {
	return Object.entries(SECURITY_CONFIG.PERMISSIONS_POLICY)
		.map(([directive, allowlist]) => {
			const sources = allowlist
				.map((source) => (source === "self" ? "self" : `"${source}"`))
				.join(" ");
			return `${directive}=(${sources})`;
		})
		.join(", ");
}

/**
 * SvelteKit server-side handle function
 * Applies security headers to all requests
 */
export const handle: Handle = async ({ event, resolve }) => {
	// Generate a nonce for this request (useful for CSP)
	const nonce = nanoid();

	// Store nonce in locals for use in templates
	event.locals.cspNonce = nonce;

	// Resolve the request
	const response = await resolve(event, {
		// Transform page chunks to add nonce to scripts in production
		transformPageChunk: ({ html, done }) => {
			if (done) {
				// Add nonce to inline scripts when present
				return html.replace(
					/<script(?![^>]*nonce=)/g,
					`<script nonce="${nonce}"`,
				);
			}
			return html;
		},
	});

	// Apply comprehensive security headers
	const headers = new Headers(response.headers);

	// Content Security Policy
	headers.set("Content-Security-Policy", generateCSP(nonce));

	// Anti-clickjacking protection
	headers.set("X-Frame-Options", "DENY");

	// HTTPS enforcement (only in production)
	if (!dev) {
		headers.set(
			"Strict-Transport-Security",
			`max-age=${SECURITY_CONFIG.HSTS.maxAge}; includeSubDomains; preload`,
		);
	}

	// Prevent MIME type sniffing
	headers.set("X-Content-Type-Options", "nosniff");

	// XSS protection (legacy browsers)
	headers.set("X-XSS-Protection", "1; mode=block");

	// Referrer policy
	headers.set("Referrer-Policy", SECURITY_CONFIG.REFERRER_POLICY);

	// Permissions policy
	headers.set("Permissions-Policy", generatePermissionsPolicy());

	// Remove server information
	headers.delete("Server");
	headers.delete("X-Powered-By");

	// Cache control for security-sensitive pages
	if (
		event.url.pathname.includes("/auth") ||
		event.url.pathname.includes("/profile") ||
		event.url.pathname.includes("/settings")
	) {
		headers.set(
			"Cache-Control",
			"no-store, no-cache, must-revalidate, proxy-revalidate",
		);
		headers.set("Pragma", "no-cache");
		headers.set("Expires", "0");
	}

	// CORS headers for API endpoints
	if (event.url.pathname.startsWith("/api")) {
		// Only allow specific origins
		const allowedOrigins = [
			"https://crescendai-backend.jai-d.workers.dev",
			"https://www.crescend.ai",
			...(dev ? ["http://localhost:5173", "http://127.0.0.1:5173"] : []),
		];

		const origin = event.request.headers.get("Origin");
		if (origin && allowedOrigins.includes(origin)) {
			headers.set("Access-Control-Allow-Origin", origin);
		}

		headers.set(
			"Access-Control-Allow-Methods",
			"GET, POST, PUT, DELETE, OPTIONS",
		);
		headers.set(
			"Access-Control-Allow-Headers",
			"Content-Type, Authorization, X-Requested-With",
		);
		headers.set("Access-Control-Max-Age", "86400"); // 24 hours
		headers.set("Access-Control-Allow-Credentials", "true");
	}

	return new Response(response.body, {
		status: response.status,
		statusText: response.statusText,
		headers,
	});
};

/**
 * Type definitions for SvelteKit locals
 */
declare global {
	namespace App {
		interface Locals {
			cspNonce?: string;
		}
	}
}
