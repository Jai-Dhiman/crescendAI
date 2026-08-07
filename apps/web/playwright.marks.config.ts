import { defineConfig } from "@playwright/test";

export default defineConfig({
	testMatch: ["tests/marks.spec.ts"],
	use: {
		headless: true,
		baseURL: "http://localhost:4173",
	},
	// Verovio WASM init plus a real score load is slow on a cold preview build.
	timeout: 120000,
	webServer: {
		command: "bun run build && bunx vite preview --port 4173 --strictPort",
		port: 4173,
		reuseExistingServer: !process.env.CI,
		timeout: 180000,
	},
});
