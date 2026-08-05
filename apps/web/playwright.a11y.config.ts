import { defineConfig } from "@playwright/test";

export default defineConfig({
	testMatch: ["tests/a11y.spec.ts"],
	use: {
		headless: true,
		baseURL: "http://localhost:4173",
	},
	timeout: 60000,
	webServer: {
		command: "bun run build && bunx vite preview --port 4173 --strictPort",
		port: 4173,
		reuseExistingServer: !process.env.CI,
		timeout: 120000,
	},
});
