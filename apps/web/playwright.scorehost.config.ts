import { defineConfig } from "@playwright/test";

export default defineConfig({
  testMatch: ["src/scorehost/*.playwright.ts"],
  use: {
    headless: true,
    baseURL: "http://localhost:5173",
  },
  timeout: 60000,
  webServer: {
    command: "bunx vite preview --config vite.config.scorehost.ts --port 5173 --strictPort",
    port: 5173,
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});
