import { defineConfig } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

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
