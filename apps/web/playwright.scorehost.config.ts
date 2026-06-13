import { defineConfig } from "@playwright/test";

export default defineConfig({
  testMatch: ["src/scorehost/*.playwright.ts"],
  use: {
    headless: true,
  },
  timeout: 30000,
});
