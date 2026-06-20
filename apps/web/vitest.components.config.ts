/// <reference types="vitest" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
	plugins: [react(), tsconfigPaths({ projects: ["./tsconfig.json"] })],
	test: {
		environment: "jsdom",
		setupFiles: ["src/test-setup.ts"],
		include: ["src/components/ChatMessages.test.tsx"],
		testTimeout: 15000,
	},
});
