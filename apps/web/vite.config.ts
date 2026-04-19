import { cloudflare } from "@cloudflare/vite-plugin";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import tailwindcss from "@tailwindcss/vite";

import { tanstackStart } from "@tanstack/react-start/plugin/vite";

import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

const isTest = process.env.VITEST === "true";

const config = defineConfig({
	build: { sourcemap: true },
	test: {
		environment: "jsdom",
		setupFiles: ["src/test-setup.ts"],
		include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
	},
	plugins: [
		tailwindcss(),
		tsconfigPaths({ projects: ["./tsconfig.json"] }),
		!isTest && cloudflare({ viteEnvironment: { name: "ssr" } }),
		!isTest && tanstackStart(),
		viteReact(),
		!isTest &&
			sentryVitePlugin({
				org: "crescendai",
				project: "crescendai-web",
				sourcemaps: { assets: "./dist/**" },
			}),
	],
});

export default config;
