import { cloudflare } from "@cloudflare/vite-plugin";
import { sentryVitePlugin } from "@sentry/vite-plugin";
import tailwindcss from "@tailwindcss/vite";

import { tanstackStart } from "@tanstack/react-start/plugin/vite";

import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

const config = defineConfig({
	build: { sourcemap: true },
	plugins: [
		tailwindcss(),
		tsconfigPaths({ projects: ["./tsconfig.json"] }),
		cloudflare({ viteEnvironment: { name: "ssr" } }),
		tanstackStart(),
		viteReact(),
		sentryVitePlugin({
			org: "crescendai",
			project: "crescendai-web",
			sourcemaps: { assets: "./dist/**" },
		}),
	],
});

export default config;
