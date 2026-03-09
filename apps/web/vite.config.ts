import { cloudflare } from "@cloudflare/vite-plugin";
import tailwindcss from "@tailwindcss/vite";

import { tanstackStart } from "@tanstack/react-start/plugin/vite";

import viteReact from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

const config = defineConfig({
	plugins: [
		tailwindcss(),
		tsconfigPaths({ projects: ["./tsconfig.json"] }),
		cloudflare({ viteEnvironment: { name: "ssr" } }),
		tanstackStart(),
		viteReact(),
	],
});

export default config;
