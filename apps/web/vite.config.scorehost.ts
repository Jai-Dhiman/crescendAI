import { defineConfig } from "vite";
import viteReact from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const scorehostRoot = path.resolve(__dirname, "src/scorehost");

export default defineConfig({
  root: scorehostRoot,
  publicDir: path.resolve(__dirname, "public"),
  build: {
    outDir: path.resolve(__dirname, "dist-scorehost"),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(scorehostRoot, "index.html"),
    },
  },
  worker: { format: "es" },
  optimizeDeps: {
    exclude: ["verovio/wasm", "verovio/esm"],
  },
  plugins: [
    tsconfigPaths({ projects: [path.resolve(__dirname, "tsconfig.json")] }),
    viteReact(),
  ],
});
