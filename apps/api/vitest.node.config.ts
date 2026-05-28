import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		environment: "node",
		include: [
			"scripts/**/*.test.ts",
			"src/harness/skills/__catalog__/**/*.test.ts",
			"src/harness/skills/validator.test.ts",
			"src/lib/**/*.test.ts",
			"src/harness/loop/**/*.test.ts",
			"src/services/**/*.test.ts",
			"src/do/session-brain.schema.test.ts",  // explicit entry; do NOT use src/do/**/*.test.ts glob
		],
		// workerd-only tests (real WASM, real bindings) must not run under node
		exclude: ["src/services/*.workerd.test.ts"],
		globalSetup: ["./scripts/test-setup-node.ts"],
	},
});
