import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
	test: {
		// Vitest REPLACES defaultExclude rather than merging, so supplying this
		// array silently drops "**/node_modules/**". Without it the pool matches
		// 573 files instead of 133 -- the extra 477 are vendored suites, mostly
		// zod's own tests reached via @better-auth/core (#144).
		exclude: [
			"**/node_modules/**",
			"src/harness/skills/__catalog__/**",
			"src/harness/skills/validator.test.ts",
		],
		poolOptions: {
			workers: {
				wrangler: { configPath: "./wrangler.toml" },
				// With the default singleWorker: false, the pool eagerly starts one
				// workerd process PER MATCHED FILE before any test runs, each holding
				// several loopback ports. That exhausts the macOS ephemeral range
				// (49152-65535) and dies with EADDRNOTAVAIL. This pool runs in the
				// main Vitest process and ignores maxWorkers/fileParallelism, so this
				// is the only knob that bounds it. isolatedStorage stays true, so
				// per-file KV/DO/D1 isolation is unchanged -- only the runtime is
				// shared (#144).
				singleWorker: true,
			},
		},
	},
});
