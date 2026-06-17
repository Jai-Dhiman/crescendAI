// apps/api/src/harness/loop/route-model.test.ts
import { describe, expect, it } from "vitest";
import type { Bindings } from "../../lib/types";
import { routeModel } from "./route-model";

const WORKERS_AI_ENV = {
	TEACHER_PROVIDER: "workers-ai",
} as unknown as Bindings;

const ANTHROPIC_ENV = {
	TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;

const NO_PROVIDER_ENV = {} as unknown as Bindings;

describe("routeModel — workers-ai provider (default)", () => {
	it("returns workers-ai provider and Qwen model when TEACHER_PROVIDER=workers-ai", () => {
		const client = routeModel("phase1_analysis", WORKERS_AI_ENV);
		expect(client.provider).toBe("workers-ai");
		expect(client.model).toBe("@cf/qwen/qwen3-30b-a3b-fp8");
	});

	it("defaults to workers-ai when TEACHER_PROVIDER is not set", () => {
		const client = routeModel("phase2_voice", NO_PROVIDER_ENV);
		expect(client.provider).toBe("workers-ai");
		expect(client.model).toBe("@cf/qwen/qwen3-30b-a3b-fp8");
	});

	it("uses TEACHER_MODEL override for the workers-ai model when set", () => {
		const env = {
			TEACHER_PROVIDER: "workers-ai",
			TEACHER_MODEL: "@cf/zai-org/glm-4.7-flash",
		} as unknown as Bindings;
		const client = routeModel("phase1_analysis", env);
		expect(client.provider).toBe("workers-ai");
		expect(client.model).toBe("@cf/zai-org/glm-4.7-flash");
	});
});

describe("routeModel — anthropic provider (toggle)", () => {
	it("returns anthropic provider and Sonnet model when TEACHER_PROVIDER=anthropic", () => {
		const client = routeModel("phase1_analysis", ANTHROPIC_ENV);
		expect(client.provider).toBe("anthropic");
		expect(client.model).toBe("claude-sonnet-4-20250514");
	});

	it("returns anthropic provider for phase2_voice when TEACHER_PROVIDER=anthropic", () => {
		const client = routeModel("phase2_voice", ANTHROPIC_ENV);
		expect(client.provider).toBe("anthropic");
		expect(client.model).toBe("claude-sonnet-4-20250514");
	});
});
