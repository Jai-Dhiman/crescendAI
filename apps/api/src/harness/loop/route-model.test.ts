import { describe, expect, it } from "vitest";
import { routeModel } from "./route-model";

describe("routeModel", () => {
	it("returns Sonnet teacher gateway for phase1_analysis", () => {
		const client = routeModel("phase1_analysis");
		expect(client.model).toBe("claude-sonnet-4-20250514");
		expect(client.gatewayUrlVar).toBe("AI_GATEWAY_TEACHER");
	});

	it("returns Sonnet teacher gateway for phase2_voice", () => {
		const client = routeModel("phase2_voice");
		expect(client.model).toBe("claude-sonnet-4-20250514");
		expect(client.gatewayUrlVar).toBe("AI_GATEWAY_TEACHER");
	});
});
