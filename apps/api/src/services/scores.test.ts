import { describe, expect, it } from "vitest";

function mockEnv(present: Set<string>, seen: string[]) {
	return {
		SCORES: {
			get: async (key: string) => {
				seen.push(key);
				return present.has(key) ? { body: new ReadableStream() } : null;
			},
		},
	} as never;
}

describe("getPieceData format resolution", () => {
	it("prefers the PD-clean .mei and returns its content type", async () => {
		const seen: string[] = [];
		const env = mockEnv(new Set(["scores/v1/abc-123.mei"]), seen);
		const { getPieceData } = await import("./scores");
		const { contentType } = await getPieceData(env, "abc-123");
		expect(seen[0]).toBe("scores/v1/abc-123.mei");
		expect(contentType).toBe("application/mei+xml");
	});

	it("falls back to the legacy .mxl when no .mei exists", async () => {
		const seen: string[] = [];
		const env = mockEnv(new Set(["scores/v1/abc-123.mxl"]), seen);
		const { getPieceData } = await import("./scores");
		const { contentType } = await getPieceData(env, "abc-123");
		expect(seen).toEqual(["scores/v1/abc-123.mei", "scores/v1/abc-123.mxl"]);
		expect(contentType).toBe("application/vnd.recordare.musicxml+zip");
	});

	it("throws NotFoundError when neither format exists", async () => {
		const env = mockEnv(new Set(), []);
		const { getPieceData } = await import("./scores");
		await expect(getPieceData(env, "abc-123")).rejects.toThrow();
	});
});
