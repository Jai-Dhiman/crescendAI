import { describe, expect, it } from "vitest";

describe("getPieceData R2 key", () => {
	it("requests the .mxl key from R2", async () => {
		let requestedKey = "";
		const mockEnv = {
			SCORES: {
				get: async (key: string) => {
					requestedKey = key;
					return { body: new ReadableStream() };
				},
			},
		};

		const { getPieceData } = await import("./scores");
		await getPieceData(mockEnv as never, "abc-123");
		expect(requestedKey).toBe("scores/v1/abc-123.mxl");
	});
});
