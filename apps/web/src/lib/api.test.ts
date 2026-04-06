import { describe, expect, it, vi } from "vitest";

describe("api.scores.getData", () => {
	it("fetches from /api/scores/:pieceId/data and returns ArrayBuffer", async () => {
		const mockBuffer = new ArrayBuffer(8);
		const mockResponse = new Response(mockBuffer, {
			status: 200,
			headers: { "Content-Type": "application/vnd.recordare.musicxml" },
		});

		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(mockResponse));

		const { api } = await import("./api");
		const result = await api.scores.getData("piece-abc-123");

		expect(fetch).toHaveBeenCalledWith(
			expect.stringContaining("/api/scores/piece-abc-123/data"),
			expect.objectContaining({ credentials: "include" }),
		);
		expect(result).toBeInstanceOf(ArrayBuffer);

		vi.unstubAllGlobals();
	});
});
