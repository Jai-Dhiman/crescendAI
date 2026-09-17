import { describe, expect, it, vi } from "vitest";

vi.mock("./api-client", () => ({ client: {} }));
vi.mock("./config", () => ({ API_BASE: "https://api.test" }));
const mockCaptureException = vi.fn();
const mockAddBreadcrumb = vi.fn();
vi.mock("./sentry", () => ({
	Sentry: {
		captureException: (...args: unknown[]) => mockCaptureException(...args),
		addBreadcrumb: (...args: unknown[]) => mockAddBreadcrumb(...args),
	},
}));

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
