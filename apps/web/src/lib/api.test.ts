import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

describe("api.exercises.assignPending", () => {
	const payload = {
		sourcePassage: "bars 5-8",
		targetSkill: "Pedaling clarity",
		exercises: [
			{
				title: "Legato run",
				instruction: "Half tempo.",
				focusDimension: "pedaling",
				exerciseId: "ex-999",
			},
		],
	};

	let fetchSpy: ReturnType<typeof vi.spyOn>;

	beforeEach(() => {
		fetchSpy = vi.spyOn(globalThis, "fetch");
		vi.clearAllMocks();
	});

	afterEach(() => {
		fetchSpy.mockRestore();
	});

	it("POSTs to /api/exercises/assign-pending and returns parsed ExerciseSetConfig on 200", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(payload), {
				status: 200,
				headers: { "Content-Type": "application/json" },
			}),
		);
		const { api } = await import("./api");
		const result = await api.exercises.assignPending({
			sessionId: "sess-1",
			exerciseId: "ex-123",
		});
		expect(fetchSpy).toHaveBeenCalledWith(
			"https://api.test/api/exercises/assign-pending",
			expect.objectContaining({
				method: "POST",
				credentials: "include",
				body: JSON.stringify({ sessionId: "sess-1", exerciseId: "ex-123" }),
			}),
		);
		expect(result).toEqual(payload);
	});

	it("throws ApiError and captures to Sentry on non-ok response", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify({ error: "not found" }), {
				status: 404,
				headers: { "Content-Type": "application/json" },
			}),
		);
		const { api, ApiError } = await import("./api");
		await expect(
			api.exercises.assignPending({ sessionId: "sess-1", exerciseId: "bad" }),
		).rejects.toBeInstanceOf(ApiError);
		expect(mockCaptureException).toHaveBeenCalledWith(
			expect.any(ApiError),
			expect.objectContaining({
				extra: expect.objectContaining({
					path: "/api/exercises/assign-pending",
				}),
			}),
		);
	});
});
