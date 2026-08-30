import { afterEach, describe, expect, it, vi } from "vitest";
import { practiceApi } from "./practice-api";

afterEach(() => {
	vi.unstubAllGlobals();
});

function stubFetch() {
	const spy = vi.fn().mockResolvedValue(
		new Response(JSON.stringify({ sessionId: "s1", conversationId: "c1" }), {
			status: 200,
			headers: { "Content-Type": "application/json" },
		}),
	);
	vi.stubGlobal("fetch", spy);
	return spy;
}

describe("practiceApi.start", () => {
	it("includes pieceId in the POST body when a piece was picked", async () => {
		const spy = stubFetch();

		await practiceApi.start(undefined, "chopin-nocturne-op9-no2");

		const [, init] = spy.mock.calls[0];
		const body = JSON.parse((init as RequestInit).body as string);
		expect(body.pieceId).toBe("chopin-nocturne-op9-no2");
	});

	it("omits pieceId from the POST body when no piece was picked", async () => {
		const spy = stubFetch();

		await practiceApi.start();

		const [, init] = spy.mock.calls[0];
		expect((init as RequestInit).body).toBeUndefined();
	});

	it("sends both conversationId and pieceId when resuming a piece-carrying session", async () => {
		const spy = stubFetch();

		await practiceApi.start("conv-1", "chopin-nocturne-op9-no2");

		const [, init] = spy.mock.calls[0];
		const body = JSON.parse((init as RequestInit).body as string);
		expect(body).toEqual({
			conversationId: "conv-1",
			pieceId: "chopin-nocturne-op9-no2",
		});
	});
});
