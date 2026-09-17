// apps/api/src/services/inference.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Bindings } from "../lib/types";

describe("callAmtEndpoint base64 encoding", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("encodes a real-sized (15s) audio chunk without exceeding the call stack", async () => {
		// 15s @ 24kHz mono 16-bit ≈ 720KB. The old spread-based encoder
		// (String.fromCharCode(...bytes)) overflows the stack at this size.
		const buf = new ArrayBuffer(720_000);
		new Uint8Array(buf).fill(0x41);

		let capturedBody = "";
		vi.stubGlobal(
			"fetch",
			vi.fn(async (_url: string, init: RequestInit) => {
				capturedBody = init.body as string;
				return new Response(
					JSON.stringify({ midi_notes: [], pedal_events: [] }),
					{ status: 200, headers: { "Content-Type": "application/json" } },
				);
			}),
		);

		const { callAmtEndpoint } = await import("./inference");
		const env = { AMT_ENDPOINT: "http://amt.test" } as unknown as Bindings;

		// Must not throw "Maximum call stack size exceeded".
		const result = await callAmtEndpoint(env, buf, null);
		expect(result.notes).toEqual([]);

		// The encoded chunk must round-trip back to the original bytes.
		const payload = JSON.parse(capturedBody) as { chunk_audio: string };
		const decoded = atob(payload.chunk_audio);
		expect(decoded.length).toBe(720_000);
		expect(decoded.charCodeAt(0)).toBe(0x41);
		expect(decoded.charCodeAt(decoded.length - 1)).toBe(0x41);
	});
});
