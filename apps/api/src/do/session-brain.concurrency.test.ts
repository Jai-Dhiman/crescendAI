/// <reference types="@cloudflare/vitest-pool-workers" />
import { describe, expect, it, vi } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";

// Mock the HF inference endpoints so handleChunkReady runs without real network.
// Each call holds the chunk "in flight" for a tick so that N concurrently-dispatched
// chunk_ready messages overlap inside the inference await — exactly the window where the
// DO input gate is open (fetch() is non-storage I/O) and the read-modify-write race fires.
vi.mock("../services/inference", async (importOriginal) => {
	const actual =
		await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callMuqEndpoint: vi.fn(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return {
				scores: {
					dynamics: 0.5,
					timing: 0.5,
					pedaling: 0.5,
					articulation: 0.5,
					phrasing: 0.5,
					interpretation: 0.5,
				},
				confidences: null,
				chromaBytes: null,
				chromaFrames: 0,
				chromaFrameRateHz: 0,
			};
		}),
		callAmtEndpoint: vi.fn(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return { notes: [], pedalEvents: [] };
		}),
	};
});

import { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

// handleChunkReady calls ws.send (chunk_processed) and never closes; a no-op stub suffices.
const fakeWs = { send() {}, close() {} } as unknown as WebSocket;

describe("handleChunkReady concurrency (#11 chunksInFlight race, #12 version-bail drop)", () => {
	it("N concurrent chunk_ready -> all N scoredChunks persist, chunksInFlight drains to 0", async () => {
		const N = 5;

		// Seed the R2 objects each chunk_ready fetches.
		for (let i = 0; i < N; i++) {
			await env.CHUNKS.put(`conc-chunk-${i}`, new Uint8Array([1, 2, 3, 4]));
		}

		const id = env.SESSION_BRAIN.idFromName("concurrency-scoredchunks");
		const stub = env.SESSION_BRAIN.get(id);

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-conc", "stud-conc", null);
			// Trim side paths so the test isolates the scoredChunks/chunksInFlight race:
			seeded.baselinesLoaded = true; // skip the Hyperdrive baseline query
			seeded.baselines = null; // skip teaching-moment selection (gated on baselines)
			seeded.pieceLocked = true; // skip piece identification (gated on !pieceLocked)
			await state.storage.put("state", seeded);

			// Fire all N through the PUBLIC WebSocket message handler, concurrently.
			await Promise.all(
				Array.from({ length: N }, (_, i) =>
					instance.webSocketMessage(
						fakeWs,
						JSON.stringify({
							type: "chunk_ready",
							index: i,
							r2Key: `conc-chunk-${i}`,
						}),
					),
				),
			);

			const after = (await state.storage.get("state")) as {
				scoredChunks: { chunkIndex: number }[];
				chunksInFlight: number;
			};

			// Every concurrent chunk must contribute its score (no silent drop).
			expect(after.scoredChunks.length).toBe(N);
			expect(new Set(after.scoredChunks.map((c) => c.chunkIndex)).size).toBe(N);
			// In-flight counter fully drained so synthesis does not fire early.
			expect(after.chunksInFlight).toBe(0);

			await state.storage.deleteAlarm();
		});
	});
});
