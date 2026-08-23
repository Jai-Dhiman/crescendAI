/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async (_env, _chunk, _ctx) => {
			return {
				notes: [{ pitch: 60, onset: 0, offset: 0.4, velocity: 70 }],
				pedalEvents: [],
			};
		}),
	};
});

import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

const fakeWs = { send() {}, close() {} } as unknown as WebSocket;

describe("no-comment zone (#162): fewer than MIN_NOTES_FOR_COMMENT transcribed notes", () => {
	it("records a zone and contributes no sessionSamples for a 1-note chunk", async () => {
		await env.CHUNKS.put("ncz-0", new Uint8Array([1]));
		const id = env.SESSION_BRAIN.idFromName("no-comment-zone");
		const stub = env.SESSION_BRAIN.get(id);

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-ncz", "stud-ncz", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "chunk_ready", index: 0, r2Key: "ncz-0" }),
			);

			const after = (await state.storage.get("state")) as {
				noCommentZones: { chunkIndex: number; reason: string }[];
				sessionSamples: Record<string, number[]>;
			};
			expect(after.noCommentZones).toEqual([
				{ chunkIndex: 0, reason: "insufficient_notes" },
			]);
			expect(after.sessionSamples).toEqual({});

			await state.storage.deleteAlarm();
		});
	});

	it("records a zone for a chunk whose R2 object is missing", async () => {
		const id = env.SESSION_BRAIN.idFromName("no-comment-zone-r2-miss");
		const stub = env.SESSION_BRAIN.get(id);

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-ncz-r2", "stud-ncz-r2", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({
					type: "chunk_ready",
					index: 0,
					r2Key: "ncz-r2-missing-key",
				}),
			);

			const after = (await state.storage.get("state")) as {
				noCommentZones: { chunkIndex: number; reason: string }[];
			};
			expect(after.noCommentZones).toEqual([
				{ chunkIndex: 0, reason: "r2_not_found" },
			]);

			await state.storage.deleteAlarm();
		});
	});
});
