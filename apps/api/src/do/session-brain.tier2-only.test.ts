/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => ({
			notes: Array.from({ length: 10 }, (_, i) => ({
				pitch: 60,
				onset: i * 0.3,
				offset: i * 0.3 + 0.2,
				velocity: 70,
			})),
			pedalEvents: [],
		})),
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

describe("tier-1 removal (#162): every chunk takes Tier-2, no chunk_bar_map", () => {
	it("never sends a chunk_bar_map message even with a piece context set", async () => {
		await env.CHUNKS.put("t2-0", new Uint8Array([1]));
		const id = env.SESSION_BRAIN.idFromName("tier2-only");
		const stub = env.SESSION_BRAIN.get(id);
		const sent: unknown[] = [];
		const fakeWs = {
			send(data: string) {
				sent.push(JSON.parse(data));
			},
			close() {},
		} as unknown as WebSocket;

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-t2", "stud-t2", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "chunk_ready", index: 0, r2Key: "t2-0" }),
			);

			await state.storage.deleteAlarm();
		});

		expect(
			sent.some((m) => (m as { type?: string }).type === "chunk_bar_map"),
		).toBe(false);
	});
});
