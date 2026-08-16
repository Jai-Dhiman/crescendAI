/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

const { callIndex, VELOCITIES } = vi.hoisted(() => ({
	callIndex: { current: 0 },
	VELOCITIES: [62, 66, 63, 65, 64, 10, 10, 10],
}));

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => {
			const velocity = VELOCITIES[callIndex.current] ?? 64;
			callIndex.current++;
			return {
				notes: Array.from({ length: 10 }, (_, i) => ({
					pitch: 60,
					onset: i * 1.0,
					offset: i * 1.0 + 0.5,
					velocity,
				})),
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

describe("mark emission (#162)", () => {
	it("emits exactly one dynamics/needs_work mark by the 8th chunk and records it in emittedMarks", async () => {
		for (let i = 0; i < 8; i++) {
			await env.CHUNKS.put(`mk-${i}`, new Uint8Array([1]));
		}
		const id = env.SESSION_BRAIN.idFromName("mark-emission");
		const stub = env.SESSION_BRAIN.get(id);
		const sent: unknown[] = [];
		const fakeWs = {
			send(data: string) {
				sent.push(JSON.parse(data));
			},
			close() {},
		} as unknown as WebSocket;

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-mk", "stud-mk", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			for (let i = 0; i < 8; i++) {
				await instance.webSocketMessage(
					fakeWs,
					JSON.stringify({ type: "chunk_ready", index: i, r2Key: `mk-${i}` }),
				);
			}

			const after = (await state.storage.get("state")) as {
				emittedMarks: string[];
			};
			expect(after.emittedMarks).toEqual(["dynamics"]);

			await state.storage.deleteAlarm();
		});

		const marks = sent.filter((m) => (m as { type?: string }).type === "mark");
		expect(marks).toHaveLength(1);
		expect(marks[0]).toMatchObject({
			type: "mark",
			mark: { dimension: "dynamics", taxonomy: "needs_work" },
		});
		expect((marks[0] as { mark: { anchor: unknown } }).mark.anchor).toEqual({
			type: "timestamp",
			atSeconds: 105, // chunkIndex 7 * 15s
		});
	});
});
