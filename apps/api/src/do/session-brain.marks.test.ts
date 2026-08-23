/// <reference types="@cloudflare/vitest-pool-workers" />

// NOTE: never call runDurableObjectAlarm(stub) nested inside a
// runInDurableObject(stub, ...) callback. Nesting the two runner control calls
// on the same stub corrupts vitest-pool-workers' isolated-storage stack and
// crashes the whole file at teardown ("Isolated storage failed" inside
// updateStackedStorage) -- and the crash masks assertion results, so the test
// can silently rot. Run the alarm between runInDurableObject calls instead.

import {
	env,
	runDurableObjectAlarm,
	runInDurableObject,
} from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

// Deterministic per-call AMT stub: 8 chunks, 10 notes each (clears
// MIN_NOTES_FOR_COMMENT so nothing lands in a no-comment zone). Velocity is
// steady for chunks 0-4 (~62-66) then drops hard for chunks 5-7 (10), which
// drags the mock scorer's "dynamics" dimension far enough below its
// within-session median band to fire after 3 consecutive out-of-band chunks
// (DEFAULT_BASELINE_CONFIG.firePersistence = 3). Every other dimension holds
// identical IOI/duration/pedal shape across all 8 chunks, so its MAD stays 0
// and its band never opens — "dynamics" is the only dimension that can fire.
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

const { fakeDb, capturedInserts } = vi.hoisted(() => {
	const capturedInserts: Array<{ table: unknown; values: unknown }> = [];
	function makeFakeDb(): unknown {
		let lastInsertTable: unknown = null;
		const handler: ProxyHandler<object> = {
			get(_target, prop) {
				if (prop === "then") {
					return (resolve: (v: unknown[]) => void) => resolve([]);
				}
				if (prop === "insert") {
					return (table: unknown) => {
						lastInsertTable = table;
						return proxy;
					};
				}
				if (prop === "values") {
					return (values: unknown) => {
						capturedInserts.push({ table: lastInsertTable, values });
						return proxy;
					};
				}
				if (prop === "query") {
					return new Proxy(
						{},
						{ get: () => ({ findFirst: async () => undefined }) },
					);
				}
				return () => proxy;
			},
		};
		const proxy = new Proxy({}, handler);
		return proxy;
	}
	return { fakeDb: makeFakeDb(), capturedInserts };
});

vi.mock("../db/client", () => ({ createDb: () => fakeDb }));

vi.mock("../services/teacher", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/teacher")>();
	return {
		...actual,
		synthesizeV6: async function* () {
			yield {
				type: "artifact" as const,
				value: {
					session_id: "sess-e2e",
					synthesis_scope: "session" as const,
					strengths: [],
					focus_areas: [],
					prescribed_exercise: null,
					dominant_dimension: "dynamics" as const,
					recurring_pattern: null,
					next_session_focus:
						"Keep the dynamic range even across repeated passages.",
					diagnosis_refs: [],
					headline: "H".repeat(320),
					assigned_loops: [],
				},
			};
		},
	};
});

import { studentProfiles } from "../db/schema/students";
import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

describe("pipeline retarget: marks + review end-to-end (#162)", () => {
	it("drives a session from chunk_ready through end_session and observes a mark, one review, and a persisted baseline", async () => {
		for (let i = 0; i < 8; i++) {
			await env.CHUNKS.put(`e2e-${i}`, new Uint8Array([1]));
		}

		const received: unknown[] = [];
		const id = env.SESSION_BRAIN.idFromName("e2e-marks-review");
		const stub = env.SESSION_BRAIN.get(id);

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-e2e", "stud-e2e", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true; // no persisted baseline: cold-start path
			await state.storage.put("state", seeded);

			const pair = new WebSocketPair();
			const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
			state.acceptWebSocket(server);
			client.accept();
			client.addEventListener("message", (evt: MessageEvent) => {
				received.push(JSON.parse(evt.data as string));
			});

			for (let i = 0; i < 8; i++) {
				await instance.webSocketMessage(
					server,
					JSON.stringify({ type: "chunk_ready", index: i, r2Key: `e2e-${i}` }),
				);
			}

			await instance.webSocketMessage(
				server,
				JSON.stringify({ type: "end_session" }),
			);

			await new Promise((resolve) => setTimeout(resolve, 20));
		});

		await runDurableObjectAlarm(stub);

		await runInDurableObject(stub, async (_instance: SessionBrain, state) => {
			await new Promise((resolve) => setTimeout(resolve, 20));
			await state.storage.deleteAlarm();
		});

		const marks = received.filter(
			(m) => (m as { type?: string }).type === "mark",
		) as Array<{
			mark: { dimension: string; anchor: { type: string }; evidence: string };
		}>;
		expect(marks.length).toBeGreaterThanOrEqual(1);
		expect(marks[0]?.mark.dimension).toBe("dynamics");
		expect(marks[0]?.mark.anchor.type).toBe("timestamp");
		expect(marks[0]?.mark.evidence).toMatch(
			/local tempo|velocity range|pedal overlap|articulation ratio|velocity slope|tempo \d+ bpm/,
		);

		const reviews = received.filter(
			(m) => (m as { type?: string }).type === "review",
		);
		expect(reviews.length).toBe(1);
		expect(reviews[0]).toMatchObject({
			type: "review",
			verdict: "H".repeat(320),
			carry_forward: "Keep the dynamic range even across repeated passages.",
		});

		const baselineInsert = capturedInserts.find(
			(c) =>
				c.table === studentProfiles &&
				(c.values as Record<string, unknown>).baselineState !== undefined,
		);
		expect(baselineInsert).toBeDefined();
	});
});
