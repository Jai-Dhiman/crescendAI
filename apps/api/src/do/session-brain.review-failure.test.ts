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

vi.mock("../db/client", () => ({
	createDb: () =>
		new Proxy(
			{},
			{
				get(_t, prop) {
					if (prop === "then")
						return (resolve: (v: unknown[]) => void) => resolve([]);
					if (prop === "query") {
						return new Proxy(
							{},
							{ get: () => ({ findFirst: async () => undefined }) },
						);
					}
					return () => new Proxy({}, {}); // reused below via a stable proxy in practice
				},
			},
		),
}));

vi.mock("../services/teacher", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/teacher")>();
	return {
		...actual,
		// biome-ignore lint/correctness/useYield: throws before it can yield, by design.
		synthesizeV6: async function* () {
			throw new Error("teacher call failed");
		},
	};
});

import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
	}
}

describe("review event on synthesis failure (#162)", () => {
	it("sends exactly one review event with verdict: null when synthesizeV6 throws", async () => {
		const id = env.SESSION_BRAIN.idFromName("review-failure");
		const stub = env.SESSION_BRAIN.get(id);
		const received: unknown[] = [];

		await runInDurableObject(stub, async (_instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-fail", "stud-fail", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			const pair = new WebSocketPair();
			const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
			state.acceptWebSocket(server);
			client.accept();
			client.addEventListener("message", (evt: MessageEvent) => {
				received.push(JSON.parse(evt.data as string));
			});

			await state.storage.setAlarm(Date.now() + 1);
		});

		await runDurableObjectAlarm(stub);

		await runInDurableObject(stub, async (_instance: SessionBrain, state) => {
			await new Promise((resolve) => setTimeout(resolve, 20));
			await state.storage.deleteAlarm();
		});

		const reviews = received.filter(
			(m) => (m as { type?: string }).type === "review",
		);
		expect(reviews).toEqual([
			{
				type: "review",
				verdict: null,
				carry_forward: null,
				no_comment_zones: [],
			},
		]);
	});
});
