/// <reference types="@cloudflare/vitest-pool-workers" />
import { describe, expect, it } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";
import { SessionBrain, nextSynthesisAlarmDelayMs } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

// Make the SESSION_BRAIN binding visible to TypeScript for this test.
declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
	}
}

// Constants mirrored from SessionBrain (private readonly fields).
const IDLE_WINDOW_MS = 30 * 60 * 1000;
const DEFERRED_WINDOW_MS = 90 * 1000;

describe("nextSynthesisAlarmDelayMs (Fix B / Finding 3)", () => {
	it("session ending with no chunks in flight -> synthesize immediately (1ms)", () => {
		expect(
			nextSynthesisAlarmDelayMs({
				sessionEnding: true,
				chunksInFlight: 0,
				deferredWindowMs: DEFERRED_WINDOW_MS,
				idleWindowMs: IDLE_WINDOW_MS,
			}),
		).toBe(1);
	});

	it("session ending with chunks still in flight -> deferred backstop window, NOT immediate", () => {
		const delay = nextSynthesisAlarmDelayMs({
			sessionEnding: true,
			chunksInFlight: 2,
			deferredWindowMs: DEFERRED_WINDOW_MS,
			idleWindowMs: IDLE_WINDOW_MS,
		});
		expect(delay).toBe(DEFERRED_WINDOW_MS);
		expect(delay).not.toBe(1); // the pre-fix bug: it fired synthesis on incomplete state
	});

	it("not ending -> extend the idle session window (existing activity behavior)", () => {
		expect(
			nextSynthesisAlarmDelayMs({
				sessionEnding: false,
				chunksInFlight: 0,
				deferredWindowMs: DEFERRED_WINDOW_MS,
				idleWindowMs: IDLE_WINDOW_MS,
			}),
		).toBe(IDLE_WINDOW_MS);
	});

	it("deferred backstop outlasts worst-case inference (dense-AMT ~53s, 2026-05-31 audit)", () => {
		// Sizing rationale: the backstop must not fire while a legitimately-slow chunk is still
		// transcribing, or it would re-introduce the very bug. 90s > 53s observed worst case.
		expect(DEFERRED_WINDOW_MS).toBeGreaterThan(53_000);
	});
});

// fake WebSocket: webSocketClose only calls ws.close(), wrapped in try/catch.
const fakeWs = { close() {} } as unknown as WebSocket;

describe("webSocketClose alarm guard (Fix B / Finding 3, DO-level)", () => {
	it("defers the synthesis alarm when chunks are still in flight", async () => {
		const id = env.SESSION_BRAIN.idFromName("ws-close-in-flight");
		const stub = env.SESSION_BRAIN.get(id);
		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-a", "stud-a", null);
			seeded.chunksInFlight = 2; // mid-inference disconnect
			await state.storage.put("state", seeded);

			const before = Date.now();
			await instance.webSocketClose(fakeWs, 1000, "client gone");

			const alarm = await state.storage.getAlarm();
			expect(alarm).not.toBeNull();
			// Deferred (~90s out), NOT the pre-fix unconditional +1ms.
			expect((alarm as number) - before).toBeGreaterThan(60_000);
			await state.storage.deleteAlarm();
		});
	});

	it("fires immediate synthesis (1ms) when no chunks are in flight", async () => {
		const id = env.SESSION_BRAIN.idFromName("ws-close-idle");
		const stub = env.SESSION_BRAIN.get(id);
		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-b", "stud-b", null);
			seeded.chunksInFlight = 0;
			await state.storage.put("state", seeded);

			const before = Date.now();
			await instance.webSocketClose(fakeWs, 1000, "client gone");

			const alarm = await state.storage.getAlarm();
			expect(alarm).not.toBeNull();
			expect((alarm as number) - before).toBeLessThan(60_000); // immediate
			await state.storage.deleteAlarm();
		});
	});

	it("does NOT bump state.version (leaves the in-flight chunk's merge undisturbed)", async () => {
		const id = env.SESSION_BRAIN.idFromName("ws-close-version");
		const stub = env.SESSION_BRAIN.get(id);
		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-c", "stud-c", null);
			seeded.chunksInFlight = 1;
			seeded.version = 7;
			await state.storage.put("state", seeded);

			await instance.webSocketClose(fakeWs, 1000, "client gone");

			const after = (await state.storage.get("state")) as { version: number; sessionEnding: boolean };
			expect(after.version).toBe(7); // unchanged: in-flight chunk merges its output on completion
			expect(after.sessionEnding).toBe(true);
			await state.storage.deleteAlarm();
		});
	});
});

// handleChunkReady's completion path uses the same nextSynthesisAlarmDelayMs decision
// (sessionEnding + drained chunksInFlight -> 1ms), exercised by the pure-function suite
// above; invoking it directly requires R2 + HF inference mocking and is out of scope here.
