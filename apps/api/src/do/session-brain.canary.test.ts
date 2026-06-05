/// <reference types="@cloudflare/vitest-pool-workers" />
import { describe, expect, it, vi, type Mock } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";

// Mock the HF inference endpoints. Each call holds the chunk "in flight" for a tick so that
// concurrently-dispatched chunk_ready messages overlap inside the inference await — the window
// where the DO input gate is open (fetch() is non-storage I/O). This drives the real production
// chunk_ready path (handleChunkReady -> finalizeChunk) in the actual workerd runtime.
//
// NOTE: all scenarios run inside ONE `it`. vitest-pool-workers snapshots isolated storage at
// every test boundary, and asserts the DO storage dir contains only `.sqlite` files; a DO that
// leaves a SQLite WAL `-shm` sidecar between tests trips that assert. Creating one DO per `it`
// across many tests hits it. A single test boundary sidesteps the quirk while still exercising
// each scenario against its own fresh DO instance.
vi.mock("../services/inference", async (importOriginal) => {
	const actual =
		await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callMuqEndpoint: vi.fn(),
		callAmtEndpoint: vi.fn(),
	};
});

import { callAmtEndpoint, callMuqEndpoint } from "../services/inference";
import { SessionBrain } from "./session-brain";
import { createInitialState, type SessionState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

const muqMock = callMuqEndpoint as unknown as Mock;
const amtMock = callAmtEndpoint as unknown as Mock;

const tick = () => new Promise((r) => setTimeout(r, 10));

const okMuq = async () => {
	await tick();
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
};
const okAmt = async () => {
	await tick();
	return { notes: [], pedalEvents: [] };
};

// no-op WebSocket stub: handleChunkReady only calls ws.send / ws.close.
const fakeWs = { send() {}, close() {} } as unknown as WebSocket;

const chunkReady = (i: number, r2Key: string) =>
	JSON.stringify({ type: "chunk_ready", index: i, r2Key });

/** Seed a session whose side paths (DB baselines, piece id) are short-circuited so each
 *  scenario isolates the chunk_ready concurrency/lifecycle behavior under test. */
async function seed(storage: DurableObjectStorage): Promise<void> {
	const s = createInitialState("sess", "stud", null);
	s.baselinesLoaded = true; // skip Hyperdrive baseline query
	s.baselines = null; // skip teaching-moment selection
	s.pieceLocked = true; // skip piece identification
	// Neutralize synthesis: lifecycle scenarios drain to a +1ms alarm that would otherwise fire
	// runSynthesisAndPersist (LLM + DB). This canary verifies DO chunk accounting, not synthesis,
	// so a fired alarm no-ops (alarm() returns early when both are already set).
	s.synthesisCompleted = true;
	s.finalized = true;
	await storage.put("state", s);
}

const readState = (storage: DurableObjectStorage) =>
	storage.get("state") as Promise<SessionState>;

const fireConcurrent = (inst: SessionBrain, prefix: string, n: number) =>
	Array.from({ length: n }, (_, i) =>
		inst.webSocketMessage(fakeWs, chunkReady(i, `${prefix}-${i}`)),
	);

describe("SessionBrain chunk_ready canary — DO robustness across situations", () => {
	it("holds up across concurrency, lifecycle, and failure edge cases", async () => {
		// All scenarios share ONE DO instance (re-seeded between scenarios). vitest-pool-workers
		// asserts the DO storage dir holds only `.sqlite` files at the test boundary; spinning up
		// many DO instances leaves a SQLite WAL `-shm` sidecar that trips it. One instance = one DB.
		const stub = env.SESSION_BRAIN.get(env.SESSION_BRAIN.idFromName("canary"));
		await runInDurableObject(stub, async (inst: SessionBrain, state) => {
			const storage = state.storage;
			// Run `body` against the shared DO after resetting its session state.
			const scenario = async (fn: () => Promise<void>) => {
				await seed(storage);
				await fn();
				await storage.deleteAlarm();
			};
		// ── Scenario 1: high concurrency (12 chunks) — every chunk persists, counter drains ──
		muqMock.mockImplementation(okMuq);
		amtMock.mockImplementation(okAmt);
		{
			const N = 12;
			for (let i = 0; i < N; i++)
				await env.CHUNKS.put(`hc-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				await Promise.all(fireConcurrent(inst, "hc", N));
				const st = await readState(storage);
				expect(st.scoredChunks.length, "12 concurrent → 12 scored").toBe(N);
				expect(
					new Set(st.scoredChunks.map((c) => c.chunkIndex)).size,
					"all indices distinct",
				).toBe(N);
				expect(st.chunksInFlight, "counter drained").toBe(0);
			});
		}

		// ── Scenario 2: end_session arrives mid-burst — nothing dropped, synthesis not claimed early ──
		{
			const N = 5;
			for (let i = 0; i < N; i++)
				await env.CHUNKS.put(`es-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				await Promise.all([
					...fireConcurrent(inst, "es", N),
					inst.webSocketMessage(
						fakeWs,
						JSON.stringify({ type: "end_session" }),
					),
				]);
				const st = await readState(storage);
				expect(st.scoredChunks.length, "no chunk lost to end_session race").toBe(
					N,
				);
				expect(st.chunksInFlight).toBe(0);
				expect(st.sessionEnding, "end_session recorded").toBe(true);
			});
		}

		// ── Scenario 3: webSocketClose mid-burst — chunks survive the disconnect ──
		{
			const N = 5;
			for (let i = 0; i < N; i++)
				await env.CHUNKS.put(`wc-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				await Promise.all([
					...fireConcurrent(inst, "wc", N),
					inst.webSocketClose(fakeWs, 1000, "client gone"),
				]);
				const st = await readState(storage);
				expect(st.scoredChunks.length, "chunks survive disconnect").toBe(N);
				expect(st.chunksInFlight).toBe(0);
				expect(st.sessionEnding).toBe(true);
			});
		}

		// ── Scenario 4: all-MuQ-failure concurrent — counter drains, no partial scoredChunks ──
		muqMock.mockImplementation(async () => {
			await tick();
			throw new Error("MuQ down");
		});
		{
			const N = 5;
			for (let i = 0; i < N; i++)
				await env.CHUNKS.put(`mf-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				await Promise.all(fireConcurrent(inst, "mf", N));
				const st = await readState(storage);
				expect(st.scoredChunks.length, "failed inference scores nothing").toBe(
					0,
				);
				expect(st.chunksInFlight, "failures still decrement").toBe(0);
				expect(st.inferenceFailures, "every failure counted").toBe(N);
			});
		}
		muqMock.mockImplementation(okMuq); // restore for remaining scenarios

		// ── Scenario 5: missing-R2 chunks interleaved with good ones — good ones persist ──
		{
			const N = 6; // even indices present, odd indices missing
			for (let i = 0; i < N; i += 2)
				await env.CHUNKS.put(`mr-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				await Promise.all(fireConcurrent(inst, "mr", N));
				const st = await readState(storage);
				expect(st.scoredChunks.length, "only present-R2 chunks score").toBe(3);
				expect(new Set(st.scoredChunks.map((c) => c.chunkIndex))).toEqual(
					new Set([0, 2, 4]),
				);
				expect(st.chunksInFlight, "missing-R2 decrements did not leak").toBe(0);
			});
		}

		// ── Scenario 6: set_piece interleaved (bumps version) — chunks NOT dropped (removed-bail) ──
		{
			const N = 5;
			for (let i = 0; i < N; i++)
				await env.CHUNKS.put(`sp-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				await Promise.all([
					...fireConcurrent(inst, "sp", N),
					inst.webSocketMessage(
						fakeWs,
						JSON.stringify({ type: "set_piece", query: "chopin nocturne" }),
					),
				]);
				const st = await readState(storage);
				expect(
					st.scoredChunks.length,
					"version bump mid-flight no longer drops chunks",
				).toBe(N);
				expect(st.chunksInFlight).toBe(0);
			});
		}

		// ── Scenario 7: sequential chunks (no-regression) — monotonic accumulation ──
		{
			const N = 4;
			for (let i = 0; i < N; i++)
				await env.CHUNKS.put(`sq-${i}`, new Uint8Array([i + 1]));
			await scenario(async () => {
				for (let i = 0; i < N; i++) {
					await inst.webSocketMessage(fakeWs, chunkReady(i, `sq-${i}`));
					const mid = await readState(storage);
					expect(mid.scoredChunks.length, "monotonic accumulation").toBe(i + 1);
					expect(mid.chunksInFlight, "each drains before next").toBe(0);
				}
			});
		}
		}); // close runInDurableObject callback
	});
});
