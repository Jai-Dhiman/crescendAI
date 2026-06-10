/// <reference types="@cloudflare/vitest-pool-workers" />
import { describe, expect, it } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";
import { SessionBrain } from "./session-brain";
import { createInitialState, type SessionState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		SCORES: R2Bucket;
	}
}

// A minimal v2 artifact: "exact" shares the query's chord-events + chroma; "decoy" is disjoint.
// This is the certified locking fixture proven in wasm-bridge.workerd.test.ts.
const V2_ARTIFACT = JSON.stringify({
	version: "v2",
	onset_tol_ms: 50,
	pieces: [
		{
			piece_id: "decoy",
			composer: "X",
			title: "Decoy",
			chroma: new Array(12).fill(0),
			events: [16, 32, 64, 128],
		},
		{
			piece_id: "exact",
			composer: "Y",
			title: "Exact",
			chroma: (() => {
				const a = new Array(12).fill(0);
				a[0] = 0.5;
				a[4] = 0.5;
				a[7] = 0.5;
				return a;
			})(),
			events: [1, 16, 128, 1],
		},
	],
});

// Two identical candidates => margin 0 < threshold => never locks (OOD/ambiguous proxy).
// Used by the buffer-cap test (never locks, so the buffer keeps growing) and by the
// negative-case tests added in Task 3.
const V2_AMBIGUOUS = JSON.stringify({
	version: "v2",
	onset_tol_ms: 50,
	pieces: [
		{
			piece_id: "a",
			composer: "X",
			title: "A",
			chroma: (() => {
				const c = new Array(12).fill(0);
				c[0] = 0.5;
				c[4] = 0.5;
				c[7] = 0.5;
				return c;
			})(),
			events: [1, 16, 128, 1],
		},
		{
			piece_id: "b",
			composer: "X",
			title: "B",
			chroma: (() => {
				const c = new Array(12).fill(0);
				c[0] = 0.5;
				c[4] = 0.5;
				c[7] = 0.5;
				return c;
			})(),
			events: [1, 16, 128, 1],
		},
	],
});

// C-E-G-C across 4 onsets -> 4 chord-events {0},{4},{7},{0}; matches "exact".
const MATCH_NOTES = [
	{ pitch: 60, onset: 0.0, offset: 0.4, velocity: 100 },
	{ pitch: 64, onset: 0.5, offset: 0.9, velocity: 100 },
	{ pitch: 67, onset: 1.0, offset: 1.4, velocity: 100 },
	{ pitch: 72, onset: 1.5, offset: 1.9, velocity: 100 },
];

const PREDICTIONS = {
	dynamics: 0.5,
	timing: 0.5,
	pedaling: 0.5,
	articulation: 0.5,
	phrasing: 0.5,
	interpretation: 0.5,
};

// Capture every WS frame the DO sends.
function recordingWs(): { ws: WebSocket; sent: unknown[] } {
	const sent: unknown[] = [];
	const ws = {
		send(data: string) {
			sent.push(JSON.parse(data));
		},
		close() {},
	} as unknown as WebSocket;
	return { ws, sent };
}

const evalChunk = (i: number, notes: typeof MATCH_NOTES) =>
	JSON.stringify({
		type: "eval_chunk",
		chunk_index: i,
		predictions: PREDICTIONS,
		midi_notes: notes,
		pedal_events: [],
	});

const readState = (storage: DurableObjectStorage) =>
	storage.get("state") as Promise<SessionState>;

describe("SessionBrain piece-ID v2 gate (eval_chunk path)", () => {
	it("locks to the in-catalog piece and emits piece_identified once the buffer crosses the threshold", async () => {
		await env.SCORES.put("fingerprint/v2/piece_index.json", V2_ARTIFACT);
		const stub = env.SESSION_BRAIN.get(
			env.SESSION_BRAIN.idFromName("pid-lock"),
		);
		await runInDurableObject(stub, async (inst: SessionBrain, state) => {
			const seeded = createInitialState("sess", "stud", null);
			seeded.baselinesLoaded = true; // skip Hyperdrive baseline query
			seeded.baselines = null;
			await state.storage.put("state", seeded);

			const { ws, sent } = recordingWs();
			// 11 four-note chunks => 44 notes accumulated > MIN_NOTES_FOR_IDENTIFICATION (30).
			for (let i = 0; i < 11; i++) {
				await inst.webSocketMessage(ws, evalChunk(i, MATCH_NOTES));
			}

			const st = await readState(state.storage);
			expect(st.pieceLocked).toBe(true);
			expect(st.pieceIdentification?.pieceId).toBe("exact");
			expect(st.pieceIdentification?.method).toBe("identify_v2");

			const identified = sent.find(
				(m): m is { type: string; pieceId: string } =>
					typeof m === "object" &&
					m !== null &&
					(m as { type?: string }).type === "piece_identified",
			);
			expect(identified?.pieceId).toBe("exact");
		});
	});

	it("caps the identification buffer at the max, keeping the most recent notes", async () => {
		// Use the ambiguous artifact so the buffer never locks and keeps growing,
		// letting us observe the truncation behavior.
		await env.SCORES.put("fingerprint/v2/piece_index.json", V2_AMBIGUOUS);
		const stub = env.SESSION_BRAIN.get(
			env.SESSION_BRAIN.idFromName("pid-buffer-cap"),
		);
		await runInDurableObject(stub, async (inst: SessionBrain, state) => {
			const seeded = createInitialState("sess", "stud", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			await state.storage.put("state", seeded);

			const { ws } = recordingWs();
			// MAX_IDENTIFICATION_BUFFER = 1200. Push 1300 distinct notes across
			// chunks of 100, so the buffer must truncate to the most recent 1200.
			let pitch = 0;
			for (let chunk = 0; chunk < 13; chunk++) {
				const notes = Array.from({ length: 100 }, () => {
					const t = pitch * 0.01;
					const n = {
						pitch: 21 + (pitch % 88),
						onset: t,
						offset: t + 0.005,
						velocity: 100,
					};
					pitch++;
					return n;
				});
				await inst.webSocketMessage(ws, evalChunk(chunk, notes));
			}

			const st = await readState(state.storage);
			expect(st.pieceLocked).toBe(false);
			expect(st.identificationNoteBuffer).toHaveLength(1200);
			// The most recent note pushed was index 1299 (onset 12.99); the oldest
			// retained must be index 100 (onset 1.0) after dropping the first 100.
			expect(st.identificationNoteBuffer[0]?.onset).toBeCloseTo(1.0, 5);
			expect(
				st.identificationNoteBuffer[
					st.identificationNoteBuffer.length - 1
				]?.onset,
			).toBeCloseTo(12.99, 5);
		});
	});
});
