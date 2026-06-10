// apps/api/src/services/wasm-bridge.workerd.test.ts
//
// Real-WASM workerd integration test. Runs under @cloudflare/vitest-pool-workers
// (vitest.config.ts), which uses actual workerd — the same runtime as production.
//
// PURPOSE: Catch the class of bug where the WASM module boots without error
// but exports are not callable (e.g. "malloc is not a function",
// "identify_piece is not a function"). vi.mock() in the node-env tests cannot
// catch this because it replaces the WASM with a mock before instantiation.
//
// These tests import the real WASM via the bridge and assert that functions
// return real computed values, proving the WASM is fully initialized.

import { describe, expect, it } from "vitest";
import {
	type PerfNote,
	type ScoreBar,
	alignChunkChroma,
	analyzeTier2,
	selectTeachingMoment,
} from "./wasm-bridge";

describe("wasm-bridge workerd (real WASM)", () => {
	// -------------------------------------------------------------------------
	// Score-analysis: selectTeachingMoment
	// -------------------------------------------------------------------------
	// Verifies that the score-analysis WASM module is fully instantiated and
	// that select_teaching_moment is callable (proves wasm.malloc works).
	//
	// Input: 2 scored chunks where chunk 0 is below baseline in dynamics.
	// Expected: returns a TeachingMoment selecting the negative deviation.

	it("selectTeachingMoment executes and returns a TeachingMoment", () => {
		const chunks = [
			{ chunk_index: 0, scores: [0.3, 0.7, 0.6, 0.6, 0.6, 0.6] as [number, number, number, number, number, number] },
			{ chunk_index: 1, scores: [0.8, 0.7, 0.6, 0.6, 0.6, 0.6] as [number, number, number, number, number, number] },
		];
		const baselines = {
			dynamics: 0.7,
			timing: 0.7,
			pedaling: 0.7,
			articulation: 0.7,
			phrasing: 0.7,
			interpretation: 0.7,
		};
		const result = selectTeachingMoment(chunks, baselines, []);

		// Result must be a non-null object with real computed values.
		// Chunk 0 has dynamics=0.3 vs baseline=0.7, so it should be selected.
		expect(result).not.toBeNull();
		expect(typeof result!.chunk_index).toBe("number");
		expect(typeof result!.dimension).toBe("string");
		expect(typeof result!.score).toBe("number");
		expect(typeof result!.baseline).toBe("number");
		// Deviation is negative (student below baseline on dynamics)
		expect(result!.deviation).toBeLessThan(0);
		expect(typeof result!.reasoning).toBe("string");
		expect(result!.reasoning.length).toBeGreaterThan(0);
	});

	// -------------------------------------------------------------------------
	// Score-analysis: alignChunkChroma (the chroma DTW path behind chunk_bar_map)
	// -------------------------------------------------------------------------
	// Mirrors the Rust fixture `valid_score_frame_zero_not_overwritten_by_gap_fill`:
	// a one-bar score with a single C4 note + two C-dominant audio chroma frames.
	// Proves the score-analysis WASM is fully instantiated AND that
	// align_chunk_chroma executes and returns a BarMapChroma with a populated
	// bar_per_frame — the exact payload the DO emits as `chunk_bar_map`.
	it("alignChunkChroma returns a BarMapChroma with a populated bar_per_frame", () => {
		// Audio chroma: 2 frames, both strongly pitch-class 0 (C). Row-major 12 x 2,
		// LE float32 bytes. index = pc * nFrames + frame.
		const nFrames = 2;
		const f32 = new Float32Array(12 * nFrames);
		f32[0 * nFrames + 0] = 1.0; // pc=0, frame 0
		f32[0 * nFrames + 1] = 1.0; // pc=0, frame 1
		const audioChromaBytes = new Uint8Array(f32.buffer);

		const bar: ScoreBar = {
			bar_number: 1,
			start_tick: 0,
			start_seconds: 0,
			time_signature: "4/4",
			notes: [
				{
					pitch: 60, // C4 -> pitch class 0
					pitch_name: "C4",
					velocity: 80,
					onset_tick: 0,
					onset_seconds: 0,
					duration_ticks: 480,
					duration_seconds: 0.2,
					track: 0,
				},
			],
			pedal_events: [],
			note_count: 1,
			pitch_range: [60],
			mean_velocity: 80,
		};

		// frame_rate=10 Hz, decim=10 Hz (1:1 so bar_per_frame has 2 entries).
		const result = alignChunkChroma(audioChromaBytes, nFrames, [bar], 10.0, 10.0);

		expect(Array.isArray(result.bar_per_frame)).toBe(true);
		expect(result.bar_per_frame.length).toBeGreaterThan(0);
		// Only one bar exists; every decimated frame must map to bar 1.
		expect(result.bar_per_frame.every((b) => b === 1)).toBe(true);
		expect(result.bar_min).toBe(1);
		expect(result.bar_max).toBe(1);
	});

	// -------------------------------------------------------------------------
	// Score-analysis: analyzeTier2 bar_range marshaling
	// -------------------------------------------------------------------------
	// Regression for the workerd serde_wasm_bindgen mismarshal: Tier 2 sets
	// bar_range = None, which must reach JS as null. Before the JSON-string fix,
	// to_value() aliased the field to the input perf_notes array, so consumers
	// hit "barStr.split is not a function" in the pre-lock bar-analysis path.
	it("analyzeTier2 returns bar_range as null (not the perf_notes array)", () => {
		const notes: PerfNote[] = [
			{ pitch: 60, onset: 0.0, offset: 0.4, velocity: 80 },
			{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 85 },
			{ pitch: 64, onset: 1.0, offset: 1.4, velocity: 75 },
		];
		const result = analyzeTier2(notes, [], [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]);

		expect(result.tier).toBe(2);
		expect(result.bar_range).toBeNull();
		expect(Array.isArray(result.dimensions)).toBe(true);
		expect(result.dimensions.length).toBe(6);
	});
});

describe("selectSessionMoments (real WASM)", () => {
	it("returns real within-session moments for a multi-chunk session", async () => {
		const { selectSessionMoments } = await import("./wasm-bridge");
		const chunks = [
			{ chunk_index: 0, scores: [0.55, 0.5, 0.5, 0.54, 0.52, 0.5] as [number, number, number, number, number, number] },
			{ chunk_index: 1, scores: [0.55, 0.5, 0.1, 0.54, 0.52, 0.5] as [number, number, number, number, number, number] },
			{ chunk_index: 2, scores: [0.55, 0.5, 0.48, 0.54, 0.52, 0.5] as [number, number, number, number, number, number] },
		];
		const reference = {
			dynamics: 0.55,
			timing: 0.5,
			pedaling: 0.36,
			articulation: 0.54,
			phrasing: 0.52,
			interpretation: 0.5,
		};
		const moments = selectSessionMoments(chunks, reference, 6);
		expect(Array.isArray(moments)).toBe(true);
		expect(moments.length).toBeGreaterThan(0);
		expect(moments[0]?.dimension).toBe("pedaling");
		expect(moments[0]?.reasoning).toContain("this session");
	});

	it("returns an empty array when fewer than 2 chunks", async () => {
		const { selectSessionMoments } = await import("./wasm-bridge");
		const chunks = [
			{ chunk_index: 0, scores: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3] as [number, number, number, number, number, number] },
		];
		const reference = {
			dynamics: 0.3,
			timing: 0.3,
			pedaling: 0.3,
			articulation: 0.3,
			phrasing: 0.3,
			interpretation: 0.3,
		};
		expect(selectSessionMoments(chunks, reference, 6)).toEqual([]);
	});
});

import { identifyPiece } from "./wasm-bridge";

describe("identifyPiece (real WASM)", () => {
	// Minimal v2 artifact: two pieces. "exact" shares the query's chord-events; "decoy" is disjoint.
	const artifact = JSON.stringify({
		version: "v2",
		onset_tol_ms: 50,
		pieces: [
			{ piece_id: "decoy", composer: "X", title: "Decoy", chroma: new Array(12).fill(0), events: [16, 32, 64, 128] },
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
	const notes: PerfNote[] = [
		{ pitch: 60, onset: 0.0, offset: 0.4, velocity: 100 }, // C  -> bit 0
		{ pitch: 64, onset: 0.5, offset: 0.9, velocity: 100 }, // E  -> bit 4
		{ pitch: 67, onset: 1.0, offset: 1.4, velocity: 100 }, // G  -> bit 7
		{ pitch: 72, onset: 1.5, offset: 1.9, velocity: 100 }, // C  -> bit 0 (octave)
	];

	it("locks to the matching piece", () => {
		const r = identifyPiece(notes, artifact, 0.0935);
		expect(r).not.toBeNull();
		expect(r!.piece_id).toBe("exact");
		expect(r!.locked).toBe(true);
	});

	it("returns null when the artifact has fewer than two pieces", () => {
		const tiny = JSON.stringify({
			version: "v2",
			onset_tol_ms: 50,
			pieces: [{ piece_id: "only", composer: "X", title: "Only", chroma: new Array(12).fill(0), events: [1, 2] }],
		});
		expect(identifyPiece(notes, tiny, 0.0935)).toBeNull();
	});
});
