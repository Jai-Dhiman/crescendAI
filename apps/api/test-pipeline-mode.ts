// Isolated ModeDetector test driven by real MuQ chunk scores.
// Run: cd apps/api && bun test-pipeline-mode.ts [path-to-chunk-scores.json]
//
// Inputs:
//   per-chunk MuQ scores JSON (output of apps/inference/muq_chunk_compare.py --out-json)
//   format: { label: number[][] } where each inner array is [6-dim scores]
//
// For Chopin we can't supply piece-ID/AMT data (not in catalog, AMT not yet run),
// so barRange=null, pitchBigrams=empty, hasPieceMatch=false, barsProgressing=false.
// This exercises the "Tier-3-equivalent" path through the mode detector — i.e.,
// what happens in production today for any non-catalog piece.

import { readFileSync } from "node:fs";
import { ModeDetector, type ChunkSignal } from "./src/services/practice-mode";

const path =
	process.argv[2] ??
	"/Users/jdhiman/Documents/crescendai/model/data/results/pipeline_test/chopin_per_chunk.json";

const data = JSON.parse(readFileSync(path, "utf8")) as Record<string, number[][]>;
const [label] = Object.keys(data);
const chunks = data[label] ?? [];

console.log(`Recording: ${label} (${chunks.length} chunks @ 15s = ${(chunks.length * 15).toFixed(0)}s)`);
console.log(`Inputs: barRange=null, pitchBigrams=empty, hasPieceMatch=false (no piece-ID / no AMT)\n`);

const md = new ModeDetector();
let totalTransitions = 0;

chunks.forEach((scores, i) => {
	const signal: ChunkSignal = {
		chunkIndex: i,
		timestampMs: i * 15_000,
		barRange: null,
		pitchBigrams: new Set<string>(),
		hasPieceMatch: false,
		barsProgressing: false,
		scores,
	};
	const transitions = md.update(signal);
	for (const t of transitions) {
		console.log(
			`  chunk ${i} (t=${(signal.timestampMs / 1000).toFixed(0)}s): ${t.from} -> ${t.to}  (dwell ${(t.dwellMs / 1000).toFixed(1)}s)`,
		);
		totalTransitions++;
	}
});

console.log(`\nFinal mode: ${md.mode}`);
console.log(`Total transitions emitted: ${totalTransitions}`);
console.log(`Observation policy at end: ${JSON.stringify(md.observationPolicy)}`);

// Quick sanity: also try a synthetic stream that fakes piece-match + repeated bars
// (what we'd see if piece-ID worked and the student looped bars 12-15 three times).
console.log(`\n--- Synthetic drilling stream (sim 6 chunks, all bars 12-15) ---`);
const md2 = new ModeDetector();
const synthBigrams = new Set(["60,64", "64,67", "67,72"]);
for (let i = 0; i < 6; i++) {
	const ts = md2.update({
		chunkIndex: i,
		timestampMs: i * 15_000,
		barRange: [12, 15],
		pitchBigrams: synthBigrams,
		hasPieceMatch: true,
		barsProgressing: false,
		scores: [0.5, 0.5, 0.6, 0.5, 0.5, 0.55],
	});
	for (const t of ts) {
		console.log(
			`  chunk ${i}: ${t.from} -> ${t.to}  (dwell ${(t.dwellMs / 1000).toFixed(1)}s)`,
		);
	}
}
console.log(`Final mode (synthetic): ${md2.mode}`);
