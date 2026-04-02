// Practice mode state machine.
// Ported from apps/api-rust/src/practice/session/practice_mode.rs

export enum PracticeMode {
	Warming = "warming",
	Drilling = "drilling",
	Running = "running",
	Winding = "winding",
	Regular = "regular",
}

export interface ChunkSignal {
	chunkIndex: number;
	timestampMs: number;
	barRange: [number, number] | null;
	pitchBigrams: Set<string>;
	hasPieceMatch: boolean;
	barsProgressing: boolean;
	scores: number[];
}

export interface ModeTransition {
	from: PracticeMode;
	to: PracticeMode;
	chunkIndex: number;
	timestampMs: number;
	dwellMs: number;
}

export interface ObservationPolicy {
	minIntervalMs: number;
	suppress: boolean;
	comparative: boolean;
}

interface DrillingPassage {
	barRange: [number, number] | null;
	repetitionCount: number;
	firstScores: number[];
	startedAtChunk: number;
}

// --- Constants ---

const SILENCE_GAP_MS = 60_000;
const WARMING_CHUNK_LIMIT = 4;
const RUNNING_DWELL_MS = 30_000;
const DRILLING_DWELL_MS = 30_000;
const REGULAR_DWELL_MS = 15_000;
const RECENT_WINDOW = 4;
const BAR_OVERLAP_THRESHOLD = 0.5;
const DICE_THRESHOLD = 0.6;

// --- Observation policies ---

const MODE_POLICIES: Record<PracticeMode, ObservationPolicy> = {
	[PracticeMode.Warming]: {
		minIntervalMs: 30_000,
		suppress: false,
		comparative: false,
	},
	[PracticeMode.Drilling]: {
		minIntervalMs: 90_000,
		suppress: false,
		comparative: true,
	},
	[PracticeMode.Running]: {
		minIntervalMs: 150_000,
		suppress: false,
		comparative: false,
	},
	[PracticeMode.Regular]: {
		minIntervalMs: 180_000,
		suppress: false,
		comparative: false,
	},
	[PracticeMode.Winding]: {
		minIntervalMs: 0,
		suppress: true,
		comparative: false,
	},
};

// --- Helpers ---

function barOverlap(a: [number, number], b: [number, number]): number {
	const overlapStart = Math.max(a[0], b[0]);
	const overlapEnd = Math.min(a[1], b[1]);
	if (overlapStart > overlapEnd) return 0;
	const overlap = overlapEnd - overlapStart + 1;
	const spanA = a[1] - a[0] + 1;
	const spanB = b[1] - b[0] + 1;
	return Math.min(overlap / spanA, overlap / spanB);
}

function diceSimilarity(a: Set<string>, b: Set<string>): number {
	if (a.size === 0 || b.size === 0) return 0;
	let intersection = 0;
	for (const item of a) {
		if (b.has(item)) intersection++;
	}
	return (2 * intersection) / (a.size + b.size);
}

function passageHasSubstance(signal: ChunkSignal): boolean {
	return signal.pitchBigrams.size >= 3;
}

// --- Serializable shape for DO storage ---

interface ModeDetectorJSON {
	mode: PracticeMode;
	enteredAtMs: number;
	chunkCount: number;
	recentSignals: Array<{
		chunkIndex: number;
		timestampMs: number;
		barRange: [number, number] | null;
		pitchBigrams: string[];
		hasPieceMatch: boolean;
		barsProgressing: boolean;
		scores: number[];
	}>;
	lastChunkAtMs: number;
	drillingPassage: DrillingPassage | null;
}

// --- ModeDetector ---

export class ModeDetector {
	mode: PracticeMode = PracticeMode.Warming;
	private enteredAtMs = 0;
	private chunkCount = 0;
	private recentSignals: ChunkSignal[] = [];
	private lastChunkAtMs = 0;
	private drillingPassage: DrillingPassage | null = null;

	get observationPolicy(): ObservationPolicy {
		return MODE_POLICIES[this.mode];
	}

	update(signal: ChunkSignal): ModeTransition[] {
		const transitions: ModeTransition[] = [];

		// Two-step silence detection: gap > 60s and not already Winding.
		const gapMs =
			this.lastChunkAtMs === 0 ? 0 : signal.timestampMs - this.lastChunkAtMs;

		if (gapMs > SILENCE_GAP_MS && this.mode !== PracticeMode.Winding) {
			const from = this.mode;
			const dwellMs = this.dwell(signal.timestampMs);
			this.setMode(PracticeMode.Winding, signal.timestampMs);
			transitions.push({
				from,
				to: PracticeMode.Winding,
				chunkIndex: signal.chunkIndex,
				timestampMs: signal.timestampMs,
				dwellMs,
			});

			// Immediately evaluate resume from Winding with the current signal.
			const resumeMode = this.evalFromWinding(signal);
			if (resumeMode !== null) {
				const windingFrom = PracticeMode.Winding;
				this.setMode(resumeMode, signal.timestampMs);
				transitions.push({
					from: windingFrom,
					to: resumeMode,
					chunkIndex: signal.chunkIndex,
					timestampMs: signal.timestampMs,
					dwellMs: 0,
				});
			}

			this.lastChunkAtMs = signal.timestampMs;
			this.chunkCount++;
			this.recentSignals.push(signal);
			if (this.recentSignals.length > RECENT_WINDOW) {
				this.recentSignals.shift();
			}
			return transitions;
		}

		// Push signal into sliding window before evaluating transitions.
		this.recentSignals.push(signal);
		if (this.recentSignals.length > RECENT_WINDOW) {
			this.recentSignals.shift();
		}
		this.lastChunkAtMs = signal.timestampMs;
		this.chunkCount++;

		// Evaluate transitions.
		const next = this.evaluateTransitions(signal);
		if (next !== null) {
			if (next === PracticeMode.Drilling) {
				if (this.mode === PracticeMode.Drilling) {
					// Already drilling: increment repetition count (no transition emitted).
					if (this.drillingPassage !== null) {
						this.drillingPassage.repetitionCount++;
					}
				} else {
					// Entering drilling: initialize passage.
					const from = this.mode;
					const dwellMs = this.dwell(signal.timestampMs);
					this.drillingPassage = {
						barRange: signal.barRange,
						repetitionCount: 1,
						firstScores: [...signal.scores],
						startedAtChunk: signal.chunkIndex,
					};
					this.setMode(PracticeMode.Drilling, signal.timestampMs);
					transitions.push({
						from,
						to: PracticeMode.Drilling,
						chunkIndex: signal.chunkIndex,
						timestampMs: signal.timestampMs,
						dwellMs,
					});
				}
			} else {
				const from = this.mode;
				const dwellMs = this.dwell(signal.timestampMs);
				this.setMode(next, signal.timestampMs);
				transitions.push({
					from,
					to: next,
					chunkIndex: signal.chunkIndex,
					timestampMs: signal.timestampMs,
					dwellMs,
				});
			}
		}

		return transitions;
	}

	// For DO storage consumption -- take the drilling passage on exit.
	takeDrillingPassage(): DrillingPassage | null {
		const dp = this.drillingPassage;
		this.drillingPassage = null;
		return dp;
	}

	toJSON(): unknown {
		return {
			mode: this.mode,
			enteredAtMs: this.enteredAtMs,
			chunkCount: this.chunkCount,
			recentSignals: this.recentSignals.map((s) => ({
				chunkIndex: s.chunkIndex,
				timestampMs: s.timestampMs,
				barRange: s.barRange,
				pitchBigrams: [...s.pitchBigrams],
				hasPieceMatch: s.hasPieceMatch,
				barsProgressing: s.barsProgressing,
				scores: s.scores,
			})),
			lastChunkAtMs: this.lastChunkAtMs,
			drillingPassage: this.drillingPassage,
		} satisfies ModeDetectorJSON;
	}

	static fromJSON(data: unknown): ModeDetector {
		const d = data as ModeDetectorJSON;
		const det = new ModeDetector();
		det.mode = d.mode;
		det.enteredAtMs = d.enteredAtMs;
		det.chunkCount = d.chunkCount;
		det.recentSignals = d.recentSignals.map((s) => ({
			chunkIndex: s.chunkIndex,
			timestampMs: s.timestampMs,
			barRange: s.barRange,
			pitchBigrams: new Set(s.pitchBigrams),
			hasPieceMatch: s.hasPieceMatch,
			barsProgressing: s.barsProgressing,
			scores: s.scores,
		}));
		det.lastChunkAtMs = d.lastChunkAtMs;
		det.drillingPassage = d.drillingPassage;
		return det;
	}

	// --- Private helpers ---

	private setMode(mode: PracticeMode, timestampMs: number): void {
		this.mode = mode;
		this.enteredAtMs = timestampMs;
	}

	private dwell(currentTsMs: number): number {
		return Math.max(0, currentTsMs - this.enteredAtMs);
	}

	private dwellElapsed(dwellMs: number, currentTsMs: number): boolean {
		return this.dwell(currentTsMs) >= dwellMs;
	}

	private detectRepetition(): boolean {
		if (this.recentSignals.length < 2) return false;

		// Take the last 3 signals, examine consecutive pairs.
		const recent = this.recentSignals.slice(-3);
		const totalPairs = recent.length - 1;
		let repeatCount = 0;

		for (let i = 0; i < recent.length - 1; i++) {
			const a = recent[i];
			const b = recent[i + 1];

			// Layer 1: bar range overlap (preferred when available).
			if (a.barRange !== null && b.barRange !== null) {
				if (barOverlap(a.barRange, b.barRange) >= BAR_OVERLAP_THRESHOLD) {
					repeatCount++;
					continue;
				}
			}

			// Layer 2: pitch bigram Dice (fallback).
			if (diceSimilarity(a.pitchBigrams, b.pitchBigrams) >= DICE_THRESHOLD) {
				repeatCount++;
			}
		}

		return repeatCount >= totalPairs;
	}

	private noRecentRepetition(): boolean {
		if (this.recentSignals.length < 2) return true;
		const latest = this.recentSignals[this.recentSignals.length - 1];
		const prev = this.recentSignals[this.recentSignals.length - 2];

		if (latest.barRange !== null && prev.barRange !== null) {
			if (barOverlap(latest.barRange, prev.barRange) >= BAR_OVERLAP_THRESHOLD) {
				return false;
			}
		}

		if (
			diceSimilarity(latest.pitchBigrams, prev.pitchBigrams) >= DICE_THRESHOLD
		) {
			return false;
		}

		return true;
	}

	private barsProgressing(): boolean {
		if (this.recentSignals.length < 2) return false;
		const latest = this.recentSignals[this.recentSignals.length - 1];
		const prev = this.recentSignals[this.recentSignals.length - 2];
		if (latest.barRange !== null && prev.barRange !== null) {
			return (
				latest.barRange[0] > prev.barRange[0] ||
				latest.barRange[1] > prev.barRange[1]
			);
		}
		return false;
	}

	private evaluateTransitions(signal: ChunkSignal): PracticeMode | null {
		switch (this.mode) {
			case PracticeMode.Warming:
				return this.evalFromWarming(signal);
			case PracticeMode.Running:
				return this.evalFromRunning(signal);
			case PracticeMode.Drilling:
				return this.evalFromDrilling(signal);
			case PracticeMode.Regular:
				return this.evalFromRegular(signal);
			case PracticeMode.Winding:
				return this.evalFromWinding(signal);
		}
	}

	private evalFromWarming(signal: ChunkSignal): PracticeMode | null {
		// Transition to Running: piece match + bar progress.
		if (signal.hasPieceMatch && this.barsProgressing()) {
			return PracticeMode.Running;
		}

		// Transition to Drilling: repetition detected across at least 3 signals
		// with a substantive passage (>= 3 bigrams) to avoid false triggers.
		if (
			this.recentSignals.length >= 3 &&
			passageHasSubstance(signal) &&
			this.detectRepetition()
		) {
			return PracticeMode.Drilling;
		}

		// Fallback to Regular after WARMING_CHUNK_LIMIT ambiguous chunks.
		if (this.chunkCount >= WARMING_CHUNK_LIMIT) {
			return PracticeMode.Regular;
		}

		return null;
	}

	private evalFromRunning(signal: ChunkSignal): PracticeMode | null {
		if (!this.dwellElapsed(RUNNING_DWELL_MS, signal.timestampMs)) return null;

		if (this.detectRepetition()) {
			return PracticeMode.Drilling;
		}

		return null;
	}

	private evalFromDrilling(signal: ChunkSignal): PracticeMode | null {
		if (!this.dwellElapsed(DRILLING_DWELL_MS, signal.timestampMs)) {
			// Still increment repetition count during dwell.
			if (this.detectRepetition()) {
				return PracticeMode.Drilling; // signals "stay + increment"
			}
			return null;
		}

		// Transition to Running when new material appears.
		if (
			this.noRecentRepetition() &&
			signal.hasPieceMatch &&
			this.barsProgressing()
		) {
			return PracticeMode.Running;
		}

		// Stay in drilling if still repeating.
		if (this.detectRepetition()) {
			return PracticeMode.Drilling;
		}

		return null;
	}

	private evalFromRegular(signal: ChunkSignal): PracticeMode | null {
		if (!this.dwellElapsed(REGULAR_DWELL_MS, signal.timestampMs)) return null;

		if (signal.hasPieceMatch && this.barsProgressing()) {
			return PracticeMode.Running;
		}

		if (this.detectRepetition()) {
			return PracticeMode.Drilling;
		}

		return null;
	}

	private evalFromWinding(signal: ChunkSignal): PracticeMode | null {
		if (signal.hasPieceMatch) {
			return PracticeMode.Running;
		}
		return PracticeMode.Regular;
	}
}
