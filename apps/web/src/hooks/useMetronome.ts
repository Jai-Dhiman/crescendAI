import { useCallback, useRef, useState } from "react";
import { useMountEffect, useSyncRef } from "./useFoundation";

interface MetronomeState {
	isPlaying: boolean;
	bpm: number;
	timeSignature: "4/4" | "3/4" | "6/8";
	accentFirstBeat: boolean;
	currentBeat: number;
}

export interface UseMetronomeReturn extends MetronomeState {
	start: () => void;
	stop: () => void;
	toggle: () => void;
	setBpm: (bpm: number) => void;
	adjustBpm: (delta: number) => void;
	setTimeSignature: (ts: "4/4" | "3/4" | "6/8") => void;
	setAccentFirstBeat: (accent: boolean) => void;
	tapTempo: () => void;
}

const MIN_BPM = 20;
const MAX_BPM = 300;
const TAP_WINDOW_MS = 2000;
const MIN_TAPS = 2;

export function useMetronome(): UseMetronomeReturn {
	const [isPlaying, setIsPlaying] = useState(false);
	const [bpm, setBpmState] = useState(120);
	const [timeSignature, setTimeSignature] = useState<"4/4" | "3/4" | "6/8">(
		"4/4",
	);
	const [accentFirstBeat, setAccentFirstBeat] = useState(true);
	const [currentBeat, setCurrentBeat] = useState(0);

	const audioCtxRef = useRef<AudioContext | null>(null);
	const nextBeatTimeRef = useRef(0);
	const schedulerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const beatCountRef = useRef(0);
	const tapTimesRef = useRef<number[]>([]);

	const beatsPerMeasure =
		timeSignature === "6/8" ? 6 : timeSignature === "3/4" ? 3 : 4;

	// Refs for values accessed in the scheduler interval (avoids stale closures)
	const bpmRef = useSyncRef(bpm);
	const accentRef = useSyncRef(accentFirstBeat);
	const beatsRef = useSyncRef(beatsPerMeasure);

	function getOrCreateAudioCtx(): AudioContext {
		if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
			audioCtxRef.current = new AudioContext();
		}
		return audioCtxRef.current;
	}

	function playClick(time: number, accent: boolean) {
		const ctx = getOrCreateAudioCtx();
		const osc = ctx.createOscillator();
		const gain = ctx.createGain();

		osc.connect(gain);
		gain.connect(ctx.destination);

		// Higher pitch + louder for accent
		osc.frequency.value = accent ? 1000 : 800;
		osc.type = "sine";

		gain.gain.setValueAtTime(accent ? 0.6 : 0.3, time);
		gain.gain.exponentialRampToValueAtTime(0.001, time + 0.05);

		osc.start(time);
		osc.stop(time + 0.05);
	}

	function scheduleBeats() {
		const ctx = getOrCreateAudioCtx();
		const secondsPerBeat = 60 / bpmRef.current;
		const lookahead = 0.1; // schedule 100ms ahead

		while (nextBeatTimeRef.current < ctx.currentTime + lookahead) {
			const beatInMeasure = beatCountRef.current % beatsRef.current;
			const isAccent = accentRef.current && beatInMeasure === 0;

			playClick(nextBeatTimeRef.current, isAccent);
			setCurrentBeat(beatInMeasure);

			beatCountRef.current++;
			nextBeatTimeRef.current += secondsPerBeat;
		}
	}

	// biome-ignore lint/correctness/useExhaustiveDependencies: getOrCreateAudioCtx and scheduleBeats are stable (only read refs)
	const start = useCallback(() => {
		const ctx = getOrCreateAudioCtx();
		if (ctx.state === "suspended") {
			ctx.resume();
		}

		beatCountRef.current = 0;
		nextBeatTimeRef.current = ctx.currentTime;
		setCurrentBeat(0);

		// Schedule at 25ms intervals for tight timing
		schedulerRef.current = setInterval(scheduleBeats, 25);
		setIsPlaying(true);
	}, []);

	const stop = useCallback(() => {
		if (schedulerRef.current) {
			clearInterval(schedulerRef.current);
			schedulerRef.current = null;
		}
		setIsPlaying(false);
		setCurrentBeat(0);
	}, []);

	const toggle = useCallback(() => {
		if (isPlaying) stop();
		else start();
	}, [isPlaying, start, stop]);

	// No restart effect needed -- bpmRef/accentRef/beatsRef are synced above,
	// so the existing 25ms scheduler loop picks up changes on the next tick.

	const setBpm = useCallback((value: number) => {
		setBpmState(Math.max(MIN_BPM, Math.min(MAX_BPM, Math.round(value))));
	}, []);

	const adjustBpm = useCallback((delta: number) => {
		setBpmState((prev) => Math.max(MIN_BPM, Math.min(MAX_BPM, prev + delta)));
	}, []);

	const tapTempo = useCallback(() => {
		const now = performance.now();
		const taps = tapTimesRef.current;

		// Remove stale taps
		while (taps.length > 0 && now - taps[0] > TAP_WINDOW_MS) {
			taps.shift();
		}

		taps.push(now);

		if (taps.length >= MIN_TAPS) {
			const intervals: number[] = [];
			for (let i = 1; i < taps.length; i++) {
				intervals.push(taps[i] - taps[i - 1]);
			}
			const avgInterval =
				intervals.reduce((a, b) => a + b, 0) / intervals.length;
			const newBpm = Math.round(60000 / avgInterval);
			setBpm(newBpm);
		}
	}, [setBpm]);

	// Cleanup on unmount
	useMountEffect(() => {
		return () => {
			if (schedulerRef.current) clearInterval(schedulerRef.current);
			if (audioCtxRef.current?.state !== "closed") {
				audioCtxRef.current?.close();
			}
		};
	});

	return {
		isPlaying,
		bpm,
		timeSignature,
		accentFirstBeat,
		currentBeat,
		start,
		stop,
		toggle,
		setBpm,
		adjustBpm,
		setTimeSignature,
		setAccentFirstBeat,
		tapTempo,
	};
}
