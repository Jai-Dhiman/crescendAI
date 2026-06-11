import { useCallback, useEffect, useRef, useState } from "react";
import { LoopPlayer } from "../lib/loop-player";
import type { ClipNote } from "../lib/score-worker";
import type { ScoreIR } from "../lib/score-ir";

export interface UseLoopPlayerConfig {
	clipIR: ScoreIR | null;
	clipNotes: ClipNote[];
	beatsPerBar: number;
	bpmAtUnity: number;
	tempoFactor: number;
}

export interface UseLoopPlayerReturn {
	isPlaying: boolean;
	isCounting: boolean;
	audioUnavailable: boolean;
	tempoFactor: number;
	play: () => void;
	pause: () => void;
	stop: () => void;
	setTempoFactor: (f: number) => void;
	qstampSource: () => number | null;
}

export function useLoopPlayer(config: UseLoopPlayerConfig): UseLoopPlayerReturn {
	const playerRef = useRef<LoopPlayer | null>(null);
	const [isPlaying, setIsPlaying] = useState(false);
	const [isCounting, setIsCounting] = useState(false);
	const [audioUnavailable, setAudioUnavailable] = useState(false);
	const [tempoFactor, setTempoFactorState] = useState(config.tempoFactor);
	const ctxRef = useRef<AudioContext | null>(null);

	// Keep clipNotes in a ref so LoopPlayer always reads the current value
	// WITHOUT triggering the player-creation effect on every render.
	const clipNotesRef = useRef<ClipNote[]>(config.clipNotes);
	useEffect(() => {
		clipNotesRef.current = config.clipNotes;
	});

	// Destroy player and AudioContext on unmount only.
	useEffect(() => {
		return () => {
			playerRef.current?.destroy();
			playerRef.current = null;
			ctxRef.current?.close().catch(() => {});
			ctxRef.current = null;
		};
	}, []);

	// Create (or recreate) the LoopPlayer when clipIR first becomes non-null,
	// or when the stable numeric config changes.
	// clipNotes intentionally excluded — read via clipNotesRef.
	// tempoFactor intentionally excluded — live changes go through setTempoFactor().
	useEffect(() => {
		if (!config.clipIR) return;
		if (playerRef.current) {
			playerRef.current.destroy();
		}
		if (!ctxRef.current || ctxRef.current.state === "closed") {
			ctxRef.current = new AudioContext();
		}
		playerRef.current = new LoopPlayer({
			ctx: ctxRef.current,
			instrumentUrl: "/soundfonts/acoustic_grand_piano-mp3.js",
			clipIR: config.clipIR,
			clipNotes: clipNotesRef.current,
			beatsPerBar: config.beatsPerBar,
			bpmAtUnity: config.bpmAtUnity,
			tempoFactor: config.tempoFactor,
		});
		setIsPlaying(false);
		setIsCounting(false);
	// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [config.clipIR, config.beatsPerBar, config.bpmAtUnity]);
	// config.tempoFactor and config.clipNotes intentionally omitted from deps.

	const play = useCallback(() => {
		const player = playerRef.current;
		if (!player) return;
		player.play().then(() => {
			setIsPlaying(true);
			setIsCounting(player.state === "counting-in");
			setAudioUnavailable(player.audioUnavailable);
			// Poll until count-in ends (player transitions counting-in → playing).
			// Interval clears itself once state moves past "counting-in".
			if (player.state === "counting-in") {
				const countInWatcher = setInterval(() => {
					if (playerRef.current?.state !== "counting-in") {
						setIsCounting(false);
						clearInterval(countInWatcher);
					}
				}, 50);
			}
		}).catch((err: unknown) => {
			console.error("[useLoopPlayer] play() failed:", err);
			setAudioUnavailable(true);
		});
	}, []);

	const pause = useCallback(() => {
		playerRef.current?.pause();
		setIsPlaying(false);
		setIsCounting(false);
	}, []);

	const stop = useCallback(() => {
		playerRef.current?.stop();
		setIsPlaying(false);
		setIsCounting(false);
	}, []);

	const setTempoFactor = useCallback((f: number) => {
		playerRef.current?.setTempoFactor(f);
		setTempoFactorState(f);
	}, []);

	const qstampSource = useCallback((): number | null => {
		return playerRef.current?.qstampSource() ?? null;
	}, []);

	return { isPlaying, isCounting, audioUnavailable, tempoFactor, play, pause, stop, setTempoFactor, qstampSource };
}
