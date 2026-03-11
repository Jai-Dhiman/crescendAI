import { Stop } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { PracticeState } from "../hooks/usePracticeSession";
import type { DimScores, ObservationEvent } from "../lib/practice-api";
import { FlowingWaves } from "./FlowingWaves";
import { ObservationToast } from "./ObservationToast";

interface ListeningModeProps {
	state: PracticeState;
	observations: ObservationEvent[];
	analyserNode: AnalyserNode | null;
	latestScores: DimScores | null;
	error: string | null;
	onStop: () => void;
	originRect: DOMRect | null;
	onExit: () => void;
}

type TransitionPhase = "collapsed" | "expanding" | "open" | "collapsing";

export function ListeningMode({
	state,
	observations,
	analyserNode,
	latestScores,
	error: _error,
	onStop,
	originRect,
	onExit,
}: ListeningModeProps) {
	const [phase, setPhase] = useState<TransitionPhase>("collapsed");
	const [contentVisible, setContentVisible] = useState(false);
	const overlayRef = useRef<HTMLDivElement>(null);
	const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set());
	const [notes, setNotes] = useState("");
	const [showNotepad, setShowNotepad] = useState(false);

	// Compute origin point for clip-path (center of record button, or fallback)
	const originX = originRect
		? ((originRect.left + originRect.width / 2) / window.innerWidth) * 100
		: 50;
	const originY = originRect
		? ((originRect.top + originRect.height / 2) / window.innerHeight) * 100
		: 90;

	// Open transition sequence
	useEffect(() => {
		// Start collapsed, then expand on next frame
		requestAnimationFrame(() => {
			setPhase("expanding");
		});

		const expandTimer = setTimeout(() => {
			setPhase("open");
			setContentVisible(true);
		}, 650); // slightly after 600ms transition

		return () => clearTimeout(expandTimer);
	}, []);

	// Close transition sequence -- defined before the useEffect that references it
	const handleClose = useCallback(() => {
		setContentVisible(false);
		setPhase("collapsing");

		setTimeout(() => {
			setPhase("collapsed");
			setTimeout(() => {
				onStop();
				onExit();
			}, 50);
		}, 600);
	}, [onStop, onExit]);

	// Exit on error or WebSocket disconnect
	useEffect(() => {
		if (state === "error") {
			handleClose();
		}
	}, [state, handleClose]);

	// Stop recording then animate out
	const handleStop = useCallback(() => {
		onStop();
		setContentVisible(false);
		setPhase("collapsing");

		setTimeout(() => {
			setPhase("collapsed");
			setTimeout(onExit, 50);
		}, 600);
	}, [onStop, onExit]);

	const handleDismiss = useCallback((idx: number) => {
		setDismissedIds((prev) => new Set(prev).add(idx));
	}, []);

	const visibleObservations = observations
		.map((obs, idx) => ({ ...obs, idx }))
		.filter(({ idx }) => !dismissedIds.has(idx))
		.slice(-3);

	const isRecording = state === "recording";

	// Screen wake lock
	useEffect(() => {
		let wakeLock: WakeLockSentinel | null = null;
		let cancelled = false;

		(async () => {
			try {
				if ("wakeLock" in navigator) {
					const sentinel = await navigator.wakeLock.request("screen");
					if (cancelled) {
						sentinel.release();
					} else {
						wakeLock = sentinel;
					}
				}
			} catch {
				// Progressive enhancement -- fail silently
			}
		})();

		return () => {
			cancelled = true;
			wakeLock?.release();
		};
	}, []);

	const transitionAttr =
		phase === "collapsing" ? "collapsed" : phase === "expanding" ? "expanding" : phase;

	return createPortal(
		<div
			ref={overlayRef}
			className="listening-overlay"
			data-transition={transitionAttr}
			style={{
				"--origin-x": `${originX}%`,
				"--origin-y": `${originY}%`,
			} as React.CSSProperties}
		>
			{contentVisible && (
				<div className="h-dvh flex flex-col animate-listening-content-in">
					{/* Top bar: piece info */}
					<div className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-border">
						<div className="flex items-center gap-3">
							{/* Metronome placeholder -- Task 6 */}
							<div className="w-10 h-10 rounded-lg bg-surface flex items-center justify-center text-text-secondary text-body-sm">
								M
							</div>
						</div>
						<div className="text-right">
							<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
								Now practicing
							</span>
							<div className="text-body-sm text-cream">
								Unknown piece
							</div>
						</div>
					</div>

					{/* Center: waveform + scores */}
					<div className="flex-1 flex flex-col items-center justify-center px-6 relative">
						{/* Observation toasts */}
						<div className="absolute top-4 right-4 flex flex-col gap-3 z-10">
							{visibleObservations.map(({ idx, text, dimension }) => (
								<ObservationToast
									key={idx}
									text={text}
									dimension={dimension}
									onDismiss={() => handleDismiss(idx)}
								/>
							))}
						</div>

						{/* Waveform */}
						<div className="w-full max-w-3xl h-32 sm:h-40 md:h-48">
							<FlowingWaves analyserNode={analyserNode} active={isRecording} />
						</div>

						{/* Dimension scores */}
						<DimensionScores scores={latestScores} />
					</div>

					{/* Bottom bar: notes + stop */}
					<div className="shrink-0 flex items-center justify-between px-6 py-4 border-t border-border">
						<div>{/* Spacer */}</div>
						<div className="flex items-center gap-4">
							{/* Notepad toggle */}
							<button
								type="button"
								onClick={() => setShowNotepad(!showNotepad)}
								className="w-10 h-10 rounded-lg bg-surface flex items-center justify-center text-text-secondary hover:text-cream transition"
								aria-label="Toggle notepad"
							>
								<span className="text-body-sm">N</span>
							</button>
							{/* Stop button */}
							<button
								type="button"
								onClick={handleStop}
								disabled={!isRecording}
								className="w-14 h-14 flex items-center justify-center rounded-full bg-red-600 hover:bg-red-500 text-on-accent transition-colors disabled:opacity-50"
								aria-label="Stop recording"
							>
								<Stop size={22} weight="fill" />
							</button>
						</div>
					</div>

					{/* Notepad drawer */}
					{showNotepad && (
						<NotepadDrawer
							notes={notes}
							onChange={setNotes}
							onClose={() => setShowNotepad(false)}
						/>
					)}
				</div>
			)}
		</div>,
		document.body,
	);
}

// --- Sub-components ---

function DimensionScores({ scores }: { scores: DimScores | null }) {
	const prevScoresRef = useRef<DimScores | null>(null);
	const [pulsing, setPulsing] = useState<Set<string>>(new Set());

	useEffect(() => {
		if (!scores || !prevScoresRef.current) {
			prevScoresRef.current = scores;
			return;
		}

		const changed = new Set<string>();
		const prev = prevScoresRef.current;
		for (const key of Object.keys(scores) as (keyof DimScores)[]) {
			if (scores[key] !== prev[key]) {
				changed.add(key);
			}
		}

		if (changed.size > 0) {
			setPulsing(changed);
			const timer = setTimeout(() => setPulsing(new Set()), 600);
			prevScoresRef.current = scores;
			return () => clearTimeout(timer);
		}

		prevScoresRef.current = scores;
	}, [scores]);

	const dims: { key: keyof DimScores; label: string }[] = [
		{ key: "dynamics", label: "DYN" },
		{ key: "timing", label: "TIM" },
		{ key: "pedaling", label: "PED" },
		{ key: "articulation", label: "ART" },
		{ key: "phrasing", label: "PHR" },
		{ key: "interpretation", label: "INT" },
	];

	return (
		<div className="flex flex-wrap justify-center gap-x-6 gap-y-3 mt-6">
			{dims.map(({ key, label }) => (
				<div key={key} className="text-center">
					<div
						className={`text-body-md font-semibold text-accent tabular-nums ${
							pulsing.has(key) ? "animate-score-pulse" : ""
						}`}
					>
						{scores ? scores[key].toFixed(1) : "--"}
					</div>
					<div className="text-body-xs text-text-tertiary uppercase tracking-wider">
						{label}
					</div>
				</div>
			))}
		</div>
	);
}

function NotepadDrawer({
	notes,
	onChange,
	onClose,
}: {
	notes: string;
	onChange: (v: string) => void;
	onClose: () => void;
}) {
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	useEffect(() => {
		textareaRef.current?.focus();
	}, []);

	return (
		<>
			{/* Backdrop */}
			<button
				type="button"
				className="fixed inset-0 z-40 bg-black/30"
				onClick={onClose}
				aria-label="Close notepad"
			/>
			{/* Drawer */}
			<div className="fixed bottom-0 left-0 right-0 z-50 bg-espresso border-t border-border rounded-t-2xl max-h-[40vh] md:max-h-[40vh] flex flex-col animate-overlay-in">
				<div className="flex items-center justify-between px-5 py-3 border-b border-border">
					<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
						Notes
					</span>
					<button
						type="button"
						onClick={onClose}
						className="text-body-sm text-accent hover:text-accent-lighter transition"
					>
						Done
					</button>
				</div>
				<div className="flex-1 overflow-y-auto p-4">
					<textarea
						ref={textareaRef}
						value={notes}
						onChange={(e) => onChange(e.target.value)}
						placeholder="Jot down thoughts while you play..."
						className="w-full h-full min-h-[120px] bg-transparent text-body-sm text-cream placeholder:text-text-tertiary outline-none resize-none"
					/>
				</div>
			</div>
		</>
	);
}
