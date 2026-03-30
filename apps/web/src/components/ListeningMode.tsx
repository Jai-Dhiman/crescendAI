import {
	CircleNotch,
	Metronome as MetronomeIcon,
	Minus,
	Plus,
	Stop,
} from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useKeyboardOffset } from "../hooks/useDom";
import { useMountEffect } from "../hooks/useFoundation";
import { useMetronome } from "../hooks/useMetronome";
import type { PracticeState, WsStatus } from "../hooks/usePracticeSession";
import { ResonanceRipples } from "./ResonanceRipples";

interface ListeningModeProps {
	state: PracticeState;
	energy: number;
	isPlaying: boolean;
	error: string | null;
	wsStatus: WsStatus;
	onStop: () => void;
	originRect: DOMRect | null;
	onExit: () => void;
	pieceContext?: { piece: string; section?: string } | null;
	sessionNotes?: string;
	onNotesChange?: (notes: string) => void;
	observations: Array<{ text: string; dimension: string; id: string }>;
}

type TransitionPhase = "collapsed" | "expanding" | "open" | "collapsing";
type ContentVisibility = "hidden" | "visible" | "fading";

export function ListeningMode({
	state,
	energy,
	isPlaying,
	error: _error,
	wsStatus,
	onStop,
	originRect,
	onExit,
	pieceContext,
	sessionNotes,
	onNotesChange,
	observations: _observations,
}: ListeningModeProps) {
	const [phase, setPhase] = useState<TransitionPhase>("collapsed");
	const [contentVis, setContentVis] = useState<ContentVisibility>("hidden");
	const overlayRef = useRef<HTMLDivElement>(null);
	const expandTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const exitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const [ringPhase, setRingPhase] = useState<"collapsed" | "active" | "expanding">("collapsed");
	const notes = sessionNotes ?? "";
	const setNotes = onNotesChange ?? (() => {});
	const [showNotepad, setShowNotepad] = useState(false);
	const [pieceName, setPieceName] = useState(
		pieceContext?.piece ?? "Unknown piece",
	);
	const [sectionName, setSectionName] = useState(pieceContext?.section ?? "");
	const [isEditingPiece, setIsEditingPiece] = useState(false);
	const metronome = useMetronome();
	const [showMetronome, setShowMetronome] = useState(false);

	// Update piece/section when pieceContext arrives asynchronously
	useEffect(() => {
		if (pieceContext) {
			setPieceName(pieceContext.piece);
			if (pieceContext.section) setSectionName(pieceContext.section);
		}
	}, [pieceContext]);

	// Compute origin point for clip-path (center of record button, or fallback)
	const originX = originRect
		? ((originRect.left + originRect.width / 2) / window.innerWidth) * 100
		: 50;
	const originY = originRect
		? ((originRect.top + originRect.height / 2) / window.innerHeight) * 100
		: 90;

	// Open transition sequence
	useMountEffect(() => {
		requestAnimationFrame(() => {
			setPhase("expanding");
			setRingPhase("active");
		});

		const ringTimer = setTimeout(() => setRingPhase("expanding"), 400);

		expandTimerRef.current = setTimeout(() => {
			setPhase("open");
			setContentVis("visible");
			setRingPhase("collapsed");
		}, 800);

		return () => {
			clearTimeout(ringTimer);
			if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
		};
	});

	// Close transition sequence -- defined before the useEffect that references it
	const handleClose = useCallback(() => {
		setContentVis("fading");
		setRingPhase("active");

		fadeTimerRef.current = setTimeout(() => {
			setContentVis("hidden");
			setPhase("collapsing");

			collapseTimerRef.current = setTimeout(() => {
				setPhase("collapsed");
				setRingPhase("collapsed");
				exitTimerRef.current = setTimeout(() => {
					onStop();
					onExit();
				}, 50);
			}, 550);
		}, 100);
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
		setContentVis("fading");
		setRingPhase("active");

		fadeTimerRef.current = setTimeout(() => {
			setContentVis("hidden");
			setPhase("collapsing");

			collapseTimerRef.current = setTimeout(() => {
				setPhase("collapsed");
				setRingPhase("collapsed");
				exitTimerRef.current = setTimeout(onExit, 50);
			}, 550);
		}, 100);
	}, [onStop, onExit]);

	useEffect(() => {
		return () => {
			if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
			if (collapseTimerRef.current) clearTimeout(collapseTimerRef.current);
			if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
			if (exitTimerRef.current) clearTimeout(exitTimerRef.current);
		};
	}, []);

	const isRecording = state === "recording";

	// Screen wake lock
	useMountEffect(() => {
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
	});

	const transitionAttr =
		phase === "collapsing"
			? "collapsed"
			: phase === "expanding"
				? "expanding"
				: phase;

	return createPortal(
		<>
		<div
			ref={overlayRef}
			className="listening-overlay"
			data-transition={transitionAttr}
			style={
				{
					"--origin-x": `${originX}%`,
					"--origin-y": `${originY}%`,
				} as React.CSSProperties
			}
		>
			{contentVis !== "hidden" && (
				<div className={`h-dvh flex flex-col ${
					contentVis === "fading" ? "listening-content-fading" : "animate-listening-content-in"
				}`}>
					{/* Top bar: piece info */}
					<div className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-border">
						<div className="flex items-center gap-3">
							<div className="relative">
								<button
									type="button"
									onClick={() => setShowMetronome(!showMetronome)}
									className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
										metronome.isPlaying
											? "bg-accent/20 text-accent"
											: "bg-surface text-text-secondary hover:text-cream"
									}`}
									aria-label="Toggle metronome"
								>
									<MetronomeIcon
										size={20}
										weight="fill"
										className={metronome.isPlaying ? "animate-pulse" : ""}
									/>
									{metronome.isPlaying && (
										<span className="text-body-sm tabular-nums">
											{metronome.bpm}
										</span>
									)}
								</button>

								{showMetronome && (
									<MetronomePanel
										metronome={metronome}
										onClose={() => setShowMetronome(false)}
									/>
								)}
							</div>
						</div>
						<div className="text-right">
							{isEditingPiece ? (
								<div className="flex flex-col gap-1 items-end">
									<input
										type="text"
										value={pieceName}
										onChange={(e) => setPieceName(e.target.value)}
										placeholder="Piece name"
										className="bg-surface border border-border rounded-lg px-3 py-1 text-body-sm text-cream outline-none w-56"
										// biome-ignore lint/a11y/noAutofocus: intentional UX for inline editor
										autoFocus
									/>
									<input
										type="text"
										value={sectionName}
										onChange={(e) => setSectionName(e.target.value)}
										placeholder="Section (e.g., bars 1-16)"
										className="bg-surface border border-border rounded-lg px-3 py-1 text-body-xs text-cream outline-none w-56"
									/>
									<button
										type="button"
										onClick={() => setIsEditingPiece(false)}
										className="text-body-xs text-accent hover:text-accent-lighter transition mt-1"
									>
										Done
									</button>
								</div>
							) : (
								<button
									type="button"
									onClick={() => setIsEditingPiece(true)}
									className="text-right group"
								>
									<span className="text-label-sm text-text-tertiary uppercase tracking-wider block">
										Now practicing
									</span>
									<span className="text-body-sm text-cream group-hover:text-accent transition">
										{pieceName}
									</span>
									{sectionName && (
										<span className="text-body-xs text-accent ml-2">
											{sectionName}
										</span>
									)}
								</button>
							)}
						</div>
					</div>

					{/* Center: waveform + scores */}
					<div className="flex-1 flex flex-col items-center justify-center px-6 relative">
						{wsStatus === "reconnecting" && isRecording && (
							<div className="absolute top-4 left-4 flex items-center gap-2 text-amber-400 z-10">
								<CircleNotch size={14} className="animate-spin" />
								<span className="text-body-xs">Reconnecting...</span>
							</div>
						)}
						{/* Waveform */}
						<div className="w-full max-w-3xl h-32 sm:h-40 md:h-48">
							<ResonanceRipples energy={energy} isPlaying={isPlaying} active={isRecording} />
						</div>
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
		</div>
		<div
			className="listening-edge-ring"
			data-transition={ringPhase}
			style={{
				left: `${originX}%`,
				top: `${originY}%`,
			}}
		/>
		</>,
		document.body,
	);
}

// --- Sub-components ---

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
	const bottomOffset = useKeyboardOffset();

	useMountEffect(() => {
		textareaRef.current?.focus();
	});

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
			<div
				className="fixed bottom-0 left-0 right-0 z-50 bg-espresso border-t border-border rounded-t-2xl max-h-[60vh] sm:max-h-[40vh] flex flex-col animate-overlay-in"
				style={{ bottom: bottomOffset }}
			>
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

function MetronomePanel({
	metronome,
	onClose,
}: {
	metronome: import("../hooks/useMetronome").UseMetronomeReturn;
	onClose: () => void;
}) {
	const panelContent = (
		<>
			{/* BPM display + controls */}
			<div className="flex items-center justify-center gap-3 mb-4">
				<button
					type="button"
					onClick={() => metronome.adjustBpm(-1)}
					className="w-8 h-8 rounded-full bg-surface-2 flex items-center justify-center text-text-secondary hover:text-cream transition"
					aria-label="Decrease BPM"
				>
					<Minus size={14} />
				</button>
				<div className="text-center">
					<span className="text-display-sm text-cream tabular-nums">
						{metronome.bpm}
					</span>
					<span className="block text-body-xs text-text-tertiary">BPM</span>
				</div>
				<button
					type="button"
					onClick={() => metronome.adjustBpm(1)}
					className="w-8 h-8 rounded-full bg-surface-2 flex items-center justify-center text-text-secondary hover:text-cream transition"
					aria-label="Increase BPM"
				>
					<Plus size={14} />
				</button>
			</div>

			{/* Tap tempo */}
			<button
				type="button"
				onClick={metronome.tapTempo}
				className="w-full py-2 rounded-lg bg-surface-2 text-body-sm text-text-secondary hover:text-cream transition mb-3"
			>
				Tap Tempo
			</button>

			{/* Time signature */}
			<div className="flex gap-2 mb-3">
				{(["4/4", "3/4", "6/8"] as const).map((ts) => (
					<button
						key={ts}
						type="button"
						onClick={() => metronome.setTimeSignature(ts)}
						className={`flex-1 py-1.5 rounded-lg text-body-sm transition ${
							metronome.timeSignature === ts
								? "bg-accent text-on-accent"
								: "bg-surface-2 text-text-secondary hover:text-cream"
						}`}
					>
						{ts}
					</button>
				))}
			</div>

			{/* Accent + On/Off */}
			<div className="flex items-center justify-between">
				<label className="flex items-center gap-2 text-body-sm text-text-secondary cursor-pointer">
					<input
						type="checkbox"
						checked={metronome.accentFirstBeat}
						onChange={(e) => metronome.setAccentFirstBeat(e.target.checked)}
						className="accent-accent"
					/>
					Accent beat 1
				</label>
				<button
					type="button"
					onClick={metronome.toggle}
					className={`px-3 py-1.5 rounded-lg text-body-sm transition ${
						metronome.isPlaying
							? "bg-red-600 text-on-accent hover:bg-red-500"
							: "bg-accent text-on-accent hover:brightness-110"
					}`}
				>
					{metronome.isPlaying ? "Stop" : "Start"}
				</button>
			</div>
		</>
	);

	return (
		<>
			<button
				type="button"
				className="fixed inset-0 z-40"
				onClick={onClose}
				aria-label="Close metronome"
			/>
			{/* Desktop dropdown */}
			<div className="absolute top-full left-0 mt-2 z-50 bg-surface border border-border rounded-xl p-4 min-w-[220px] shadow-card animate-overlay-in hidden md:block">
				{panelContent}
			</div>
			{/* Mobile bottom sheet */}
			<div className="fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border rounded-t-2xl p-4 shadow-card animate-overlay-in md:hidden">
				{panelContent}
			</div>
		</>
	);
}
