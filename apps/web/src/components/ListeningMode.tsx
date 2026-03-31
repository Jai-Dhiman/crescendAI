import {
	CircleNotch,
	Metronome as MetronomeIcon,
	Minus,
	Plus,
	Stop,
} from "@phosphor-icons/react";
import { AnimatePresence, LazyMotion, domAnimation, m } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useKeyboardOffset } from "../hooks/useDom";
import { useMountEffect } from "../hooks/useFoundation";
import { useMetronome } from "../hooks/useMetronome";
import type { PracticeState, WsStatus } from "../hooks/usePracticeSession";
import { AudioWaveformRing } from "./AudioWaveformRing";

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
	analyserNode: AnalyserNode | null;
}

/** Custom easing matching the original cubic-bezier(0.16, 1, 0.3, 1) */
const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

export function ListeningMode({
	state,
	energy: _energy,
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
	analyserNode,
}: ListeningModeProps) {
	const [isOpen, setIsOpen] = useState(true);
	const [contentVisible, setContentVisible] = useState(false);
	const overlayRef = useRef<HTMLDivElement>(null);
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

	const clipOrigin = `${originX}% ${originY}%`;

	// Show content after overlay expansion completes
	useMountEffect(() => {
		const timer = setTimeout(() => setContentVisible(true), 750);
		return () => clearTimeout(timer);
	});

	// Exit on error
	useEffect(() => {
		if (state === "error") {
			setIsOpen(false);
		}
	}, [state]);

	// Stop recording then animate out
	const handleStop = useCallback(() => {
		onStop();
		setIsOpen(false);
	}, [onStop]);

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
				// Progressive enhancement
			}
		})();

		return () => {
			cancelled = true;
			wakeLock?.release();
		};
	});

	return createPortal(
		<LazyMotion features={domAnimation}>
		<AnimatePresence onExitComplete={onExit}>
			{isOpen && (
				<>
					{/* Edge ring -- expands from origin with visible border */}
					<m.div
						key="edge-ring"
						className="fixed z-49 pointer-events-none rounded-full"
						style={{
							left: `${originX}%`,
							top: `${originY}%`,
							x: "-50%",
							y: "-50%",
							border: "2px solid rgba(122, 154, 130, 0.7)",
							boxShadow: "0 0 20px rgba(122, 154, 130, 0.3), inset 0 0 20px rgba(122, 154, 130, 0.1)",
						}}
						initial={{ width: 0, height: 0, opacity: 0 }}
						animate={{
							width: "300vmax",
							height: "300vmax",
							opacity: [0, 1, 1, 0],
							transition: {
								width: { duration: 0.75, ease: EASE_OUT_EXPO },
								height: { duration: 0.75, ease: EASE_OUT_EXPO },
								opacity: {
									duration: 0.75,
									times: [0, 0.1, 0.7, 1],
									ease: "easeOut",
								},
							},
						}}
						exit={{
							width: 0,
							height: 0,
							opacity: 0,
							transition: { duration: 0.5, ease: EASE_OUT_EXPO },
						}}
					/>

					{/* Main overlay -- clip-path circle expansion */}
					<m.div
						key="overlay"
						ref={overlayRef}
						className="fixed inset-0 z-50 bg-espresso"
						initial={{
							clipPath: `circle(0% at ${clipOrigin})`,
						}}
						animate={{
							clipPath: `circle(150vmax at ${clipOrigin})`,
							transition: { duration: 0.75, ease: EASE_OUT_EXPO },
						}}
						exit={{
							clipPath: `circle(0% at ${clipOrigin})`,
							transition: { duration: 0.5, ease: EASE_OUT_EXPO },
						}}
					>
						<AnimatePresence>
							{contentVisible && (
								<m.div
									className="h-dvh flex flex-col"
									initial={{ opacity: 0 }}
									animate={{ opacity: 1 }}
									exit={{ opacity: 0 }}
									transition={{ duration: 0.2 }}
								>
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

									{/* Center: waveform */}
									<div className="flex-1 flex flex-col items-center justify-center px-6 relative">
										{wsStatus === "reconnecting" && isRecording && (
											<div className="absolute top-4 left-4 flex items-center gap-2 text-amber-400 z-10">
												<CircleNotch size={14} className="animate-spin" />
												<span className="text-body-xs">Reconnecting...</span>
											</div>
										)}
										{/* Waveform */}
										<div className="w-full max-w-md aspect-square">
											<AudioWaveformRing
												analyserNode={analyserNode}
												isPlaying={isPlaying}
												active={isRecording}
											/>
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
								</m.div>
							)}
						</AnimatePresence>
					</m.div>
				</>
			)}
		</AnimatePresence>
		</LazyMotion>,
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
			<m.div
				className="fixed bottom-0 left-0 right-0 z-50 bg-espresso border-t border-border rounded-t-2xl max-h-[60vh] sm:max-h-[40vh] flex flex-col"
				style={{ bottom: bottomOffset }}
				initial={{ opacity: 0, y: 20, scale: 0.95 }}
				animate={{ opacity: 1, y: 0, scale: 1 }}
				transition={{ duration: 0.35, ease: EASE_OUT_EXPO }}
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
			</m.div>
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
			<m.div
				className="absolute top-full left-0 mt-2 z-50 bg-surface border border-border rounded-xl p-4 min-w-[220px] shadow-card hidden md:block"
				initial={{ opacity: 0, y: -8, scale: 0.95 }}
				animate={{ opacity: 1, y: 0, scale: 1 }}
				transition={{ duration: 0.25, ease: EASE_OUT_EXPO }}
			>
				{panelContent}
			</m.div>
			{/* Mobile bottom sheet */}
			<m.div
				className="fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border rounded-t-2xl p-4 shadow-card md:hidden"
				initial={{ opacity: 0, y: 20 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.35, ease: EASE_OUT_EXPO }}
			>
				{panelContent}
			</m.div>
		</>
	);
}
