import { ArrowsOut, CaretDown } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { ClipSvg } from "../ClipSvg";
import { LoopTransport } from "../LoopTransport";
import { scoreRenderer } from "../../lib/score-renderer";
import { ScoreCursor } from "../../lib/score-cursor";
import { api } from "../../lib/api";
import { handsLabel } from "../../lib/exercise-utils";
import { useLoopPlayer } from "../../hooks/useLoopPlayer";
import type { ExerciseSetConfig } from "../../lib/types";
import type { ScoreIR } from "../../lib/score-ir";
import type { ClipNote } from "../../lib/score-worker";
import { useArtifactStore } from "../../stores/artifact";

interface ExerciseSetCardProps {
	config: ExerciseSetConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type LocalAssignState = "idle" | "loading" | "assigned" | "error";

interface ExerciseItemProps {
	exercise: ExerciseSetConfig["exercises"][number];
	isExpanded: boolean;
	onToggle: () => void;
	artifactId?: string;
	isFirst: boolean;
}

function ExerciseItem({
	exercise,
	isExpanded,
	onToggle,
	artifactId,
	isFirst,
}: ExerciseItemProps) {
	const [localState, setLocalState] = useState<LocalAssignState>("idle");

	const exerciseState = useArtifactStore((s) => {
		if (!artifactId || !exercise.exerciseId) return undefined;
		return s.states[artifactId]?.exerciseStates?.[exercise.exerciseId];
	});

	const setExerciseStatus = useArtifactStore((s) => s.setExerciseStatus);

	const useStore = Boolean(artifactId);
	const status = useStore ? (exerciseState?.status ?? "idle") : localState;

	async function handleAssign() {
		if (!exercise.exerciseId) return;

		if (useStore && artifactId) {
			setExerciseStatus(artifactId, exercise.exerciseId, "loading");
			try {
				await api.exercises.assign({ exerciseId: exercise.exerciseId });
				setExerciseStatus(artifactId, exercise.exerciseId, "assigned");
			} catch {
				setExerciseStatus(artifactId, exercise.exerciseId, "error");
			}
		} else {
			setLocalState("loading");
			try {
				await api.exercises.assign({ exerciseId: exercise.exerciseId });
				setLocalState("assigned");
			} catch {
				setLocalState("error");
			}
		}
	}

	const actionLabel =
		status === "loading"
			? "Saving..."
			: status === "assigned"
				? "Saved"
				: status === "completed"
					? "Completed"
					: status === "error"
						? "Try again"
						: "Add to practice";

	const actionClass =
		status === "assigned" || status === "completed"
			? "border-accent/60 text-accent cursor-default"
			: "border-border text-text-secondary hover:border-accent hover:text-cream";

	return (
		<div>
			{!isFirst && <div className="border-t border-border/50 mx-4" />}
			<button
				type="button"
				onClick={onToggle}
				className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-surface/30 transition-colors group"
			>
				<span className="text-body-sm text-text-primary group-hover:text-cream transition-colors">
					{exercise.title}
				</span>
				<CaretDown
					size={12}
					weight="bold"
					className={`text-text-tertiary shrink-0 transition-transform duration-200 ${
						isExpanded ? "-rotate-180" : ""
					}`}
				/>
			</button>
			{isExpanded && (
				<div className="px-4 pb-4 flex flex-col gap-3">
					<p className="text-body-sm text-text-secondary leading-relaxed">
						{exercise.instruction}
					</p>
					<div className="flex items-center justify-between gap-4">
						<div className="flex items-center gap-3 min-w-0">
							{exercise.hands && (
								<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
									{handsLabel(exercise.hands)}
								</span>
							)}
							{exercise.focusDimension && (
								<span className="text-label-sm text-text-tertiary">
									{exercise.focusDimension}
								</span>
							)}
						</div>
						{exercise.exerciseId && (
							<button
								type="button"
								onClick={handleAssign}
								disabled={
									status === "loading" ||
									status === "assigned" ||
									status === "completed"
								}
								className={`shrink-0 text-body-xs px-3 py-1.5 rounded-lg border transition-colors disabled:opacity-60 ${actionClass}`}
							>
								{actionLabel}
							</button>
						)}
					</div>
				</div>
			)}
		</div>
	);
}

export function ExerciseSetCard({
	config,
	onExpand,
	artifactId,
}: ExerciseSetCardProps) {
	const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
	const [scoreClipSvg, setScoreClipSvg] = useState<string | null>(null);
	const [clipLoadError, setClipLoadError] = useState(false);
	const [clipIR, setClipIR] = useState<ScoreIR | null>(null);
	const [clipNotes, setClipNotes] = useState<ClipNote[]>([]);
	const scoreContainerRef = useRef<HTMLDivElement>(null);

	const hasTempoFactor = !!config.scoreClip?.tempoFactor;

	useEffect(() => {
		if (!config.scoreClip) return;
		let cancelled = false;
		const { pieceId, bars, transpose } = config.scoreClip;
		// own_passage clips carry no transpose; corpus drills carry the semitone
		// shift into the student's key. Pass a literal 0 (never undefined) so the
		// renderer's composite cache key ("pieceId:transpose") matches own_passage's
		// transpose-0 load byte-for-byte.
		const transposeSemitones = transpose ?? 0;
		// getClip/getClipPlayback require the piece bytes to have been sent to the
		// worker via load() first (the worker errors "bytes required on first
		// request" otherwise). Production preloads via ScorePanel, but an exercise
		// card can render with no panel open, so load() here makes the card
		// self-sufficient. load() is idempotent (cached), so it's a no-op when the
		// piece was already loaded upstream.
		(async () => {
			const loaded = await scoreRenderer.load(pieceId, transposeSemitones);
			if (cancelled) return;
			if (loaded === "failed") {
				console.error("ExerciseSetCard: failed to load score for clip", pieceId);
				setClipLoadError(true);
				return;
			}
			try {
				if (hasTempoFactor) {
					const r = await scoreRenderer.getClipPlayback(
						pieceId,
						bars[0],
						bars[1],
						transposeSemitones,
					);
					if (!cancelled) {
						setScoreClipSvg(r.svg);
						setClipIR(r.ir);
						setClipNotes(r.notes);
					}
				} else {
					const svg = await scoreRenderer.getClip(
						pieceId,
						bars[0],
						bars[1],
						transposeSemitones,
					);
					if (!cancelled) setScoreClipSvg(svg);
				}
			} catch (err) {
				console.error("ExerciseSetCard: failed to load score clip", err);
				if (!cancelled) setClipLoadError(true);
			}
		})();
		return () => { cancelled = true; };
	}, [config.scoreClip, hasTempoFactor]);

	const loopPlayer = useLoopPlayer({
		clipIR: hasTempoFactor ? clipIR : null,
		clipNotes,
		beatsPerBar: 4,
		bpmAtUnity: 120,
		tempoFactor: config.scoreClip?.tempoFactor ?? 1.0,
	});

	useEffect(() => {
		if (!hasTempoFactor || clipIR === null || scoreContainerRef.current === null) return;
		const cursor = new ScoreCursor({
			pieceId: config.scoreClip!.pieceId,
			container: scoreContainerRef.current,
			ir: clipIR,
			qstampSource: loopPlayer.qstampSource,
		});
		cursor.start();
		return () => cursor.stop();
	}, [clipIR, hasTempoFactor, config.scoreClip?.pieceId, loopPlayer.qstampSource]);

	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{/* HERO: Score clip */}
			{config.scoreClip && scoreClipSvg && !clipLoadError && (
				<div
					ref={hasTempoFactor ? scoreContainerRef : undefined}
					className="border-b border-border/60 bg-white"
				>
					<ClipSvg svg={scoreClipSvg} />
				</div>
			)}

			{/* Transport (own_passage_loop with tempoFactor only) */}
			{hasTempoFactor && scoreClipSvg && !clipLoadError && (
				<LoopTransport
					isPlaying={loopPlayer.isPlaying}
					isCounting={loopPlayer.isCounting}
					audioUnavailable={loopPlayer.audioUnavailable}
					tempoFactor={loopPlayer.tempoFactor}
					onPlay={loopPlayer.play}
					onPause={loopPlayer.pause}
					onStop={loopPlayer.stop}
					onTempoChange={loopPlayer.setTempoFactor}
				/>
			)}

			{/* Header */}
			<div className="px-4 pt-4 pb-3 flex items-start justify-between gap-3">
				<div className="min-w-0">
					<h4 className="font-display text-body-md text-text-primary leading-snug">
						{config.targetSkill}
					</h4>
					<p className="text-body-xs text-text-tertiary mt-0.5 truncate">
						{config.sourcePassage}
					</p>
				</div>
				{onExpand && (
					<button
						type="button"
						onClick={onExpand}
						className="shrink-0 text-text-tertiary hover:text-cream transition-colors pt-0.5"
						aria-label="Expand exercise set"
					>
						<ArrowsOut size={14} />
					</button>
				)}
			</div>

			{/* Divider */}
			<div className="border-t border-border/60" />

			{/* Exercise rows */}
			<div>
				{config.exercises.map((exercise, i) => (
					<ExerciseItem
						key={exercise.title}
						exercise={exercise}
						isExpanded={expandedIndex === i}
						onToggle={() => setExpandedIndex(expandedIndex === i ? null : i)}
						artifactId={artifactId}
						isFirst={i === 0}
					/>
				))}
			</div>
		</div>
	);
}
