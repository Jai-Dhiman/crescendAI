import { ArrowsOut, CaretDown } from "@phosphor-icons/react";
import { useState } from "react";
import { api } from "../../lib/api";
import { handsLabel } from "../../lib/exercise-utils";
import type { ExerciseSetConfig } from "../../lib/types";
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

	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
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
