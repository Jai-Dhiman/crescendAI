import { useState } from "react";
import { ArrowsOut } from "@phosphor-icons/react";
import { api } from "../../lib/api";
import type { ExerciseSetConfig } from "../../lib/types";
import { useArtifactStore } from "../../stores/artifact";
import { handsLabel } from "../../lib/exercise-utils";

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
}

function ExerciseItem({ exercise, isExpanded, onToggle, artifactId }: ExerciseItemProps) {
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
				await api.exercises.assign({ exerciseId: exercise.exerciseId! });
				setExerciseStatus(artifactId, exercise.exerciseId, "assigned");
			} catch (err) {
				setExerciseStatus(artifactId, exercise.exerciseId, "error");
			}
		} else {
			setLocalState("loading");
			try {
				await api.exercises.assign({ exerciseId: exercise.exerciseId! });
				setLocalState("assigned");
			} catch (err) {
				setLocalState("error");
			}
		}
	}

	return (
		<div className="border border-border rounded-lg overflow-hidden">
			<button
				type="button"
				onClick={onToggle}
				className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-surface transition"
			>
				<span className="text-body-sm text-cream font-medium">
					{exercise.title}
				</span>
				<div className="flex items-center gap-1.5 ml-2 shrink-0">
					{exercise.hands && (
						<span className="text-body-xs text-text-tertiary bg-surface px-1.5 py-0.5 rounded">
							{handsLabel(exercise.hands)}
						</span>
					)}
					<span className="text-body-xs text-text-tertiary">
						{exercise.focusDimension}
					</span>
				</div>
			</button>
			{isExpanded && (
				<div className="px-3 pb-3 pt-1 border-t border-border">
					<p className="text-body-sm text-text-secondary mb-3">
						{exercise.instruction}
					</p>
					{exercise.exerciseId && (
						<button
							type="button"
							onClick={handleAssign}
							disabled={
								status === "loading" ||
								status === "assigned" ||
								status === "completed"
							}
							className={`text-body-xs px-3 py-1.5 rounded-lg border transition ${
								status === "assigned" || status === "completed"
									? "border-accent text-accent cursor-default"
									: status === "error"
										? "border-red-500 text-red-400 hover:bg-red-500/10"
										: "border-border text-text-secondary hover:text-cream hover:border-accent hover:bg-surface disabled:opacity-50"
							}`}
						>
							{status === "loading"
								? "Assigning..."
								: status === "assigned"
									? "Added to practice"
									: status === "completed"
										? "Completed"
										: status === "error"
											? "Try again"
											: "Try this"}
						</button>
					)}
				</div>
			)}
		</div>
	);
}

export function ExerciseSetCard({ config, onExpand, artifactId }: ExerciseSetCardProps) {
	const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			<div className="flex items-center justify-between mb-1">
				<h4 className="text-body-sm font-medium text-accent">
					{config.targetSkill}
				</h4>
				{onExpand && (
					<button
						type="button"
						onClick={onExpand}
						className="text-text-tertiary hover:text-cream transition p-0.5 -mr-0.5"
						aria-label="Expand exercise set"
					>
						<ArrowsOut size={14} />
					</button>
				)}
			</div>
			<p className="text-body-xs text-text-secondary mb-3">
				{config.sourcePassage}
			</p>
			<div className="space-y-2">
				{config.exercises.map((exercise, i) => (
					<ExerciseItem
						key={exercise.title}
						exercise={exercise}
						isExpanded={expandedIndex === i}
						onToggle={() => setExpandedIndex(expandedIndex === i ? null : i)}
						artifactId={artifactId}
					/>
				))}
			</div>
		</div>
	);
}
