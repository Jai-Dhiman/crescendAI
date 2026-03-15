import { useState } from "react";
import { api } from "../../lib/api";
import type { ExerciseSetConfig } from "../../lib/types";

interface ExerciseSetCardProps {
	config: ExerciseSetConfig;
}

type AssignState = "idle" | "loading" | "assigned" | "error";

interface ExerciseItemProps {
	exercise: ExerciseSetConfig["exercises"][number];
	isExpanded: boolean;
	onToggle: () => void;
}

function handsLabel(hands: "left" | "right" | "both"): string {
	if (hands === "left") return "LH";
	if (hands === "right") return "RH";
	return "Both";
}

function ExerciseItem({ exercise, isExpanded, onToggle }: ExerciseItemProps) {
	const [assignState, setAssignState] = useState<AssignState>("idle");

	async function handleAssign() {
		if (!exercise.exercise_id) return;
		setAssignState("loading");
		try {
			await api.exercises.assign({ exercise_id: exercise.exercise_id });
			setAssignState("assigned");
		} catch (err) {
			setAssignState("error");
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
						{exercise.focus_dimension}
					</span>
				</div>
			</button>
			{isExpanded && (
				<div className="px-3 pb-3 pt-1 border-t border-border">
					<p className="text-body-sm text-text-secondary mb-3">
						{exercise.instruction}
					</p>
					{exercise.exercise_id && (
						<button
							type="button"
							onClick={handleAssign}
							disabled={assignState === "loading" || assignState === "assigned"}
							className={`text-body-xs px-3 py-1.5 rounded-lg border transition ${
								assignState === "assigned"
									? "border-accent text-accent cursor-default"
									: assignState === "error"
										? "border-red-500 text-red-400 hover:bg-red-500/10"
										: "border-border text-text-secondary hover:text-cream hover:border-accent hover:bg-surface disabled:opacity-50"
							}`}
						>
							{assignState === "loading"
								? "Assigning..."
								: assignState === "assigned"
									? "Added to practice"
									: assignState === "error"
										? "Try again"
										: "Try this"}
						</button>
					)}
				</div>
			)}
		</div>
	);
}

export function ExerciseSetCard({ config }: ExerciseSetCardProps) {
	const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			<h4 className="text-body-sm font-medium text-accent mb-1">
				{config.target_skill}
			</h4>
			<p className="text-body-xs text-text-secondary mb-3">
				{config.source_passage}
			</p>
			<div className="space-y-2">
				{config.exercises.map((exercise, i) => (
					<ExerciseItem
						key={exercise.title}
						exercise={exercise}
						isExpanded={expandedIndex === i}
						onToggle={() => setExpandedIndex(expandedIndex === i ? null : i)}
					/>
				))}
			</div>
		</div>
	);
}
