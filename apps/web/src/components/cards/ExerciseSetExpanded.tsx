import { useArtifactStore, type ExerciseStatus } from "../../stores/artifact";
import { handsLabel } from "../../lib/exercise-utils";
import { api } from "../../lib/api";
import type { ExerciseSetConfig } from "../../lib/types";

type Exercise = ExerciseSetConfig["exercises"][number];

interface ButtonProps {
	label: string;
	onClick?: () => void;
	disabled: boolean;
	className: string;
}

function buttonProps(
	status: ExerciseStatus,
	hasExerciseId: boolean,
	studentExerciseId: string | undefined,
	onStart: () => void,
	onComplete: () => void,
	onRetry: () => void,
): ButtonProps {
	const base = "text-body-sm px-4 py-2 rounded-lg border transition font-medium";
	const active = `${base} border-accent text-accent hover:bg-accent/10`;
	const disabled_cls = `${base} border-border text-text-tertiary opacity-50`;
	const error_cls = `${base} border-red-500 text-red-400 hover:bg-red-500/10`;
	const completed_cls = `${base} border-accent text-accent cursor-default`;

	switch (status) {
		case "idle":
			if (!hasExerciseId) {
				return { label: "Not yet saved", onClick: undefined, disabled: true, className: disabled_cls };
			}
			return { label: "Start", onClick: onStart, disabled: false, className: active };
		case "loading":
			return { label: "Starting...", onClick: undefined, disabled: true, className: disabled_cls };
		case "assigned":
			return { label: "Complete", onClick: onComplete, disabled: false, className: active };
		case "completing":
			return { label: "Completing...", onClick: undefined, disabled: true, className: disabled_cls };
		case "completed":
			return { label: "Completed", onClick: undefined, disabled: true, className: completed_cls };
		case "error":
			return {
				label: "Try again",
				onClick: onRetry,
				disabled: false,
				className: error_cls,
			};
	}
}

interface ExpandedExerciseItemProps {
	exercise: Exercise;
	artifactId: string;
}

function ExpandedExerciseItem({ exercise, artifactId }: ExpandedExerciseItemProps) {
	const exerciseState = useArtifactStore(
		(s) => s.states[artifactId]?.exerciseStates?.[exercise.exerciseId ?? ""],
	);
	const setExerciseStatus = useArtifactStore((s) => s.setExerciseStatus);

	const status: ExerciseStatus = exerciseState?.status ?? "idle";
	const studentExerciseId = exerciseState?.studentExerciseId;

	async function handleStart() {
		if (!exercise.exerciseId) return;
		setExerciseStatus(artifactId, exercise.exerciseId, "loading");
		try {
			const result = await api.exercises.assign({ exerciseId: exercise.exerciseId! });
			setExerciseStatus(artifactId, exercise.exerciseId, "assigned", result.id);
		} catch {
			setExerciseStatus(artifactId, exercise.exerciseId, "error");
		}
	}

	async function handleComplete() {
		if (!exercise.exerciseId || !studentExerciseId) return;
		setExerciseStatus(artifactId, exercise.exerciseId, "completing", studentExerciseId);
		try {
			await api.exercises.complete({ studentExerciseId });
			setExerciseStatus(artifactId, exercise.exerciseId, "completed", studentExerciseId);
		} catch {
			setExerciseStatus(artifactId, exercise.exerciseId, "error", studentExerciseId);
		}
	}

	function handleRetry() {
		if (studentExerciseId) {
			handleComplete();
		} else {
			handleStart();
		}
	}

	const btn = buttonProps(
		status,
		!!exercise.exerciseId,
		studentExerciseId,
		handleStart,
		handleComplete,
		handleRetry,
	);

	return (
		<div className="border border-border rounded-xl p-4 flex flex-col gap-3">
			<div className="flex items-start justify-between gap-3">
				<div className="flex flex-col gap-1 min-w-0">
					<span className="text-body-md font-medium text-text-primary">{exercise.title}</span>
					<div className="flex items-center gap-2 flex-wrap">
						{exercise.hands && (
							<span className="text-body-sm text-text-secondary bg-surface-elevated px-2 py-0.5 rounded">
								{handsLabel(exercise.hands)}
							</span>
						)}
						<span className="text-body-sm text-text-tertiary">{exercise.focusDimension}</span>
					</div>
				</div>
				<button
					type="button"
					className={btn.className}
					onClick={btn.onClick}
					disabled={btn.disabled}
				>
					{btn.label}
				</button>
			</div>
			<p className="text-body-sm text-text-secondary">{exercise.instruction}</p>
		</div>
	);
}

interface ExerciseSetExpandedProps {
	config: ExerciseSetConfig;
	artifactId: string;
}

export function ExerciseSetExpanded({ config, artifactId }: ExerciseSetExpandedProps) {
	const exerciseStates = useArtifactStore((s) => s.states[artifactId]?.exerciseStates);

	const savedExercises = config.exercises.filter((e) => !!e.exerciseId);
	const completedCount = savedExercises.filter(
		(e) => exerciseStates?.[e.exerciseId!]?.status === "completed",
	).length;
	const totalCount = savedExercises.length;

	const progressPercent = totalCount > 0 ? (completedCount / totalCount) * 100 : 0;

	return (
		<div className="flex flex-col gap-4">
			<div className="flex flex-col gap-1">
				<span className="text-body-lg font-semibold text-text-primary">{config.targetSkill}</span>
				<span className="text-body-sm text-text-secondary">{config.sourcePassage}</span>
			</div>

			<div className="flex flex-col gap-3">
				{config.exercises.map((exercise, index) => (
					<ExpandedExerciseItem
						key={exercise.exerciseId ?? index}
						exercise={exercise}
						artifactId={artifactId}
					/>
				))}
			</div>

			<div className="mt-6 pt-4 border-t border-border flex flex-col gap-2">
				<span className="text-body-sm text-text-secondary">
					{completedCount} of {totalCount} completed
				</span>
				<div className="w-full h-1.5 bg-surface-elevated rounded-full overflow-hidden">
					<div
						className="h-full bg-accent rounded-full transition-all duration-300"
						style={{ width: `${progressPercent}%` }}
					/>
				</div>
			</div>
		</div>
	);
}
