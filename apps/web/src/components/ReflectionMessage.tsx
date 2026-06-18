import { useState } from "react";
import { api } from "../lib/api";
import type { ExerciseSetConfig, PendingExerciseConfig } from "../lib/types";
import { Artifact } from "./Artifact";

interface ReflectionMessageProps {
	sessionId: string;
	reflectionText: string;
	pendingConfig: PendingExerciseConfig;
	onDecline: (focusDimension: string) => void;
}

type ConfirmState = "idle" | "loading" | "revealed" | "error";

export function ReflectionMessage({
	sessionId,
	reflectionText,
	pendingConfig,
	onDecline,
}: ReflectionMessageProps) {
	const [confirmState, setConfirmState] = useState<ConfirmState>("idle");
	const [revealedConfig, setRevealedConfig] =
		useState<ExerciseSetConfig | null>(null);
	const artifactId = `pending-${pendingConfig.exerciseId}`;

	async function handleConfirm() {
		if (confirmState !== "idle") return;
		setConfirmState("loading");
		try {
			const config = await api.exercises.assignPending({
				sessionId,
				exerciseId: pendingConfig.exerciseId,
			});
			setRevealedConfig(config);
			setConfirmState("revealed");
		} catch {
			setConfirmState("error");
		}
	}

	function handleDecline() {
		onDecline(pendingConfig.focusDimension);
	}

	return (
		<div className="flex flex-col gap-3 mt-1">
			<p className="text-body-sm text-text-primary leading-relaxed" data-testid="synthesis-headline">
				{reflectionText}
			</p>

			{confirmState !== "revealed" && (
				<div className="flex gap-2">
					<button
						type="button"
						onClick={handleConfirm}
						disabled={confirmState === "loading"}
						data-testid="confirm-exercise-button"
						className="text-body-xs px-3 py-1.5 rounded-lg border border-accent text-accent hover:bg-accent/10 transition disabled:opacity-50"
					>
						{confirmState === "loading" ? "Adding..." : "Confirm"}
					</button>
					<button
						type="button"
						onClick={handleDecline}
						disabled={confirmState === "loading"}
						className="text-body-xs px-3 py-1.5 rounded-lg border border-border text-text-tertiary hover:text-cream hover:border-accent transition disabled:opacity-50"
					>
						Not now
					</button>
				</div>
			)}

			{confirmState === "error" && (
				<p className="text-body-xs text-red-400">
					Failed to load exercise. Try again.
				</p>
			)}

			{confirmState === "revealed" && revealedConfig && (
				<Artifact
					artifactId={artifactId}
					component={{ type: "exercise_set", config: revealedConfig }}
				/>
			)}
		</div>
	);
}
