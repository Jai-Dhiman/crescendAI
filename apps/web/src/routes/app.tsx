import { createFileRoute, redirect } from "@tanstack/react-router";
import { useState } from "react";
import { PracticeMode } from "../components/PracticeMode";
import { authQueryOptions } from "../hooks/useAuth";
import { usePracticeSession } from "../hooks/usePracticeSession";
import { queryClient } from "../lib/query-client";
import type { VerdictState } from "../lib/session-review";

export const Route = createFileRoute("/app")({
	beforeLoad: async () => {
		if (import.meta.env.VITE_AUTH_MODE !== "live") return;
		try {
			const user = await queryClient.ensureQueryData(authQueryOptions);
			if (!user) {
				throw redirect({ to: "/signin" });
			}
		} catch (err) {
			if (err && typeof err === "object" && "to" in err) throw err;
			throw redirect({ to: "/signin" });
		}
	},
	component: AppPractice,
});

/**
 * Score-first practice stand: /app's production surface post-#154. Chat is
 * dead (AppChat deleted in #164); this mounts PracticeMode directly and
 * drives it from the real usePracticeSession hook (same WS/session-start
 * wiring AppChat used), transitioning to the review phase once the session's
 * final summary event arrives. No piece picker / home surface yet -- that is
 * #160.
 */
function AppPractice() {
	const [view, setView] = useState<"start" | "practice" | "review">("start");

	const practice = usePracticeSession({
		onSummary: () => setView("review"),
	});

	function handleStart() {
		setView("practice");
		practice.start();
	}

	function handleStop() {
		practice.stop();
	}

	function handleExitReview() {
		setView("start");
	}

	if (view === "start" && practice.state === "idle") {
		return (
			<div className="flex h-dvh flex-col items-center justify-center gap-6 bg-surface-page px-6">
				<img
					src="/icon_nobackground.png"
					alt=""
					className="w-20 h-20 opacity-50"
				/>
				<button
					type="button"
					onClick={handleStart}
					className="rounded-full bg-accent px-6 py-3 text-body-sm text-on-accent"
				>
					Start practicing
				</button>
			</div>
		);
	}

	const verdict: VerdictState = practice.verdict
		? { status: "ready", text: practice.verdict }
		: practice.state === "error"
			? { status: "failed" }
			: { status: "loading" };

	return (
		<div className="fixed inset-0 z-50 bg-surface-page">
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={practice.marks}
				elapsedSeconds={practice.elapsedSeconds}
				isPlaying={practice.isPlaying}
				isRecording={practice.state === "recording"}
				onStop={handleStop}
				phase={view === "review" ? "review" : "practice"}
				sessionId={practice.sessionId ?? undefined}
				verdict={verdict}
				carryForward={
					practice.carryForward ? { text: practice.carryForward } : null
				}
				wsGaps={[]}
				noCommentZones={practice.noCommentZones}
				onExitReview={handleExitReview}
			/>
		</div>
	);
}
