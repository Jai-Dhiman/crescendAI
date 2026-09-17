import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { PracticeMode } from "../components/PracticeMode";
import { usePracticeSession } from "../hooks/usePracticeSession";
import type { VerdictState } from "../lib/session-review";

export const Route = createFileRoute("/app/practice")({
	validateSearch: (search: Record<string, unknown>) => ({
		pieceId: typeof search.pieceId === "string" ? search.pieceId : null,
	}),
	component: RouteComponent,
});

function RouteComponent() {
	const { pieceId } = Route.useSearch();
	return <AppPractice pieceId={pieceId} />;
}

interface AppPracticeProps {
	/** From ?pieceId=. A search param, not a route param: a pieceless session
	 * is a permanent first-class state, and /app/practice/none would be a lie
	 * about the URL space. */
	pieceId: string | null;
}

/**
 * Score-first practice stand. Exported separately from the route so it can be
 * rendered in tests without a router, matching how routes/index.tsx's page is
 * reached in index.test.tsx.
 *
 * #160 closes #158's inbound dependency here: userPickedPieceId is the piece
 * the student chose on home, and confidentGuess is the live piece_identified
 * result. Both were hardcoded null before.
 */
export function AppPractice({ pieceId }: AppPracticeProps) {
	const [view, setView] = useState<"start" | "practice" | "review">("start");

	const practice = usePracticeSession({
		onSummary: () => setView("review"),
	});

	function handleStart() {
		setView("practice");
		void practice.start(undefined, pieceId ?? undefined);
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
					className="h-20 w-20 opacity-50"
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
				userPickedPieceId={pieceId}
				confidentGuess={practice.confidentGuess}
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
