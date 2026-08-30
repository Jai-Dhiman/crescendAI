import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PieceSessionHistory } from "../components/PieceSessionHistory";
import { ScoreStand } from "../components/ScoreStand";
import { SessionTimelineStrip } from "../components/SessionTimelineStrip";
import { pieceHistoryQueryOptions } from "../hooks/useRepertoire";
import {
	PIECE_PAGE_ALL_MARKS,
	PIECE_PAGE_MARKS_SPAN_SECONDS,
} from "../lib/piece-page-fixtures";

export const Route = createFileRoute("/app/piece/$pieceId")({
	component: RouteComponent,
});

function RouteComponent() {
	const { pieceId } = Route.useParams();
	return <AppPiecePage pieceId={pieceId} />;
}

interface AppPiecePageProps {
	pieceId: string;
}

/**
 * The piece page -- "the score across time" (docs/apps/05-ui-system.md#4).
 *
 * The marks are FIXTURE data. Marks have no persisted store yet: they exist
 * only as DO state and WebSocket events within a live session, so there is
 * nothing to query for "every mark this piece has accumulated". #160's
 * success criterion is worded for exactly this ("marks aging across >= 2
 * sessions of fixture data"); the mark-persistence follow-up replaces this
 * one import with a query and changes nothing else on this surface.
 *
 * Session history and open drills, by contrast, are real.
 */
export function AppPiecePage({ pieceId }: AppPiecePageProps) {
	const navigate = useNavigate();
	const history = useQuery(pieceHistoryQueryOptions(pieceId));

	if (history.isError) {
		return (
			<div className="mx-auto w-full max-w-2xl px-6 py-10">
				<p className="text-body-sm text-danger">
					Couldn't load this piece. Your practice history is still there -- this
					is a connection problem.
				</p>
			</div>
		);
	}

	return (
		<div className="mx-auto flex w-full max-w-3xl flex-col gap-8 bg-surface-page px-6 py-10">
			<header className="flex flex-col gap-1">
				<h1 className="font-display text-heading text-ink-primary">
					{history.data?.title ?? ""}
				</h1>
				<span className="text-body-sm text-ink-secondary">
					{history.data?.composer ?? ""}
				</span>
			</header>

			{/*
			 * The score itself. It carries NO marks: Canvas A is lossy by
			 * design (docs/apps/05-ui-system.md) -- it can only place marks
			 * whose anchor is `bars` AND whose bars are on the rendered page,
			 * so it can never be the complete across-time view this page owes.
			 */}
			<section className="h-96 rounded-lg border border-border-subtle">
				<ScoreStand
					pieceId={pieceId}
					marks={[]}
					elapsedSeconds={0}
					isRecording={false}
				/>
			</section>

			{/*
			 * Canvas B, the total view: every mark this piece has accumulated,
			 * across every session, faded by lifecycle. Elapsed time is the one
			 * coordinate every anchor carries, which is what makes this canvas
			 * able to show marks the score overlay cannot place.
			 *
			 * durationSeconds spans the whole accumulated history rather than
			 * one session, so a mark late in a long session is not clamped on
			 * top of an early one.
			 */}
			<section className="flex flex-col gap-2">
				<h2 className="font-display text-body text-ink-secondary">
					Marks across sessions
				</h2>
				<SessionTimelineStrip
					durationSeconds={PIECE_PAGE_MARKS_SPAN_SECONDS}
					marks={PIECE_PAGE_ALL_MARKS}
				/>
			</section>

			<button
				type="button"
				onClick={() =>
					void navigate({ to: "/app/practice", search: { pieceId } })
				}
				className="self-start rounded-full bg-accent px-5 py-3 text-body-sm text-on-accent"
			>
				Practice this
			</button>

			<section className="flex flex-col gap-2">
				<h2 className="font-display text-body text-ink-secondary">Sessions</h2>
				<PieceSessionHistory sessions={history.data?.sessions ?? []} />
			</section>

			<section className="flex flex-col gap-2">
				<h2 className="font-display text-body text-ink-secondary">
					Open drills
				</h2>
				{history.data?.openDrills.length ? (
					<ul className="flex flex-col gap-1">
						{history.data.openDrills.map((drill) => (
							<li
								key={drill.id}
								className="flex items-baseline justify-between border-border-subtle border-b py-2 text-body-sm text-ink-primary"
							>
								<span>
									bars {drill.barsStart}-{drill.barsEnd}
								</span>
								<span className="text-body-xs text-ink-tertiary">
									{drill.attemptsCompleted} / {drill.requiredCorrect}
								</span>
							</li>
						))}
					</ul>
				) : (
					<p className="text-body-sm text-ink-tertiary">No open drills</p>
				)}
			</section>
		</div>
	);
}
