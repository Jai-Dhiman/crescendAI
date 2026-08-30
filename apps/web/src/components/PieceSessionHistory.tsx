import type { PieceSessionSummary } from "../lib/api";

interface PieceSessionHistoryProps {
	sessions: readonly PieceSessionSummary[];
}

/**
 * UTC formatting throughout: session timestamps are stored UTC and this repo's
 * conventions avoid local-clock assumptions, which have caused drift before.
 */
function formatDate(iso: string): string {
	return new Date(iso).toLocaleDateString("en-GB", {
		day: "numeric",
		month: "short",
		year: "numeric",
		timeZone: "UTC",
	});
}

/** null endedAt means the session never ended cleanly -- no duration to show. */
function formatDuration(
	startedAt: string,
	endedAt: string | null,
): string | null {
	if (endedAt === null) return null;
	const minutes = Math.round(
		(Date.parse(endedAt) - Date.parse(startedAt)) / 60000,
	);
	return `${minutes} min`;
}

export function PieceSessionHistory({ sessions }: PieceSessionHistoryProps) {
	if (sessions.length === 0) {
		return <p className="text-body-sm text-ink-tertiary">No sessions yet</p>;
	}

	return (
		<ul className="flex flex-col gap-1">
			{sessions.map((session) => {
				const duration = formatDuration(session.startedAt, session.endedAt);
				return (
					<li
						key={session.sessionId}
						className="flex items-baseline justify-between border-b border-border-subtle py-2"
					>
						<span className="text-body-sm text-ink-primary">
							{formatDate(session.startedAt)}
						</span>
						{duration && (
							<span className="text-body-xs text-ink-tertiary">{duration}</span>
						)}
					</li>
				);
			})}
		</ul>
	);
}
