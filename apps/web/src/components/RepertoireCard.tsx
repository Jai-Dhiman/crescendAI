import type { RepertoireEntry } from "../lib/api";
import { describeArc } from "../lib/learning-arc";

interface RepertoireCardProps {
	entry: RepertoireEntry;
	/** Injected clock, so the arc line is deterministic and timezone-safe. */
	now: Date;
	onOpen: (pieceId: string) => void;
}

/**
 * One piece on the home surface. A callback rather than a <Link> so this
 * renders and is testable without a router context; routes/app.index.tsx owns
 * navigation.
 *
 * openMarkCount renders only when non-null. There is no persisted mark store
 * yet, so the value is null and the count is simply absent -- the system falls
 * silent rather than showing a 0 a student would read as "nothing to work on".
 */
export function RepertoireCard({ entry, now, onOpen }: RepertoireCardProps) {
	return (
		<button
			type="button"
			onClick={() => onOpen(entry.pieceId)}
			// py-3 keeps the target well past the 32px minimum (WEB-A11Y-003).
			className="flex w-full flex-col items-start gap-1 rounded-lg border border-border-subtle bg-surface-raised px-4 py-3 text-left"
		>
			<span className="font-display text-body-md text-ink-primary">
				{entry.title}
			</span>
			<span className="text-body-sm text-ink-secondary">{entry.composer}</span>
			<span className="text-body-xs text-ink-tertiary">
				{describeArc({
					sessionCount: entry.sessionCount,
					lastPracticedAt: entry.lastPracticedAt,
					now,
				})}
			</span>
			{entry.openMarkCount !== null && (
				<span
					data-testid="open-mark-count"
					className="text-body-xs text-ink-tertiary"
				>
					{entry.openMarkCount} open marks
				</span>
			)}
		</button>
	);
}
