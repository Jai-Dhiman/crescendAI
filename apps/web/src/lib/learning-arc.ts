export interface LearningArcInput {
	readonly sessionCount: number;
	/** ISO 8601, or null for a piece with no sessions. */
	readonly lastPracticedAt: string | null;
	/**
	 * Injected rather than read from the system clock. Every relative branch
	 * below is then testable without fake timers, and the function cannot
	 * silently change behaviour with the machine's timezone.
	 */
	readonly now: Date;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

/** Whole days between two instants, counted in UTC calendar days. */
function daysBetween(from: Date, to: Date): number {
	const fromDay = Date.UTC(
		from.getUTCFullYear(),
		from.getUTCMonth(),
		from.getUTCDate(),
	);
	const toDay = Date.UTC(
		to.getUTCFullYear(),
		to.getUTCMonth(),
		to.getUTCDate(),
	);
	return Math.round((toDay - fromDay) / MS_PER_DAY);
}

function describeWhen(last: Date, now: Date): string {
	const days = daysBetween(last, now);
	if (days <= 0) return "today";
	if (days === 1) return "yesterday";
	if (days < 7) {
		return last.toLocaleDateString("en-GB", {
			weekday: "long",
			timeZone: "UTC",
		});
	}
	return last.toLocaleDateString("en-GB", {
		day: "numeric",
		month: "short",
		timeZone: "UTC",
	});
}

/**
 * The repertoire card's single line of history. Extracted from the card so
 * the card carries no date arithmetic and every branch here is testable
 * without rendering anything.
 */
export function describeArc(input: LearningArcInput): string {
	const { sessionCount, lastPracticedAt, now } = input;
	if (sessionCount === 0 || lastPracticedAt === null) {
		return "Not practiced yet";
	}
	const noun = sessionCount === 1 ? "session" : "sessions";
	const when = describeWhen(new Date(lastPracticedAt), now);
	return `${sessionCount} ${noun} · last practiced ${when}`;
}
