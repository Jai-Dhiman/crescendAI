/** One joined sessions-to-pieces row. The input to buildRepertoire. */
export interface RepertoireRow {
	pieceId: string;
	composer: string;
	title: string;
	startedAt: Date;
}

export interface RepertoireEntry {
	pieceId: string;
	composer: string;
	title: string;
	sessionCount: number;
	/** ISO 8601. The most recent session's startedAt. */
	lastPracticedAt: string;
	/**
	 * null until a persisted mark store exists -- never 0. A 0 here reads to a
	 * student as "nothing to work on", which is a claim this system cannot
	 * make yet; null means "not known" and the card omits the count entirely.
	 * The mark-persistence follow-up changes this line and no contract.
	 */
	openMarkCount: number | null;
}

/**
 * The grouping half of the repertoire query, kept pure and separate from the
 * database call. Two reasons, and the second is the real one:
 *
 * 1. It is the deep half. The query is a join; the policy -- one entry per
 *    piece, count the sessions, take the latest, order by recency -- is what
 *    a caller actually depends on.
 * 2. This codebase's route tests have no database. They stub services with
 *    vi.spyOn or hand-build a db.query.X.findFirst object; neither technique
 *    can stub a select().from().leftJoin() chain. Aggregation written inline
 *    would therefore be verifiable only against a live Postgres. As a
 *    function over rows it is testable with real inputs and no mocks.
 */
export function buildRepertoire(
	rows: readonly RepertoireRow[],
): RepertoireEntry[] {
	const byPiece = new Map<string, RepertoireEntry>();

	for (const row of rows) {
		const existing = byPiece.get(row.pieceId);
		if (!existing) {
			byPiece.set(row.pieceId, {
				pieceId: row.pieceId,
				composer: row.composer,
				title: row.title,
				lastPracticedAt: row.startedAt.toISOString(),
				sessionCount: 1,
				openMarkCount: null,
			});
			continue;
		}
		existing.sessionCount += 1;
		// Compare explicitly rather than trusting the query's ORDER BY: the
		// ordering guarantee would then live in two places and could drift.
		if (row.startedAt.toISOString() > existing.lastPracticedAt) {
			existing.lastPracticedAt = row.startedAt.toISOString();
		}
	}

	return [...byPiece.values()].sort((a, b) =>
		b.lastPracticedAt.localeCompare(a.lastPracticedAt),
	);
}
