/**
 * Structurally the subset of a GET /api/scores row this feature reads.
 * Deliberately not the full `pieces` table type: the search surface has no
 * use for bar counts or pitch ranges, and widening it would couple every
 * caller to catalog schema changes.
 */
export interface PieceSummary {
	readonly pieceId: string;
	readonly composer: string;
	readonly title: string;
}

const DEFAULT_LIMIT = 20;

/**
 * NFD + combining-mark strip, so "faure" matches "Fauré". The catalog is
 * classical repertoire; accented composer names are the rule, not the edge
 * case, and a student typing on a US keyboard cannot produce them.
 */
function normalize(value: string): string {
	return value.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

/**
 * Empty query returns nothing rather than everything: this drives an
 * add-piece surface where showing the entire catalog on focus is noise, and
 * "no query means no results" keeps the caller from needing its own guard.
 */
export function searchPieces(
	pieces: readonly PieceSummary[],
	query: string,
	limit: number = DEFAULT_LIMIT,
): PieceSummary[] {
	const needle = normalize(query.trim());
	if (needle === "") return [];

	const scored: { piece: PieceSummary; rank: number }[] = [];
	for (const piece of pieces) {
		const composer = normalize(piece.composer);
		const title = normalize(piece.title);
		// 0 = prefix match (strongest), 1 = substring, absent = no match.
		let rank: number | null = null;
		if (composer.startsWith(needle) || title.startsWith(needle)) {
			rank = 0;
		} else if (composer.includes(needle) || title.includes(needle)) {
			rank = 1;
		}
		if (rank !== null) scored.push({ piece, rank });
	}

	// Stable sort by rank alone: within a rank the catalog's own ordering
	// (composer, then title -- services/scores.ts orders it that way) is
	// already the order a student expects.
	scored.sort((a, b) => a.rank - b.rank);
	return scored.slice(0, limit).map((s) => s.piece);
}
