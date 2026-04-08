export interface TitleFields {
	opusNumber: number | null;
	pieceNumber: number | null;
	catalogueType: string | null;
}

/**
 * Parses an ASAP dataset title string into structured catalog fields.
 * Titles follow a regular naming convention derived from directory paths:
 *   "Waltz Op. 64 No. 2"   → opus=64, piece=2, type="op"
 *   "WTC I - Prelude - 1"  → opus=null, piece=1, type="wtc"
 *   "Ballades No. 1"       → opus=null, piece=1, type=null
 *   "Arabesques"           → all null
 */
export function parseTitleFields(title: string): TitleFields {
	const opusMatch = title.match(/Op\.\s*(\d+)/i);
	const numberMatch = title.match(/No\.\s*(\d+)/i);
	const isWtc = /WTC/i.test(title);

	const opusNumber = opusMatch ? parseInt(opusMatch[1], 10) : null;
	let pieceNumber = numberMatch ? parseInt(numberMatch[1], 10) : null;

	// WTC titles use trailing "- N" format instead of "No. N"
	if (isWtc && pieceNumber === null) {
		const trailingNum = title.match(/[-\u2013]\s*(\d+)\s*$/);
		if (trailingNum) {
			pieceNumber = parseInt(trailingNum[1], 10);
		}
	}

	let catalogueType: string | null = null;
	if (isWtc) {
		catalogueType = "wtc";
	} else if (opusNumber !== null) {
		catalogueType = "op";
	}

	return { opusNumber, pieceNumber, catalogueType };
}
