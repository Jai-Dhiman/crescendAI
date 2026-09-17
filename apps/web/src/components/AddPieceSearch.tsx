import { useState } from "react";
import type { PieceSummary } from "../lib/piece-search";
import { searchPieces } from "../lib/piece-search";

interface AddPieceSearchProps {
	catalog: readonly PieceSummary[];
	onPick: (pieceId: string) => void;
}

/**
 * Score-library search for the home surface. The query string is the only
 * client state this feature has; the catalog itself is server state and stays
 * in react-query (WEB-QUERY-002).
 */
export function AddPieceSearch({ catalog, onPick }: AddPieceSearchProps) {
	const [query, setQuery] = useState("");

	// Computed during render, not mirrored into state by an effect
	// (WEB-STATE-001), and deliberately not memoized: nothing downstream takes
	// it as a dependency and the catalog slice is small (WEB-STATE-002).
	const results = searchPieces(catalog, query);

	return (
		<div className="flex flex-col gap-2">
			<input
				type="search"
				value={query}
				onChange={(e) => setQuery(e.target.value)}
				placeholder="Search the score library"
				className="w-full rounded-lg border border-border-subtle bg-surface-raised px-4 py-3 text-body-sm text-ink-primary"
			/>
			{results.length > 0 && (
				<ul className="flex flex-col gap-1">
					{results.map((piece) => (
						<li key={piece.pieceId}>
							<button
								type="button"
								onClick={() => onPick(piece.pieceId)}
								className="flex w-full flex-col items-start rounded-lg px-4 py-2 text-left"
							>
								<span className="text-body-sm text-ink-primary">
									{piece.title}
								</span>
								<span className="text-body-xs text-ink-tertiary">
									{piece.composer}
								</span>
							</button>
						</li>
					))}
				</ul>
			)}
		</div>
	);
}
