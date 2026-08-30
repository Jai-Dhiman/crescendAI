import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { AddPieceSearch } from "../components/AddPieceSearch";
import { RepertoireCard } from "../components/RepertoireCard";
import { pieceCatalogQueryOptions } from "../hooks/usePieceCatalog";
import { repertoireQueryOptions } from "../hooks/useRepertoire";

export const Route = createFileRoute("/app/")({
	component: AppHome,
});

/**
 * Home -- "your music" (docs/apps/05-ui-system.md#1). Exported so it renders
 * in tests without a router, matching routes/index.tsx.
 *
 * Query keys come from the shared queryOptions modules, never a literal here
 * (WEB-QUERY-001).
 */
export function AppHome() {
	const navigate = useNavigate();
	const repertoire = useQuery(repertoireQueryOptions);
	const catalog = useQuery(pieceCatalogQueryOptions);

	const now = new Date();

	function openPiece(pieceId: string) {
		void navigate({ to: "/app/piece/$pieceId", params: { pieceId } });
	}

	function practise(pieceId: string | null) {
		void navigate({ to: "/app/practice", search: { pieceId } });
	}

	return (
		<div className="mx-auto flex w-full max-w-2xl flex-col gap-8 bg-surface-page px-6 py-10">
			<section className="flex flex-col gap-3">
				<h1 className="font-display text-heading text-ink-primary">
					Your music
				</h1>
				{/* An empty repertoire and a failed fetch must not look alike
				    (WEB-ERR-001). */}
				{repertoire.isError && (
					<p className="text-body-sm text-danger">
						Couldn't load your pieces. Your practice history is still there --
						this is a connection problem.
					</p>
				)}
				{repertoire.data?.map((entry) => (
					<RepertoireCard
						key={entry.pieceId}
						entry={entry}
						now={now}
						onOpen={openPiece}
					/>
				))}
			</section>

			<section className="flex flex-col gap-3">
				<h2 className="font-display text-body text-ink-secondary">
					Add a piece
				</h2>
				{catalog.isError ? (
					<p className="text-body-sm text-danger">
						Couldn't load the score library.
					</p>
				) : (
					<AddPieceSearch
						catalog={catalog.data ?? []}
						onPick={(pieceId) => practise(pieceId)}
					/>
				)}
			</section>

			<button
				type="button"
				onClick={() => practise(null)}
				className="self-start rounded-full border border-border-subtle px-5 py-3 text-body-sm text-ink-secondary"
			>
				Just play
			</button>
		</div>
	);
}
