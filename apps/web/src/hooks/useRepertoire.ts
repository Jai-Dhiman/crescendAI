import { queryOptions } from "@tanstack/react-query";
import { api } from "../lib/api";

export const repertoireQueryOptions = queryOptions({
	queryKey: ["repertoire", "list"] as const,
	queryFn: () => api.repertoire.list(),
});

export const pieceHistoryQueryOptions = (pieceId: string) =>
	queryOptions({
		queryKey: ["repertoire", "piece", pieceId] as const,
		queryFn: () => api.repertoire.history(pieceId),
	});
