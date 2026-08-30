import { queryOptions } from "@tanstack/react-query";
import { api } from "../lib/api";

// The catalog is codegen-seeded and effectively static within a session, so
// it is fetched once and reused by the add-piece search.
export const pieceCatalogQueryOptions = queryOptions({
	queryKey: ["scores", "catalog"] as const,
	queryFn: () => api.scores.list(),
	staleTime: Number.POSITIVE_INFINITY,
});
