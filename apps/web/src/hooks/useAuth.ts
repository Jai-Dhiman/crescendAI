import { queryOptions } from "@tanstack/react-query";
import { ApiError, api } from "../lib/api";

export const authQueryOptions = queryOptions({
	queryKey: ["auth", "me"] as const,
	queryFn: async () => {
		try {
			return await api.auth.me();
		} catch (err) {
			if (err instanceof ApiError && err.status === 401) {
				return null;
			}
			throw err;
		}
	},
	retry: false,
	staleTime: 5 * 60 * 1000, // 5 minutes
});
