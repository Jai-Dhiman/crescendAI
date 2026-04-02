import { queryOptions } from "@tanstack/react-query";
import type { AuthUser } from "../lib/api";
import { authClient } from "../lib/auth-client";

export const authQueryOptions = queryOptions({
	queryKey: ["auth", "me"] as const,
	queryFn: async (): Promise<AuthUser | null> => {
		const { data, error } = await authClient.getSession();
		if (error) {
			throw new Error(error.message);
		}
		if (!data) {
			return null;
		}
		return {
			studentId: data.user.id,
			email: data.user.email ?? null,
			displayName: data.user.name ?? null,
		};
	},
	retry: false,
	staleTime: 5 * 60 * 1000, // 5 minutes
});
