import { createFileRoute, Outlet, redirect } from "@tanstack/react-router";
import { authQueryOptions } from "../hooks/useAuth";
import { queryClient } from "../lib/query-client";

export const Route = createFileRoute("/app")({
	beforeLoad: async () => {
		if (import.meta.env.VITE_AUTH_MODE !== "live") return;
		try {
			const user = await queryClient.ensureQueryData(authQueryOptions);
			if (!user) {
				throw redirect({ to: "/signin" });
			}
		} catch (err) {
			if (err && typeof err === "object" && "to" in err) throw err;
			throw redirect({ to: "/signin" });
		}
	},
	component: AppLayout,
});

/**
 * The /app shell. Carries the auth guard for every child route -- home, the
 * practice stand, and the piece page -- so the guard is written once rather
 * than repeated per route.
 */
function AppLayout() {
	return <Outlet />;
}
