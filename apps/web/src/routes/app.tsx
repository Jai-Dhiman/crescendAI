import { createFileRoute, Outlet, redirect } from "@tanstack/react-router";
import { queryClient } from "../lib/query-client";
import { authQueryOptions } from "../hooks/useAuth";

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
	component: () => <Outlet />,
});
