import {
	createFileRoute,
	Outlet,
	redirect,
	useRouterState,
} from "@tanstack/react-router";
import AppChat from "../components/AppChat";
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

function AppLayout() {
	const isChatsRoute = useRouterState({
		select: (s) => s.matches.some((m) => m.routeId === "/app/chats"),
	});

	if (isChatsRoute) return <Outlet />;
	return <AppChat />;
}
