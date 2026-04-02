import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, type ReactNode, useCallback, useContext } from "react";
import { authQueryOptions } from "../hooks/useAuth";
import type { AuthUser } from "./api";
import { authClient } from "./auth-client";

interface AuthContextValue {
	user: AuthUser | null;
	isLoading: boolean;
	isAuthenticated: boolean;
	setUser: (user: AuthUser | null) => void;
	signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
	const isLive = import.meta.env.VITE_AUTH_MODE === "live";
	const queryClient = useQueryClient();

	const query = useQuery({
		...authQueryOptions,
		enabled: isLive,
	});

	// When auth is disabled (waitlist mode), skip query entirely
	const user = isLive ? (query.data ?? null) : null;
	const isLoading = isLive ? query.isLoading : false;

	const setUser = useCallback(
		(u: AuthUser | null) => {
			queryClient.setQueryData(authQueryOptions.queryKey, u);
		},
		[queryClient],
	);

	const signOut = useCallback(async () => {
		await authClient.signOut();
		queryClient.setQueryData(authQueryOptions.queryKey, null);
	}, [queryClient]);

	return (
		<AuthContext
			value={{
				user,
				isLoading,
				isAuthenticated: user !== null,
				setUser,
				signOut,
			}}
		>
			{children}
		</AuthContext>
	);
}

export function useAuth(): AuthContextValue {
	const ctx = useContext(AuthContext);
	if (!ctx) {
		throw new Error("useAuth must be used within an AuthProvider");
	}
	return ctx;
}
