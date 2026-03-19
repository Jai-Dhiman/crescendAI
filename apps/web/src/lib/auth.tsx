import {
	createContext,
	type ReactNode,
	useCallback,
	useContext,
	useEffect,
	useState,
} from "react";
import { ApiError, type AuthUser, api } from "./api";

interface AuthContextValue {
	user: AuthUser | null;
	isLoading: boolean;
	isAuthenticated: boolean;
	setUser: (user: AuthUser | null) => void;
	signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
	const [user, setUser] = useState<AuthUser | null>(null);
	const [isLoading, setIsLoading] = useState(true);

	useEffect(() => {
		// In waitlist mode, nobody can authenticate -- skip the API call
		// to avoid 401 Sentry noise on every visitor page load.
		if (import.meta.env.VITE_AUTH_MODE !== "live") {
			setIsLoading(false);
			return;
		}

		api.auth
			.me()
			.then(setUser)
			.catch((err) => {
				if (err instanceof ApiError && err.status === 401) {
					setUser(null);
				} else {
					console.error("Auth check failed:", err);
					setUser(null);
				}
			})
			.finally(() => setIsLoading(false));
	}, []);

	const signOut = useCallback(async () => {
		try {
			await api.auth.signout();
		} catch (err) {
			console.error("Signout API call failed:", err);
		}
		setUser(null);
	}, []);

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
