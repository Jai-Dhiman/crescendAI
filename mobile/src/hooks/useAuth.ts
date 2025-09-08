import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "../stores";
import { apiClient } from "../services/api";
import { queryKeys } from "../services/queryClient";
import type { User } from "../types";

export function useAuth() {
	const queryClient = useQueryClient();
	const {
		user,
		tokens,
		isAuthenticated,
		isLoading,
		signIn,
		signOut: signOutStore,
		setLoading,
		updateUser,
		isTokenExpired,
	} = useAuthStore();

	// Google OAuth mutation
	const googleAuthMutation = useMutation({
		mutationFn: (token: string) => apiClient.authenticateWithGoogle(token),
		onMutate: () => {
			setLoading(true);
		},
		onSuccess: (response) => {
			const { user, tokens } = response.data;
			signIn(user, tokens);
		},
		onError: (error) => {
			console.error("Google auth failed:", error);
			setLoading(false);
		},
	});

	// Token refresh mutation
	const refreshTokenMutation = useMutation({
		mutationFn: (refreshToken: string) => apiClient.refreshToken(refreshToken),
		onSuccess: (response) => {
			const { tokens: newTokens } = response.data;
			useAuthStore.getState().setTokens(newTokens);
		},
		onError: () => {
			// If refresh fails, sign out
			signOut();
		},
	});

	// Sign out mutation
	const signOutMutation = useMutation({
		mutationFn: () => apiClient.signOut(),
		onSuccess: () => {
			signOutStore();
			queryClient.clear();
		},
		onError: () => {
			// Even if API call fails, sign out locally
			signOutStore();
			queryClient.clear();
		},
	});

	// Update profile mutation
	const updateProfileMutation = useMutation({
		mutationFn: (updates: Partial<User>) => {
			if (!user?.id) throw new Error("No user ID");
			return apiClient.updateUserProfile(user.id, updates);
		},
		onSuccess: (response) => {
			updateUser(response.data);
			queryClient.setQueryData(queryKeys.user(user!.id), response.data);
		},
	});

	// Auto-refresh token if needed
	const checkAndRefreshToken = async () => {
		if (tokens && isTokenExpired() && tokens.refreshToken) {
			try {
				await refreshTokenMutation.mutateAsync(tokens.refreshToken);
			} catch (error) {
				console.error("Token refresh failed:", error);
			}
		}
	};

	const signInWithGoogle = (token: string) => {
		return googleAuthMutation.mutateAsync(token);
	};

	const signOut = () => {
		signOutMutation.mutate();
	};

	const updateProfile = (updates: Partial<User>) => {
		return updateProfileMutation.mutateAsync(updates);
	};

	return {
		// State
		user,
		tokens,
		isAuthenticated,
		isLoading:
			isLoading || googleAuthMutation.isPending || signOutMutation.isPending,

		// Actions
		signInWithGoogle,
		signOut,
		updateProfile,
		checkAndRefreshToken,

		// Mutation states
		isSigningIn: googleAuthMutation.isPending,
		isSigningOut: signOutMutation.isPending,
		isUpdatingProfile: updateProfileMutation.isPending,

		// Errors
		signInError: googleAuthMutation.error,
		signOutError: signOutMutation.error,
		updateProfileError: updateProfileMutation.error,
	};
}
