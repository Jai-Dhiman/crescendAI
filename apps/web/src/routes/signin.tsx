import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

const isWaitlistMode = import.meta.env.VITE_AUTH_MODE !== "live";

export const Route = createFileRoute("/signin")({
	component: isWaitlistMode ? WaitlistPage : SignInPage,
});

function WaitlistPage() {
	const [email, setEmail] = useState("");
	const [context, setContext] = useState("");
	const [honeypot, setHoneypot] = useState("");
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [submitted, setSubmitted] = useState(false);

	const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (honeypot) {
			setSubmitted(true);
			return;
		}
		setLoading(true);
		setError(null);
		try {
			await api.waitlist.join(email, context || undefined);
			setSubmitted(true);
		} catch {
			setError("Something went wrong. Please try again.");
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="relative h-screen w-full overflow-hidden">
			<img
				src="/Image5.jpg"
				alt="Hands playing piano in warm light"
				className="absolute inset-0 w-full h-full object-cover"
			/>

			<div
				className="absolute inset-0"
				style={{
					background:
						"radial-gradient(ellipse at center, rgba(45,41,38,0.4) 0%, rgba(45,41,38,0.85) 100%)",
				}}
			/>

			<div className="relative z-10 flex items-center justify-center h-full px-6">
				<div
					className="w-full max-w-sm bg-surface/80 backdrop-blur-xl border border-border px-8 py-14 text-center rounded-2xl"
					style={{
						animation:
							"fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both",
					}}
				>
					<h1 className="font-display text-display-sm text-cream">
						crescend
					</h1>

					<p className="mt-3 text-body-md text-text-secondary">
						A teacher for every pianist.
					</p>

					{submitted ? (
						<div
							style={{
								animation:
									"fade-in-up 500ms cubic-bezier(0.16, 1, 0.3, 1) both",
							}}
						>
							<p className="mt-8 font-display text-body-lg text-cream font-medium">
								You're on the list.
							</p>
							<p className="mt-2 text-body-sm text-text-secondary">
								We'll reach out when your spot is ready.
							</p>
							<Link
								to="/"
								className="mt-6 inline-block text-body-xs text-text-tertiary underline hover:text-text-secondary transition"
							>
								Back to homepage
							</Link>
						</div>
					) : (
						<form onSubmit={handleSubmit} className="mt-8">
							<p className="text-body-sm text-text-secondary mb-6">
								We're building something new. Join the beta
								waitlist to be first in.
							</p>

							<input
								type="email"
								required
								value={email}
								onChange={(e) => setEmail(e.target.value)}
								placeholder="Your email"
								className="w-full bg-surface-2 text-cream border border-border rounded-lg px-4 py-3 text-body-sm placeholder:text-text-tertiary focus:outline-none focus:border-accent transition"
							/>

							{/* Honeypot */}
							<input
								name="name"
								value={honeypot}
								onChange={(e) => setHoneypot(e.target.value)}
								style={{ position: "absolute", left: "-9999px" }}
								tabIndex={-1}
								autoComplete="off"
							/>

							<label className="block mt-4 text-left">
								<span className="text-body-xs text-text-tertiary">
									What do you play or practice?
								</span>
								<textarea
									value={context}
									onChange={(e) => setContext(e.target.value)}
									rows={2}
									placeholder="e.g., Working through Chopin Nocturnes, intermediate level"
									className="mt-1.5 w-full bg-surface-2 text-cream border border-border rounded-lg px-4 py-3 text-body-sm placeholder:text-text-tertiary focus:outline-none focus:border-accent transition resize-none"
								/>
							</label>

							{error && (
								<p className="mt-4 text-body-sm text-red-400">
									{error}
								</p>
							)}

							<button
								type="submit"
								disabled={loading}
								className="mt-6 w-full bg-accent text-cream px-6 py-3 text-body-sm font-medium rounded-lg hover:brightness-110 transition disabled:opacity-50"
							>
								{loading ? "Joining..." : "Join the Waitlist"}
							</button>

							<p className="mt-6 text-body-xs text-text-tertiary">
								We'll only email you about beta access.
							</p>
						</form>
					)}
				</div>
			</div>
		</div>
	);
}

let gsiInitialized = false;

const APPLE_CLIENT_ID = "ai.crescend.web";
const REDIRECT_URI = import.meta.env.PROD
	? "https://crescend.ai/signin"
	: "http://localhost:3000/signin";

function SignInPage() {
	const navigate = useNavigate();
	const { setUser, isAuthenticated, isLoading: authLoading } = useAuth();
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const googleButtonRef = useRef<HTMLDivElement>(null);

	const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

	useEffect(() => {
		if (!authLoading && isAuthenticated) {
			navigate({ to: "/app" });
		}
	}, [authLoading, isAuthenticated, navigate]);

	const handleGoogleSignIn = useCallback(
		async (response: google.accounts.id.CredentialResponse) => {
			setLoading(true);
			setError(null);
			try {
				const result = await api.auth.google(response.credential);
				setUser({
					student_id: result.student_id,
					email: result.email ?? null,
					display_name: result.display_name ?? null,
				});
				navigate({ to: "/app" });
			} catch (err) {
				console.error("Google sign in failed:", err);
				setError("Sign in failed. Please try again.");
			} finally {
				setLoading(false);
			}
		},
		[setUser, navigate],
	);

	// Use a ref so the Google SDK callback always calls the latest version
	const googleCallbackRef = useRef(handleGoogleSignIn);
	googleCallbackRef.current = handleGoogleSignIn;

	useEffect(() => {
		if (!GOOGLE_CLIENT_ID) return;

		function renderButton() {
			if (googleButtonRef.current) {
				google.accounts.id.renderButton(googleButtonRef.current, {
					type: "standard",
					theme: "outline",
					size: "large",
					width: googleButtonRef.current.parentElement?.offsetWidth ?? 300,
					text: "signin_with",
				});
			}
		}

		function initGsi() {
			if (!gsiInitialized) {
				google.accounts.id.initialize({
					client_id: GOOGLE_CLIENT_ID,
					callback: (resp: google.accounts.id.CredentialResponse) =>
						googleCallbackRef.current(resp),
				});
				gsiInitialized = true;
			}
			renderButton();
		}

		// If the GSI script is already loaded (e.g. StrictMode re-mount), just init
		if (typeof google !== "undefined" && google.accounts?.id) {
			initGsi();
			return;
		}

		// Avoid appending duplicate script tags (StrictMode double-mount)
		const existing = document.querySelector(
			'script[src="https://accounts.google.com/gsi/client"]',
		);
		if (existing) {
			existing.addEventListener("load", initGsi);
			return;
		}

		const script = document.createElement("script");
		script.src = "https://accounts.google.com/gsi/client";
		script.async = true;
		script.onload = initGsi;
		document.head.appendChild(script);
	}, [GOOGLE_CLIENT_ID]);

	useEffect(() => {
		if (window.AppleID) {
			window.AppleID.auth.init({
				clientId: APPLE_CLIENT_ID,
				scope: "name email",
				redirectURI: REDIRECT_URI,
				usePopup: true,
			});
		}
	}, []);

	async function handleSignIn() {
		if (!window.AppleID) {
			setError("Apple Sign In not available. Please try again.");
			return;
		}

		setLoading(true);
		setError(null);

		try {
			const appleResponse = await window.AppleID.auth.signIn();
			const idToken = appleResponse.authorization.id_token;

			// Decode the JWT to extract the subject (Apple user ID)
			const base64 = idToken
				.split(".")[1]
				.replace(/-/g, "+")
				.replace(/_/g, "/");
			const payload = JSON.parse(atob(base64));
			const userId = payload.sub;

			const email = appleResponse.user?.email ?? undefined;
			const firstName = appleResponse.user?.name?.firstName;
			const lastName = appleResponse.user?.name?.lastName;
			const displayName = firstName
				? [firstName, lastName].filter(Boolean).join(" ")
				: undefined;

			const result = await api.auth.apple(idToken, userId, email, displayName);

			setUser({
				student_id: result.student_id,
				email: result.email,
				display_name: result.display_name,
			});

			navigate({ to: "/app" });
		} catch (err) {
			if (
				err instanceof Error &&
				err.message.includes("popup_closed_by_user")
			) {
				return;
			}
			console.error("Sign in failed:", err);
			setError("Sign in failed. Please try again.");
		} finally {
			setLoading(false);
		}
	}

	return (
		<div className="relative h-screen w-full overflow-hidden">
			<img
				src="/Image5.jpg"
				alt="Hands playing piano in warm light"
				className="absolute inset-0 w-full h-full object-cover"
			/>

			<div
				className="absolute inset-0"
				style={{
					background:
						"radial-gradient(ellipse at center, rgba(45,41,38,0.4) 0%, rgba(45,41,38,0.85) 100%)",
				}}
			/>

			<div className="relative z-10 flex items-center justify-center h-full px-6">
				<div
					className="w-full max-w-sm bg-surface/80 backdrop-blur-xl border border-border px-8 py-14 text-center rounded-2xl"
					style={{
						animation: "fade-in-up 600ms cubic-bezier(0.16, 1, 0.3, 1) both",
					}}
				>
					<h1 className="font-display text-display-sm text-cream">crescend</h1>

					<p className="mt-3 text-body-md text-text-secondary">
						A teacher for every pianist.
					</p>

					{error && <p className="mt-4 text-body-sm text-red-400">{error}</p>}

					{GOOGLE_CLIENT_ID && (
						<div className="relative mt-8 w-full h-[42px]">
							{/* Custom styled button (visible) */}
							<div className="absolute inset-0 bg-white text-black px-6 py-3 text-body-sm font-medium flex items-center justify-center gap-3 rounded-lg pointer-events-none">
								<svg className="w-5 h-5" viewBox="0 0 24 24" role="img" aria-label="Google logo">
									<path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4" />
									<path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
									<path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
									<path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
								</svg>
								{loading ? "Signing in..." : "Sign in with Google"}
							</div>
							{/* Google's real button (invisible overlay, receives clicks) */}
							<div
								ref={googleButtonRef}
								className="absolute inset-0 overflow-hidden rounded-lg"
								style={{ opacity: 0.01 }}
							/>
						</div>
					)}

					<button
						type="button"
						onClick={handleSignIn}
						disabled={loading}
						className={`${GOOGLE_CLIENT_ID ? "mt-3" : "mt-8"} w-full bg-white text-black px-6 py-3 text-body-sm font-medium flex items-center justify-center gap-3 hover:bg-white/90 transition rounded-lg disabled:opacity-50`}
					>
						<svg
							className="w-5 h-5"
							viewBox="0 0 24 24"
							fill="currentColor"
							role="img"
							aria-label="Apple logo"
						>
							<path d="M17.05 20.28c-.98.95-2.05.88-3.08.4-1.09-.5-2.08-.48-3.24 0-1.44.62-2.2.44-3.06-.4C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z" />
						</svg>
						{loading ? "Signing in..." : "Sign in with Apple"}
					</button>

					{!import.meta.env.PROD && (
						<button
							type="button"
							onClick={async () => {
								setLoading(true);
								setError(null);
								try {
									const result = await api.auth.debug();
									setUser({
										student_id: result.student_id,
										email: result.email,
										display_name: result.display_name,
									});
									navigate({ to: "/app" });
								} catch (err) {
									console.error("Debug login failed:", err);
									setError("Debug login failed.");
								} finally {
									setLoading(false);
								}
							}}
							disabled={loading}
							className="mt-3 w-full text-text-tertiary text-body-xs underline hover:text-text-secondary transition disabled:opacity-50"
						>
							Debug Login
						</button>
					)}

					<p className="mt-6 text-body-xs text-text-tertiary">
						By signing in, you agree to our Terms of Service
					</p>
				</div>
			</div>
		</div>
	);
}
