import "../lib/sentry";
import { QueryClientProvider } from "@tanstack/react-query";
import {
	createRootRoute,
	HeadContent,
	Outlet,
	Scripts,
	useRouterState,
} from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { ToastContainer } from "../components/ToastContainer";
import { useMountEffect, useSyncRef } from "../hooks/useFoundation";
import { AuthProvider } from "../lib/auth";
import { queryClient } from "../lib/query-client";
import { useThemeStore } from "../stores/theme";

import appCss from "../styles/app.css?url";

export const Route = createRootRoute({
	head: () => ({
		meta: [
			{ charSet: "utf-8" },
			{ name: "viewport", content: "width=device-width, initial-scale=1" },
			{ title: "Crescend" },
			{
				name: "description",
				content:
					"Record yourself playing piano. Get the feedback a great teacher would give you: on your tone, your dynamics, your phrasing.",
			},
		],
		links: [
			{ rel: "preconnect", href: "https://fonts.googleapis.com" },
			{
				rel: "preconnect",
				href: "https://fonts.gstatic.com",
				crossOrigin: "anonymous",
			},
			{
				rel: "stylesheet",
				href: "https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap",
			},
			{ rel: "stylesheet", href: appCss },
			{ rel: "icon", type: "image/png", href: "/crescendai.png" },
		],
	}),
	component: RootDocument,
});

const THEME_FLASH_SCRIPT = `(function(){var path=location.pathname;if(path==="/"||path==="/signin")return;var p=localStorage.getItem("crescend-theme");var t=p==="light"||p==="dark"?p:window.matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";if(t==="light")document.documentElement.dataset.theme="light"})();`;

function RootDocument() {
	const pathname = useRouterState({ select: (s) => s.location.pathname });
	const isAppShell = pathname === "/signin" || pathname.startsWith("/app");
	const theme = useThemeStore((s) => s.theme);

	const pathnameRef = useSyncRef(pathname);
	const themeRef = useSyncRef(theme);

	useEffect(() => {
		function applyTheme() {
			const p = pathnameRef.current;
			const t = themeRef.current;
			const isAlwaysDark = p === "/" || p === "/signin";
			if (isAlwaysDark) {
				delete document.documentElement.dataset.theme;
			} else if (t === "light") {
				document.documentElement.dataset.theme = "light";
			} else {
				delete document.documentElement.dataset.theme;
			}
		}

		applyTheme(); // Apply once on mount

		// Subscribe to store changes as the event-driven update signal (Rule 3)
		let subscribed = true;
		const unsubscribe = useThemeStore.subscribe(() => {
			if (subscribed) applyTheme();
		});

		return () => {
			subscribed = false;
			unsubscribe();
		};
	}, []);

	return (
		<html lang="en">
			<head>
				{/* Static script to prevent theme flash - content is hardcoded, not user input */}
				<script dangerouslySetInnerHTML={{ __html: THEME_FLASH_SCRIPT }} />
				<HeadContent />
				<script
					type="text/javascript"
					src="https://appleid.cdn-apple.com/appleauth/static/jsapi/appleid/1/en_US/appleid.auth.js"
				/>
			</head>
			<body className="bg-espresso text-text-primary font-sans">
				<QueryClientProvider client={queryClient}>
					<AuthProvider>
						{!isAppShell && <Header />}
						<main>
							<ErrorBoundary pathname={pathname}>
								<Outlet />
							</ErrorBoundary>
						</main>
						{!isAppShell && <Footer />}
						<ToastContainer />
					</AuthProvider>
				</QueryClientProvider>
				<Scripts />
			</body>
		</html>
	);
}

function Header() {
	const [hidden, setHidden] = useState(false);
	const lastScrollY = useRef(0);

	useMountEffect(() => {
		function onScroll() {
			const y = window.scrollY;
			setHidden(y > 64 && y > lastScrollY.current);
			lastScrollY.current = y;
		}
		window.addEventListener("scroll", onScroll, { passive: true });
		return () => window.removeEventListener("scroll", onScroll);
	});

	return (
		<header
			className="fixed top-0 left-0 right-0 z-50 transition-transform duration-300"
			style={{ transform: hidden ? "translateY(-100%)" : "translateY(0)" }}
		>
			<div className="max-w-7xl mx-auto px-6 lg:px-12 flex items-center justify-between h-16">
				<a href="/" className="font-display text-lg text-cream tracking-tight">
					crescend
				</a>
				<a
					href="/signin"
					className="font-display text-body-sm text-cream hover:text-text-secondary transition-colors"
				>
					Sign In
				</a>
			</div>
		</header>
	);
}

function Footer() {
	return (
		<footer className="py-12 lg:py-16">
			<div className="max-w-7xl mx-auto px-6 lg:px-12">
				<div className="flex flex-col md:flex-row items-center justify-between gap-6 text-body-xs text-text-tertiary">
					<a
						href="/"
						className="font-display text-sm text-cream tracking-tight"
					>
						crescend
					</a>
					<div className="flex items-center gap-6">
						<a
							href="/terms"
							className="hover:text-text-secondary transition-colors"
						>
							Terms of Service
						</a>
						<a
							href="/privacy"
							className="hover:text-text-secondary transition-colors"
						>
							Privacy Policy
						</a>
					</div>
					<p>2026</p>
				</div>
			</div>
		</footer>
	);
}
