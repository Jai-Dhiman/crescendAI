import { createFileRoute, redirect } from "@tanstack/react-router";
import { type FormEvent, useEffect, useState } from "react";
import { ScoreHighlightCard } from "../components/cards/ScoreHighlightCard";
import { BrowserFrame, PhoneFrame } from "../components/landing/DeviceFrames";
import { authQueryOptions } from "../hooks/useAuth";
import { api } from "../lib/api";
import { trackLandingEvent } from "../lib/landing-analytics";
import { queryClient } from "../lib/query-client";
import { scoreRenderer } from "../lib/score-renderer";
import type { ScoreHighlightConfig } from "../lib/types";

export const Route = createFileRoute("/")({
	beforeLoad: async () => {
		if (import.meta.env.VITE_AUTH_MODE !== "live") return;
		const user = queryClient.getQueryData(authQueryOptions.queryKey);
		if (user) {
			throw redirect({ to: "/app" });
		}
	},
	component: LandingPage,
});

// Real, renderable score: Chopin Nocturne Op.9 No.2. The annotation illustrates
// the feedback style, labelled so we never imply it's analysis of the visitor.
const NOCTURNE_HIGHLIGHT: ScoreHighlightConfig = {
	pieceId: "chopin.nocturnes.9-2",
	highlights: [
		{
			bars: [1, 4],
			dimension: "phrasing",
			annotation:
				"The opening line should breathe as one long phrase. Let it lift toward the high point rather than arriving there too soon.",
		},
	],
};

function scrollToWaitlist() {
	document.getElementById("waitlist")?.scrollIntoView({ behavior: "smooth" });
}

function LandingPage() {
	return (
		<div data-landing="">
			<HeroSection />
			<CascadingQuoteSection />
			<GlimpseSection />
			<WaitlistSection />
		</div>
	);
}

function HeroSection() {
	return (
		<section className="relative h-screen flex items-center justify-center overflow-hidden">
			<img
				src="/Image1.jpg"
				alt="Grand piano seen from above"
				className="absolute inset-0 w-full h-full object-cover"
			/>
			<div
				className="absolute inset-0"
				style={{
					background:
						"linear-gradient(to top, #2D2926 0%, #2D2926 5%, rgba(45,41,38,0.7) 30%, rgba(45,41,38,0.2) 60%, rgba(45,41,38,0.05) 100%)",
				}}
			/>
			<div className="relative z-10 text-center px-6">
				<h1
					className="font-display text-cream text-balance"
					style={{
						fontSize: "clamp(3rem, 8vw, 7rem)",
						lineHeight: 1.05,
						letterSpacing: "-0.03em",
					}}
				>
					A teacher for every pianist.
				</h1>
				<div className="mt-10">
					<button
						type="button"
						className="bg-cream text-espresso px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
						onClick={() => {
							trackLandingEvent("landing_hero_cta_click");
							scrollToWaitlist();
						}}
					>
						Start Practicing
					</button>
				</div>
			</div>
		</section>
	);
}

function GlimpseSection() {
	return (
		<section className="py-24 lg:py-32 overflow-hidden">
			<div className="max-w-6xl mx-auto px-6 lg:px-12">
				<div className="text-center mb-16">
					<h2 className="font-display text-display-md lg:text-display-lg text-cream">
						See what your playing sounds like.
					</h2>
					<p className="mt-4 text-body-md text-text-secondary">
						Record on your phone. Get back the exact bars to focus on, and why.
					</p>
				</div>

				{/* Overlapping device frames: web (score) + phone (capture) */}
				<div className="relative max-w-4xl mx-auto pb-16 lg:pb-0">
					<BrowserFrame className="w-full lg:w-[78%]">
						<div className="p-4 lg:p-6">
							<div className="flex items-center gap-2 mb-3">
								<span className="text-label-sm uppercase tracking-wide text-text-tertiary">
									Sample
								</span>
								<span className="text-body-xs text-text-tertiary">
									· Chopin, Nocturne Op. 9 No. 2
								</span>
							</div>
							<NocturneScore config={NOCTURNE_HIGHLIGHT} />
						</div>
					</BrowserFrame>

					<PhoneFrame className="absolute bottom-0 right-0 w-[42%] sm:w-[34%] lg:w-[26%] translate-y-6 lg:translate-y-12">
						<img
							src="/landing/app-ios.png"
							alt="CrescendAI iOS app"
							className="w-full h-full object-cover object-top"
						/>
					</PhoneFrame>
				</div>

				<p className="mt-8 lg:mt-20 text-center text-body-xs text-text-tertiary">
					Real score rendering from CrescendAI. Full performance analysis
					arrives with the app.
				</p>
			</div>
		</section>
	);
}

// ScoreHighlightCard calls scoreRenderer.getClip directly, which needs the piece
// bytes loaded first. On a cold landing nothing loads them, so we do it here.
function NocturneScore({ config }: { config: ScoreHighlightConfig }) {
	const [status, setStatus] = useState<"loading" | "ready" | "error">(
		"loading",
	);

	useEffect(() => {
		let cancelled = false;
		scoreRenderer
			.load(config.pieceId)
			.then(() => {
				if (!cancelled) setStatus("ready");
			})
			.catch(() => {
				if (!cancelled) setStatus("error");
			});
		return () => {
			cancelled = true;
		};
	}, [config.pieceId]);

	if (status === "error") return null;
	if (status === "loading") {
		return (
			<div className="h-40 flex items-center justify-center">
				<div className="w-4 h-4 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
			</div>
		);
	}
	return <ScoreHighlightCard config={config} />;
}

function CascadingQuoteSection() {
	return (
		<section className="py-24 lg:py-32">
			<div className="max-w-7xl mx-auto px-6 lg:px-12">
				<div className="grid grid-cols-1 lg:grid-cols-[5fr_6fr] gap-12 lg:gap-16 items-center">
					<div className="flex flex-col">
						<div className="w-[55%] self-start">
							<img
								src="/Image2.jpg"
								alt="Practicing alone, the struggle of hearing your own mistakes"
								className="w-full object-cover"
								style={{ aspectRatio: "3/2" }}
							/>
						</div>
						<div className="w-[55%] self-center">
							<img
								src="/Image3.jpg"
								alt="A moment of guidance, focused attention on the score"
								className="w-full object-cover"
								style={{ aspectRatio: "3/2" }}
							/>
						</div>
						<div className="w-[55%] self-end">
							<img
								src="/Image4.jpg"
								alt="The breakthrough, playing with confidence"
								className="w-full object-cover"
								style={{ aspectRatio: "3/2" }}
							/>
						</div>
					</div>
					<div>
						<blockquote className="font-display italic text-display-md lg:text-display-lg text-cream leading-snug">
							"What's the one thing that sounds off that I can't hear myself?"
						</blockquote>
					</div>
				</div>
			</div>
		</section>
	);
}

function WaitlistSection() {
	return (
		<section id="waitlist" className="py-32 lg:py-40 scroll-mt-8">
			<div className="max-w-2xl mx-auto px-6 lg:px-12 text-center">
				<h2 className="font-display text-display-md lg:text-display-xl text-cream">
					Every pianist deserves a great teacher.
				</h2>
				<p className="mt-5 text-body-md text-text-secondary">
					We're opening access soon. Join the waitlist and we'll let you know
					the moment it's ready.
				</p>
				<div className="mt-10">
					<WaitlistForm />
				</div>
			</div>
		</section>
	);
}

type SubmitState = "idle" | "submitting" | "success" | "error";

function WaitlistForm() {
	const [email, setEmail] = useState("");
	const [website, setWebsite] = useState(""); // honeypot
	const [state, setState] = useState<SubmitState>("idle");
	const [errorMsg, setErrorMsg] = useState("");

	async function handleSubmit(e: FormEvent) {
		e.preventDefault();
		if (state === "submitting") return;
		if (website) {
			// Bot filled the honeypot, pretend success, do nothing.
			setState("success");
			return;
		}
		setState("submitting");
		setErrorMsg("");
		try {
			await api.waitlist.join(email, "landing");
			trackLandingEvent("landing_waitlist_signup");
			setState("success");
		} catch (err) {
			setState("error");
			setErrorMsg(
				err instanceof Error
					? err.message
					: "Something went wrong. Please try again.",
			);
		}
	}

	if (state === "success") {
		return (
			<p className="text-body-lg text-cream font-display">
				You're on the list. We'll be in touch.
			</p>
		);
	}

	return (
		<form
			onSubmit={handleSubmit}
			className="flex flex-col sm:flex-row gap-3 max-w-md mx-auto"
		>
			{/* Honeypot: hidden from humans, catches bots */}
			<input
				type="text"
				name="website"
				tabIndex={-1}
				autoComplete="off"
				value={website}
				onChange={(e) => setWebsite(e.target.value)}
				className="hidden"
				aria-hidden="true"
			/>
			<label htmlFor="waitlist-email" className="sr-only">
				Email address
			</label>
			<input
				id="waitlist-email"
				type="email"
				required
				value={email}
				onChange={(e) => setEmail(e.target.value)}
				placeholder="you@example.com"
				disabled={state === "submitting"}
				className="flex-1 bg-surface border border-border px-4 py-3 text-body-sm text-cream placeholder:text-text-tertiary focus:outline-none focus:border-accent transition disabled:opacity-60"
			/>
			<button
				type="submit"
				disabled={state === "submitting"}
				className="bg-accent text-cream px-6 py-3 text-body-sm font-medium hover:brightness-110 transition disabled:opacity-60 whitespace-nowrap"
			>
				{state === "submitting" ? "Joining…" : "Join the waitlist"}
			</button>
			{state === "error" && (
				<p
					className="sm:absolute sm:mt-16 text-body-xs text-red-400"
					role="alert"
				>
					{errorMsg}
				</p>
			)}
		</form>
	);
}
