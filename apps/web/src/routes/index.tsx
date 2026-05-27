import { createFileRoute, redirect } from "@tanstack/react-router";
import { authQueryOptions } from "../hooks/useAuth";
import { queryClient } from "../lib/query-client";
import { ExerciseProofBlock } from "../components/ExerciseProofBlock";
import { trackLandingEvent } from "../lib/landing-analytics";
import type { ProofCardManifest } from "../types/landing";

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

const CARD_MANIFESTS: [ProofCardManifest, ProofCardManifest, ProofCardManifest] = [
	{
		pieceId: "chopin.nocturnes.9-2",
		title: "Nocturne Op. 9 No. 2",
		era: "romantic",
		audioUrl: "/landing/card-1/recording.opus",
		scoreIRUrl: "/landing/card-1/scoreir.json",
		scoreSvgUrl: "/landing/card-1/score.svg",
		focusBar: 4,
		focusBarRange: [3, 5],
		diagnosis:
			"The diminuendo in bar 4 arrives too early — the phrase peak is on beat 2 but your dynamics have already started receding by beat 1. Try holding the swell through beat 2 before releasing.",
		exerciseUrl: "/landing/card-1/exercise.json",
		barTimeline: [
			{ bar: 1, tSec: 0.0 },
			{ bar: 2, tSec: 4.2 },
			{ bar: 3, tSec: 8.5 },
			{ bar: 4, tSec: 12.8 },
			{ bar: 5, tSec: 17.1 },
			{ bar: 6, tSec: 21.4 },
		],
		perBarScores: {
			1: { dynamics: 0.72, timing: 0.81, pedaling: 0.68, articulation: 0.75, phrasing: 0.70, interpretation: 0.74 },
			2: { dynamics: 0.75, timing: 0.79, pedaling: 0.71, articulation: 0.73, phrasing: 0.72, interpretation: 0.76 },
			3: { dynamics: 0.68, timing: 0.82, pedaling: 0.65, articulation: 0.74, phrasing: 0.67, interpretation: 0.71 },
			4: { dynamics: 0.52, timing: 0.78, pedaling: 0.60, articulation: 0.71, phrasing: 0.54, interpretation: 0.63 },
			5: { dynamics: 0.74, timing: 0.80, pedaling: 0.69, articulation: 0.76, phrasing: 0.71, interpretation: 0.75 },
			6: { dynamics: 0.77, timing: 0.83, pedaling: 0.72, articulation: 0.78, phrasing: 0.73, interpretation: 0.77 },
		},
	},
	{
		pieceId: "bach.wtc1.prelude-c-major",
		title: "Prelude in C Major, BWV 846",
		era: "baroque",
		audioUrl: "/landing/card-2/recording.opus",
		scoreIRUrl: "/landing/card-2/scoreir.json",
		scoreSvgUrl: "/landing/card-2/score.svg",
		focusBar: 3,
		focusBarRange: [2, 4],
		diagnosis:
			"Bar 3 has a slight rush on the second arpeggio group — your timing score dips here. Isolate beats 3–4 and practice with a metronome at 80% tempo before returning to full speed.",
		exerciseUrl: "/landing/card-2/exercise.json",
		barTimeline: [
			{ bar: 1, tSec: 0.0 },
			{ bar: 2, tSec: 3.1 },
			{ bar: 3, tSec: 6.2 },
			{ bar: 4, tSec: 9.3 },
			{ bar: 5, tSec: 12.4 },
			{ bar: 6, tSec: 15.5 },
		],
		perBarScores: {
			1: { dynamics: 0.80, timing: 0.85, pedaling: 0.78, articulation: 0.82, phrasing: 0.79, interpretation: 0.81 },
			2: { dynamics: 0.78, timing: 0.80, pedaling: 0.76, articulation: 0.81, phrasing: 0.77, interpretation: 0.79 },
			3: { dynamics: 0.79, timing: 0.59, pedaling: 0.77, articulation: 0.80, phrasing: 0.76, interpretation: 0.78 },
			4: { dynamics: 0.81, timing: 0.84, pedaling: 0.79, articulation: 0.83, phrasing: 0.80, interpretation: 0.82 },
			5: { dynamics: 0.82, timing: 0.86, pedaling: 0.80, articulation: 0.84, phrasing: 0.81, interpretation: 0.83 },
			6: { dynamics: 0.80, timing: 0.85, pedaling: 0.78, articulation: 0.82, phrasing: 0.79, interpretation: 0.81 },
		},
	},
	{
		pieceId: "satie.gymnopedies.1",
		title: "Gymnopédie No. 1",
		era: "contemporary",
		audioUrl: "/landing/card-3/recording.opus",
		scoreIRUrl: "/landing/card-3/scoreir.json",
		scoreSvgUrl: "/landing/card-3/score.svg",
		focusBar: 5,
		focusBarRange: [4, 6],
		diagnosis:
			"The melody note on beat 1 of bar 5 needs more weight relative to the left-hand chord. Your interpretation score reflects that the top voice isn't projecting above the accompaniment — try voicing the right hand with the fifth finger leading.",
		exerciseUrl: "/landing/card-3/exercise.json",
		barTimeline: [
			{ bar: 1, tSec: 0.0 },
			{ bar: 2, tSec: 5.0 },
			{ bar: 3, tSec: 10.0 },
			{ bar: 4, tSec: 15.0 },
			{ bar: 5, tSec: 20.0 },
			{ bar: 6, tSec: 25.0 },
		],
		perBarScores: {
			1: { dynamics: 0.70, timing: 0.76, pedaling: 0.73, articulation: 0.68, phrasing: 0.72, interpretation: 0.69 },
			2: { dynamics: 0.71, timing: 0.77, pedaling: 0.74, articulation: 0.69, phrasing: 0.73, interpretation: 0.70 },
			3: { dynamics: 0.72, timing: 0.78, pedaling: 0.75, articulation: 0.70, phrasing: 0.74, interpretation: 0.71 },
			4: { dynamics: 0.69, timing: 0.75, pedaling: 0.72, articulation: 0.67, phrasing: 0.71, interpretation: 0.68 },
			5: { dynamics: 0.58, timing: 0.76, pedaling: 0.71, articulation: 0.66, phrasing: 0.70, interpretation: 0.53 },
			6: { dynamics: 0.71, timing: 0.77, pedaling: 0.74, articulation: 0.69, phrasing: 0.73, interpretation: 0.70 },
		},
	},
];

function LandingPage() {
	return (
		<div data-landing="">
			<HeroSection />
			<ExerciseProofBlock manifests={CARD_MANIFESTS} />
			<FinalCtaSection />
			<LandingFooter />
		</div>
	);
}

function HeroSection() {
	return (
		<section className="relative h-screen flex items-center justify-center overflow-hidden">
			{/* Full-bleed background image */}
			<img
				src="/Image1.jpg"
				alt="Grand piano seen from above"
				className="absolute inset-0 w-full h-full object-cover"
			/>

			{/* Gradient overlay for text legibility */}
			<div
				className="absolute inset-0"
				style={{
					background:
						"linear-gradient(to top, #2D2926 0%, #2D2926 5%, rgba(45,41,38,0.7) 30%, rgba(45,41,38,0.2) 60%, rgba(45,41,38,0.05) 100%)",
				}}
			/>

			{/* Content */}
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
					<a
						href="/app"
						className="bg-cream text-espresso px-8 py-3.5 text-body-sm font-medium hover:brightness-95 transition inline-block"
						onClick={() => trackLandingEvent("landing_hero_cta_click")}
					>
						Start Practicing
					</a>
				</div>
			</div>
		</section>
	);
}

function FinalCtaSection() {
	return (
		<section className="py-32 lg:py-40">
			<div className="max-w-4xl mx-auto px-6 lg:px-12 text-center">
				<h2 className="font-display text-display-md lg:text-display-xl text-cream">
					Your playing. Heard clearly.
				</h2>

				<div className="mt-10">
					<a
						href="/app"
						className="bg-accent text-cream px-8 py-3.5 text-body-sm font-medium hover:brightness-110 transition inline-block"
						onClick={() => trackLandingEvent("landing_final_cta_click")}
					>
						Start your first session
					</a>
				</div>
			</div>
		</section>
	);
}

function LandingFooter() {
	return (
		<footer className="py-8 border-t border-border">
			<div className="max-w-5xl mx-auto px-6 lg:px-12">
				<p className="text-body-xs text-text-tertiary">
					<sup>1</sup> Foscarin S. et al., &ldquo;MIDI2Score: Automatic Score Transcription for Piano Music,&rdquo; ISMIR 2024.
				</p>
			</div>
		</footer>
	);
}
