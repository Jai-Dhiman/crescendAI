import type { DimScores, ObservationEvent } from "./practice-api";

export interface MockSessionData {
	piece: string;
	section: string;
	durationSeconds: number;
	observations: ObservationEvent[];
	scores: DimScores;
}

type Dimension =
	| "dynamics"
	| "timing"
	| "pedaling"
	| "articulation"
	| "phrasing"
	| "interpretation";

interface ObservationTemplate {
	dimension: Dimension;
	framing: string;
	text: string;
	barRange: [number, number];
}

// Observation templates grounded in the Chopin Nocturne Op.9 No.2 score.
// Bar numbers correspond to the actual score structure.
const OBSERVATION_TEMPLATES: ObservationTemplate[] = [
	// Dynamics
	{
		dimension: "dynamics",
		framing: "correction",
		text: "The melody entrance at bar 1 could use a warmer piano -- let the note bloom rather than landing flat.",
		barRange: [1, 2],
	},
	{
		dimension: "dynamics",
		framing: "recognition",
		text: "Nice crescendo shape through bars 5-8. The peak felt natural and well-timed.",
		barRange: [5, 8],
	},
	{
		dimension: "dynamics",
		framing: "correction",
		text: "The dolce passage at bar 13 needs more contrast -- you're playing at roughly the same level as the previous phrase.",
		barRange: [13, 16],
	},

	// Timing
	{
		dimension: "timing",
		framing: "correction",
		text: "The rubato in bars 3-4 is rushing slightly on the way back to tempo. Let the phrase breathe before settling.",
		barRange: [3, 4],
	},
	{
		dimension: "timing",
		framing: "encouragement",
		text: "Your sense of pulse is steady through bars 9-12. That foundation lets the ornaments feel free without losing direction.",
		barRange: [9, 12],
	},
	{
		dimension: "timing",
		framing: "correction",
		text: "The trill at bar 17 is compressing the beat. Try keeping the left hand absolutely steady and letting the trill float above it.",
		barRange: [17, 18],
	},

	// Pedaling
	{
		dimension: "pedaling",
		framing: "correction",
		text: "Bars 5-6 sound muddy -- the pedal is catching the bass note change. Try a half-pedal or quicker change on beat 1.",
		barRange: [5, 6],
	},
	{
		dimension: "pedaling",
		framing: "recognition",
		text: "Clean pedaling through bars 21-24. The harmonic changes come through clearly.",
		barRange: [21, 24],
	},
	{
		dimension: "pedaling",
		framing: "correction",
		text: "The sustained passage at bar 25 could use more pedal to connect the legato line. It sounds a bit dry right now.",
		barRange: [25, 26],
	},

	// Articulation
	{
		dimension: "articulation",
		framing: "correction",
		text: "The ornamental turn at bar 7 is slightly uneven -- the second note is getting swallowed. Try isolating it slowly.",
		barRange: [7, 8],
	},
	{
		dimension: "articulation",
		framing: "recognition",
		text: "Beautiful legato connection in bars 9-10. Each note passes seamlessly to the next.",
		barRange: [9, 10],
	},

	// Phrasing
	{
		dimension: "phrasing",
		framing: "correction",
		text: "The phrase across bars 1-4 should arc as a single breath. Right now it feels like two separate gestures at bar 3.",
		barRange: [1, 4],
	},
	{
		dimension: "phrasing",
		framing: "encouragement",
		text: "The long phrase from bar 13 to bar 20 had a lovely shape -- you found the peak and let it resolve naturally.",
		barRange: [13, 20],
	},
	{
		dimension: "phrasing",
		framing: "question",
		text: "Where do you feel the climax of bars 21-28? Experiment with placing it at bar 24 versus bar 26 and see which feels more convincing.",
		barRange: [21, 28],
	},

	// Interpretation
	{
		dimension: "interpretation",
		framing: "encouragement",
		text: "There is a singing quality in bars 9-16 that feels genuinely personal. That interpretive voice is worth developing.",
		barRange: [9, 16],
	},
	{
		dimension: "interpretation",
		framing: "question",
		text: "The coda at bar 29 can be played inward and reflective or with more projection. Which character are you going for?",
		barRange: [29, 32],
	},
	{
		dimension: "interpretation",
		framing: "correction",
		text: "The repeated section at bar 13 sounds identical to the first time. Try a different color -- quieter, or with a slight tempo shift.",
		barRange: [13, 16],
	},
];

function randomBetween(min: number, max: number): number {
	return min + Math.random() * (max - min);
}

function pickRandom<T>(arr: T[], count: number): T[] {
	const shuffled = [...arr].sort(() => Math.random() - 0.5);
	return shuffled.slice(0, count);
}

export function generateMockSession(): MockSessionData {
	const observationCount = 3 + Math.floor(Math.random() * 3); // 3-5
	const selected = pickRandom(OBSERVATION_TEMPLATES, observationCount);

	// Ensure dimension variety -- no more than 2 from same dimension
	const dimensionCounts = new Map<string, number>();
	const filtered = selected.filter((obs) => {
		const count = dimensionCounts.get(obs.dimension) ?? 0;
		if (count >= 2) return false;
		dimensionCounts.set(obs.dimension, count + 1);
		return true;
	});

	const observations: ObservationEvent[] = filtered.map((t) => ({
		text: t.text,
		dimension: t.dimension,
		framing: t.framing,
		barRange: t.barRange,
	}));

	const scores: DimScores = {
		dynamics: Number.parseFloat(randomBetween(0.4, 0.85).toFixed(2)),
		timing: Number.parseFloat(randomBetween(0.45, 0.8).toFixed(2)),
		pedaling: Number.parseFloat(randomBetween(0.35, 0.75).toFixed(2)),
		articulation: Number.parseFloat(randomBetween(0.5, 0.9).toFixed(2)),
		phrasing: Number.parseFloat(randomBetween(0.4, 0.8).toFixed(2)),
		interpretation: Number.parseFloat(randomBetween(0.45, 0.85).toFixed(2)),
	};

	return {
		piece: "Chopin - Nocturne Op. 9 No. 2 in E-flat Major",
		section: "Bars 1-32",
		durationSeconds: Math.floor(randomBetween(480, 1200)), // 8-20 min
		observations,
		scores,
	};
}

export const DIMENSION_COLORS: Record<Dimension, string> = {
	dynamics: "#b0816a",
	timing: "#9a8a7a",
	pedaling: "#7a8a9a",
	articulation: "#9a7a8a",
	phrasing: "#8a9a7a",
	interpretation: "#9a917a",
};

export const DIMENSION_LABELS: Record<Dimension, string> = {
	dynamics: "Dynamics",
	timing: "Timing",
	pedaling: "Pedaling",
	articulation: "Articulation",
	phrasing: "Phrasing",
	interpretation: "Interpretation",
};
