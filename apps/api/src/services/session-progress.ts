import { z } from "zod";
import { DIMS_6, type Dimension } from "../lib/dims";

// ---------------------------------------------------------------------------
// ProgressSummary — the single source of truth for a show_session_data result.
//
// `chartData` holds the raw numeric rows the CLIENT renders as a chart.
// `modelSummary` is qualitative prose with ZERO raw scores — the only field the
// teacher model ever sees. Separating them by field (not by convention) is the
// context-hygiene boundary: the model-facing path reads `modelSummary`; the
// client-facing path reads `chartData`. Mirrors the numbers-in / prose-out
// contract that the synthesis molecules enforce via one_sentence_finding.
// ---------------------------------------------------------------------------

export const ProgressSummarySchema = z.object({
	queryType: z.enum(["dimension_history", "recent_sessions", "session_detail"]),
	/** Qualitative prose for the teacher model. Never contains raw scores. */
	modelSummary: z.string(),
	/** Raw numeric rows for the client chart. Never shown to the model. */
	chartData: z.unknown(),
});

export type ProgressSummary = z.infer<typeof ProgressSummarySchema>;

interface SessionAvgRow {
	avgDynamics: number | null;
	avgTiming: number | null;
	avgPedaling: number | null;
	avgArticulation: number | null;
	avgPhrasing: number | null;
	avgInterpretation: number | null;
}

interface DimHistoryRow {
	dimension: string;
	dimensionScore: number | null;
	observationText: string | null;
	framing: string | null;
	createdAt: Date | string;
}

const AVG_COL: Record<Dimension, keyof SessionAvgRow> = {
	dynamics: "avgDynamics",
	timing: "avgTiming",
	pedaling: "avgPedaling",
	articulation: "avgArticulation",
	phrasing: "avgPhrasing",
	interpretation: "avgInterpretation",
};

const TREND_EPSILON = 0.03;

function trendWord(earliest: number, latest: number): string {
	const delta = latest - earliest;
	if (delta > TREND_EPSILON) return "improving";
	if (delta < -TREND_EPSILON) return "an area to focus on";
	return "holding steady";
}

function rankDimensions(row: SessionAvgRow): {
	strongest: Dimension | null;
	weakest: Dimension | null;
} {
	const scored = DIMS_6.map((d) => ({ d, v: row[AVG_COL[d]] })).filter(
		(x): x is { d: Dimension; v: number } => typeof x.v === "number",
	);
	if (scored.length === 0) return { strongest: null, weakest: null };
	scored.sort((a, b) => b.v - a.v);
	const strongest = scored[0];
	const weakest = scored[scored.length - 1];
	// Collapse to a single area when the spread is within trend noise: otherwise
	// an all-equal session reads as a real strong/weak gap that isn't there.
	if (strongest.v - weakest.v <= TREND_EPSILON) {
		return { strongest: strongest.d, weakest: null };
	}
	return { strongest: strongest.d, weakest: weakest.d };
}

function distill(queryType: string, data: unknown): string {
	if (queryType === "dimension_history") {
		const rows = Array.isArray(data) ? (data as DimHistoryRow[]) : [];
		if (rows.length === 0) {
			return "No progress observations recorded yet for that query.";
		}
		const byDim = new Map<string, DimHistoryRow[]>();
		for (const r of rows) {
			const list = byDim.get(r.dimension) ?? [];
			list.push(r);
			byDim.set(r.dimension, list);
		}
		const parts: string[] = [];
		for (const [dim, list] of byDim) {
			const chron = [...list].sort(
				(a, b) =>
					new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime(),
			);
			const scored = chron.filter(
				(r): r is DimHistoryRow & { dimensionScore: number } =>
					typeof r.dimensionScore === "number",
			);
			const latest = chron[chron.length - 1];
			const note = latest.framing ?? latest.observationText ?? null;
			const trend =
				scored.length >= 2
					? trendWord(
							scored[0].dimensionScore,
							scored[scored.length - 1].dimensionScore,
						)
					: "not enough history to call a trend";
			const noteClause = note ? ` Most recent note: "${note}".` : "";
			parts.push(
				`${dim} (${list.length} observation${list.length === 1 ? "" : "s"}): trend is ${trend}.${noteClause}`,
			);
		}
		return `Progress history — ${parts.join(" ")}`;
	}

	if (queryType === "recent_sessions") {
		const rows = Array.isArray(data) ? (data as SessionAvgRow[]) : [];
		if (rows.length === 0) {
			return "No recent sessions recorded yet.";
		}
		// rows arrive newest-first (ordered by startedAt desc)
		const newest = rows[0];
		const oldest = rows[rows.length - 1];
		const { strongest, weakest } = rankDimensions(newest);
		const rankClause =
			strongest && weakest
				? `In your most recent session the strongest area was ${strongest}; the area with most room to grow was ${weakest}.`
				: strongest
					? `In your most recent session your work was fairly even across dimensions, led by ${strongest}.`
					: "The most recent session has no scored dimensions yet.";
		const trendParts: string[] = [];
		if (rows.length >= 2) {
			for (const d of DIMS_6) {
				const a = oldest[AVG_COL[d]];
				const b = newest[AVG_COL[d]];
				if (typeof a === "number" && typeof b === "number") {
					trendParts.push(`${d} ${trendWord(a, b)}`);
				}
			}
		}
		const trendClause = trendParts.length
			? ` Trends across the last ${rows.length} sessions: ${trendParts.join(", ")}.`
			: "";
		return `${rankClause}${trendClause}`;
	}

	// session_detail
	if (data === null || data === undefined) {
		return "No session data found for that session.";
	}
	const { strongest, weakest } = rankDimensions(data as SessionAvgRow);
	if (!strongest) {
		return "This session has no scored dimensions yet.";
	}
	if (!weakest) {
		return `In this session your work was fairly even across dimensions, led by ${strongest}.`;
	}
	return `In this session your strongest area was ${strongest}; the area with most room to grow was ${weakest}.`;
}

/**
 * Build a validated ProgressSummary from a show_session_data query result.
 * The distilled prose and the raw chart rows come from one call, so the
 * model-facing and client-facing views can never drift apart.
 */
export function buildProgressSummary(
	queryType: string,
	data: unknown,
): ProgressSummary {
	return ProgressSummarySchema.parse({
		queryType,
		modelSummary: distill(queryType, data),
		chartData: data,
	});
}
