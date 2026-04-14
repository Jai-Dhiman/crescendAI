import { eq } from "drizzle-orm";
import { studentProfiles } from "../db/schema/students";
import { ValidationError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";
import { callWorkersAI } from "./llm";

interface GoalDeadline {
	description: string;
	date?: string | null;
}

interface ExtractedGoals {
	pieces: string[];
	focusAreas: string[];
	deadlines: GoalDeadline[];
	rawText: string;
}

interface ExplicitGoals {
	pieces: string[];
	focusAreas: string[];
	deadlines: GoalDeadline[];
}

export async function extractGoals(
	ctx: ServiceContext,
	studentId: string,
	message: string,
): Promise<ExtractedGoals> {
	const extracted = await extractGoalsWithLlm(ctx, message);
	await mergeGoals(ctx, studentId, extracted);
	return extracted;
}

async function extractGoalsWithLlm(
	ctx: ServiceContext,
	message: string,
): Promise<ExtractedGoals> {
	const systemPrompt =
		"You extract structured data from pianist messages. Return only valid JSON.";

	const userPrompt = `Extract structured practice goals from this pianist's message. Return ONLY valid JSON with no other text.

Message: "${message}"

Return this exact JSON structure:
{
  "pieces": ["list of piece names mentioned"],
  "focusAreas": ["list of musical dimensions or techniques to focus on, e.g. pedaling, dynamics, articulation"],
  "deadlines": [{"description": "what the deadline is for", "date": "YYYY-MM-DD or null if not specific"}],
  "rawText": "the original message"
}

If a field has no matches, use an empty array. Always include rawText.`;

	const responseText = await callWorkersAI(
		ctx.env,
		"@cf/google/gemma-4-26b-a4b-it",
		[
			{ role: "system", content: systemPrompt },
			{ role: "user", content: userPrompt },
		],
		3000,
	);

	let extracted: ExtractedGoals;
	try {
		extracted = JSON.parse(responseText) as ExtractedGoals;
	} catch {
		console.error(
			JSON.stringify({
				level: "error",
				message: "LLM returned invalid JSON for goal extraction",
				raw: responseText,
			}),
		);
		throw new ValidationError(
			"Failed to parse extracted goals from LLM response",
		);
	}

	return extracted;
}

async function mergeGoals(
	ctx: ServiceContext,
	studentId: string,
	newGoals: ExtractedGoals,
): Promise<void> {
	const student = await ctx.db.query.studentProfiles.findFirst({
		where: (s, { eq: eqFn }) => eqFn(s.studentId, studentId),
		columns: { explicitGoals: true },
	});

	let merged: ExplicitGoals = { pieces: [], focusAreas: [], deadlines: [] };

	if (student?.explicitGoals) {
		try {
			merged = JSON.parse(student.explicitGoals) as ExplicitGoals;
		} catch {
			console.error(
				JSON.stringify({
					level: "error",
					message: "Failed to parse existing explicit_goals, resetting",
					studentId,
				}),
			);
			merged = { pieces: [], focusAreas: [], deadlines: [] };
		}
	}

	for (const piece of newGoals.pieces) {
		if (!merged.pieces.includes(piece)) {
			merged.pieces.push(piece);
		}
	}

	for (const area of newGoals.focusAreas) {
		if (!merged.focusAreas.includes(area)) {
			merged.focusAreas.push(area);
		}
	}

	for (const deadline of newGoals.deadlines) {
		merged.deadlines.push(deadline);
	}

	await ctx.db
		.update(studentProfiles)
		.set({ explicitGoals: JSON.stringify(merged), updatedAt: new Date() })
		.where(eq(studentProfiles.studentId, studentId));
}
