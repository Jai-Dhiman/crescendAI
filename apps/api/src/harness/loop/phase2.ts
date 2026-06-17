import { zodToJsonSchema } from "zod-to-json-schema";
import { FIRST_SESSION_GUARDRAIL } from "../../services/prompts";
import { callModel } from "./gateway-client";
import { withRetries } from "./middleware";
import { routeModel } from "./route-model";
import type { HookEvent, Phase2Binding, PhaseContext } from "./types";

export function buildPhase2Prompt(
	digest: Record<string, unknown>,
	diagnoses: unknown[],
	guardrail: string,
): string {
	const reflectionInstruction =
		"Headline instructions: write a light reflection in 2-4 sentences about what happened " +
		"in this session, ending in exactly one directional question about the dominant_dimension " +
		"(e.g. 'Want a drill targeting that?'). The headline must be 300-500 characters total. " +
		"Do not list all dimensions; focus on the one area that matters most.\n\n";

	const exerciseInstruction =
		"Exercise instructions: set prescribed_exercise to a single routing decision that targets " +
		"the dominant_dimension. Use kind='own_passage_loop' when the student has been identified " +
		"playing a specific piece and you want them to loop a bar range from it; use " +
		"kind='corpus_drill' when no piece is identified or a general technique drill would be " +
		"more appropriate. Set prescribed_exercise to null if no exercise is warranted. " +
		"Do NOT put a pieceId in prescribed_exercise — that is bound at the serving layer.\n\n";

	return (
		`Session digest:\n${JSON.stringify(digest, null, 2)}\n\n` +
		`Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
		guardrail +
		reflectionInstruction +
		exerciseInstruction +
		`Write the SynthesisArtifact now using the write_synthesis_artifact tool.`
	);
}

// Initial forced-tool call + up to (MAX_PHASE2_ATTEMPTS - 1) validation-repair
// turns. On sparse cold-start sessions the model reliably undershoots the
// headline 300-char minimum; a single forced call with no repair path makes any
// validation miss fatal (the historical "strict validation no-ops on sparse
// state" no-op). Showing the model its rejected artifact + the zod error lets it
// fix the flagged fields while preserving the 300-500 char product contract.
const MAX_PHASE2_ATTEMPTS = 3;

export async function* runPhase2(
	ctx: PhaseContext,
	binding: Phase2Binding,
	diagnoses: unknown[],
): AsyncGenerator<HookEvent<unknown>> {
	yield { type: "phase2_started" };

	const writeTool = {
		name: binding.artifactToolName,
		description:
			"Write the final compound artifact. Call this exactly once with the structured fields.",
		input_schema: artifactInputSchema(binding.artifactSchema),
	};

	const guardrail =
		ctx.digest.reference_mode === "within_session"
			? `${FIRST_SESSION_GUARDRAIL}\n\n`
			: "";

	const userPrompt = buildPhase2Prompt(ctx.digest, diagnoses, guardrail);
	const client = routeModel("phase2_voice", ctx.env);
	const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
		{ role: "user", content: userPrompt },
	];

	let lastInvalid: { raw: unknown; zodError: string } | null = null;

	for (let attempt = 1; attempt <= MAX_PHASE2_ATTEMPTS; attempt++) {
		const response = await withRetries(() =>
			callModel(ctx.env, client, {
				model: client.model,
				max_tokens: 2048,
				messages: messages as Parameters<typeof callModel>[2]["messages"],
				tools: [writeTool],
				tool_choice: { type: "tool", name: binding.artifactToolName },
			}),
		);

		const toolUse = response.content.find(
			(
				b,
			): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
				b.type === "tool_use" && b.name === binding.artifactToolName,
		);

		if (!toolUse) {
			yield {
				type: "phase_error",
				phase: 2,
				error: "no tool_use returned despite forced tool_choice",
			};
			return;
		}

		const parsed = binding.artifactSchema.safeParse(toolUse.input);
		if (parsed.success) {
			yield { type: "artifact", value: parsed.data };
			return;
		}

		lastInvalid = { raw: toolUse.input, zodError: parsed.error.message };

		if (attempt < MAX_PHASE2_ATTEMPTS) {
			// A user turn following an assistant tool_use must carry a tool_result
			// for that tool_use_id (Anthropic API contract), or the call 400s.
			messages.push({ role: "assistant", content: response.content });
			messages.push({
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: toolUse.id,
						is_error: true,
						content:
							`The artifact you wrote failed validation:\n${parsed.error.message}\n\n` +
							`Fix ONLY the flagged fields and call ${binding.artifactToolName} again. ` +
							`In particular, the headline must be between 300 and 500 characters — ` +
							`expand the reflection with concrete detail from the digest if it is too short.`,
					},
				],
			});
		}
	}

	yield {
		type: "validation_error",
		raw: lastInvalid?.raw,
		zodError: lastInvalid?.zodError ?? "unknown validation failure",
	};
}

function artifactInputSchema(
	schema: Phase2Binding["artifactSchema"],
): Record<string, unknown> {
	// The Anthropic Messages API requires tool input_schema to be valid JSON
	// Schema draft 2020-12. The `openApi3` target emits OpenAPI-3.0 constructs
	// (nullable: true, boolean exclusiveMinimum) and `$ref`-dedups repeated
	// subschemas — all rejected with HTTP 400. The default (draft-07) target
	// with refs inlined produces a self-contained schema the API accepts.
	return zodToJsonSchema(schema, { $refStrategy: "none" }) as Record<
		string,
		unknown
	>;
}
