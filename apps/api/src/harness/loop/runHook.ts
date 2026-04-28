import type { SynthesisArtifact } from "../artifacts/synthesis";
import { getCompoundBinding } from "./compound-registry";
import { runPhase1 } from "./phase1";
import { runPhase2 } from "./phase2";
import type { CompoundBinding, HookContext, HookEvent, HookKind, Phase2Binding, PhaseContext } from "./types";

export type ArtifactFor<H extends HookKind> = H extends "OnSessionEnd"
	? SynthesisArtifact
	: unknown;

const DEFAULT_TURN_CAP = 8;

function isPhase2Binding(b: CompoundBinding): b is Phase2Binding {
	return b.phases === 2 && b.artifactSchema !== undefined && b.artifactToolName !== undefined;
}

export async function* runHook<H extends HookKind>(
	hook: H,
	ctx: HookContext,
): AsyncGenerator<HookEvent<ArtifactFor<H>>> {
	const binding = getCompoundBinding(hook);
	if (!binding) {
		yield {
			type: "phase_error",
			phase: 1,
			error: `no compound bound to hook ${hook}`,
		};
		return;
	}

	const phaseCtx: PhaseContext = { ...ctx, turnCap: DEFAULT_TURN_CAP };
	const collectedDiagnoses: unknown[] = [];

	try {
		for await (const ev of runPhase1(phaseCtx, binding)) {
			if (ev.type === "phase1_tool_result" && ev.ok) {
				collectedDiagnoses.push(ev.output);
			}
			yield ev as HookEvent<ArtifactFor<H>>;
		}
	} catch (err) {
		yield {
			type: "phase_error",
			phase: 1,
			error: err instanceof Error ? err.message : String(err),
		};
		return;
	}

	if (isPhase2Binding(binding)) {
		try {
			for await (const ev of runPhase2(phaseCtx, binding, collectedDiagnoses)) {
				yield ev as HookEvent<ArtifactFor<H>>;
			}
		} catch (err) {
			yield {
				type: "phase_error",
				phase: 2,
				error: err instanceof Error ? err.message : String(err),
			};
		}
	}
}
