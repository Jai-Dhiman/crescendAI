import { describe, expect, it } from "vitest";
import { buildV6WsPayload } from "./session-brain";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";

const HEADLINE =
	"You showed up and put in real work today. The session was short but focused, and we'll keep building from here. There is plenty to dig into next time, and I'll be ready when you are. Keep listening for the shape of each phrase as you play. " +
	"Tomorrow we'll come at it fresh with one specific thing to chase down.";

const ARTIFACT: SynthesisArtifact = {
	session_id: "sess_42",
	synthesis_scope: "session",
	strengths: [],
	focus_areas: [],
	proposed_exercises: [],
	dominant_dimension: "phrasing",
	recurring_pattern: null,
	next_session_focus: null,
	diagnosis_refs: [],
	headline: HEADLINE,
};

describe("buildV6WsPayload", () => {
	it("maps artifact.headline to WebSocket text field", () => {
		const payload = buildV6WsPayload(ARTIFACT);
		expect(payload.type).toBe("synthesis");
		expect(payload.text).toBe(HEADLINE);
		expect(payload.components).toEqual([]);
		expect(payload.isFallback).toBe(false);
	});

	it("uses artifact.headline as text, not a free-form string", () => {
		const customHeadline =
			"Different headline for this test, long enough to pass schema validation at three hundred characters minimum so lets add some more content here to ensure we exceed the threshold easily and have a proper test fixture that will not fail Zod checks now.";
		const payload = buildV6WsPayload({ ...ARTIFACT, headline: customHeadline });
		expect(payload.text).toBe(customHeadline);
		expect(payload.text).not.toBe(""); // guard: never produces empty text
	});

	it("always returns empty components array in V6", () => {
		const payload = buildV6WsPayload({
			...ARTIFACT,
			proposed_exercises: ["ex1", "ex2"],
		});
		expect(payload.components).toEqual([]); // Plan 4 fills in real exercise component construction
	});
});
