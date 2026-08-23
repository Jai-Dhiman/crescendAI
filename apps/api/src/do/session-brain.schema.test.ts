// apps/api/src/do/session-brain.schema.test.ts
import { describe, expect, it } from "vitest";
import {
	createInitialState,
	sessionStateSchema,
	wsOutgoingMessageSchema,
} from "./session-brain.schema";

describe("sessionStateSchema followerState removal", () => {
	it("parsed state does not include followerState property", () => {
		const initial = createInitialState("sess-1", "student-1", null);
		expect("followerState" in initial).toBe(false);
	});

	it("schema strips stale followerState key from legacy persisted state", () => {
		const legacyRaw = {
			version: 0,
			sessionId: "sess-1",
			studentId: "student-1",
			conversationId: null,
			followerState: { lastKnownBar: 5 }, // stale key from old schema
		};
		const parsed = sessionStateSchema.parse(legacyRaw);
		expect("followerState" in parsed).toBe(false);
	});

	it("createInitialState produces a valid schema parse without followerState", () => {
		const state = createInitialState("sess-2", "student-2", "conv-1");
		expect(() => sessionStateSchema.parse(state)).not.toThrow();
	});
});

describe("identificationNoteBuffer state field", () => {
	it("createInitialState starts with an empty note buffer and no legacy count", () => {
		const s = createInitialState("sess", "stud", null);
		expect(s.identificationNoteBuffer).toEqual([]);
		expect("identificationNoteCount" in s).toBe(false);
	});

	it("sessionStateSchema accepts a populated note buffer and round-trips it", () => {
		const s = createInitialState("sess", "stud", null);
		s.identificationNoteBuffer = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 100 },
			{ pitch: 64, onset: 0.5, offset: 0.9, velocity: 90 },
		];
		const parsed = sessionStateSchema.parse(s);
		expect(parsed.identificationNoteBuffer).toHaveLength(2);
		expect(parsed.identificationNoteBuffer[0]?.pitch).toBe(60);
	});

	it("sessionStateSchema defaults the buffer to [] when absent", () => {
		const raw = { ...createInitialState("sess", "stud", null) } as Record<
			string,
			unknown
		>;
		delete raw.identificationNoteBuffer;
		const parsed = sessionStateSchema.parse(raw);
		expect(parsed.identificationNoteBuffer).toEqual([]);
	});
});

describe("createInitialState — pipeline retarget fields (#162)", () => {
	it("seeds markBaseline null, markBaselineLoaded false, empty samples/marks/zones", () => {
		const state = createInitialState("sess", "stud", null);
		expect(state.markBaseline).toBeNull();
		expect(state.markBaselineLoaded).toBe(false);
		expect(state.sessionSamples).toEqual({});
		expect(state.emittedMarks).toEqual([]);
		expect(state.noCommentZones).toEqual([]);
	});
});

describe("wsOutgoingMessageSchema — mark and review (#162)", () => {
	it("accepts a valid mark payload", () => {
		const msg = {
			type: "mark",
			mark: {
				id: "sess-dynamics-7",
				anchor: { type: "timestamp", atSeconds: 105 },
				taxonomy: "needs_work",
				dimension: "dynamics",
				evidence: "velocity range 60, mean 40",
				lifecycle: "active",
			},
		};
		expect(() => wsOutgoingMessageSchema.parse(msg)).not.toThrow();
	});

	it("accepts a valid review payload with null verdict on synthesis failure", () => {
		const msg = {
			type: "review",
			verdict: null,
			carry_forward: null,
			no_comment_zones: [{ atSeconds: 30, reason: "insufficient_notes" }],
		};
		expect(() => wsOutgoingMessageSchema.parse(msg)).not.toThrow();
	});
});
