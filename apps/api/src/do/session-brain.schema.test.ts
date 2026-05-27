// apps/api/src/do/session-brain.schema.test.ts
import { describe, expect, it } from "vitest";
import { sessionStateSchema, createInitialState } from "./session-brain.schema";

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
