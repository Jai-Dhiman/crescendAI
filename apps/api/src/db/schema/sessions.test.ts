import { getTableName } from "drizzle-orm";
import { describe, expect, it } from "vitest";
import { sessions } from "./sessions";

describe("sessions schema", () => {
	it("is the sessions table", () => {
		expect(getTableName(sessions)).toBe("sessions");
	});

	it("carries the piece a session was practising", () => {
		// Nullable by design: a pieceless session is a permanent first-class
		// state (docs/apps/05-ui-system.md#2), not a missing value.
		expect(Object.keys(sessions)).toContain("pieceId");
		expect(sessions.pieceId.notNull).toBe(false);
	});
});
