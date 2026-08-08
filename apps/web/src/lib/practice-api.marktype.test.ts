import { describe, expect, it } from "vitest";
import type { Mark } from "./mark";
import { resolveAnchor } from "./mark";
import type { PracticeWsEvent } from "./practice-api";

describe("PracticeWsEvent mark variant", () => {
	it("accepts a { type: 'mark', mark } event and narrows it by discriminant", () => {
		const mark: Mark = {
			id: "m1",
			anchor: resolveAnchor({ atSeconds: 12, alignmentQuality: 0 }),
			taxonomy: "needs_work",
			dimension: "pedaling",
			evidence: "test evidence",
			lifecycle: "active",
		};
		const event: PracticeWsEvent = { type: "mark", mark };

		// Runtime assertion so this test is not pure compile-time theatre: the
		// discriminant narrows the union and the payload survives a JSON round
		// trip unchanged, which is what the WS transport actually does to it.
		const roundTripped: PracticeWsEvent = JSON.parse(JSON.stringify(event));
		expect(roundTripped.type).toBe("mark");
		if (roundTripped.type === "mark") {
			expect(roundTripped.mark.id).toBe("m1");
		}
	});
});
