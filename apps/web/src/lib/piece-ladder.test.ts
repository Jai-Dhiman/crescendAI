import { describe, expect, it } from "vitest";
import { resolvePieceLadderState } from "./piece-ladder";

const guess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

describe("resolvePieceLadderState", () => {
	it("prefers the user's pick over a confident guess", () => {
		expect(
			resolvePieceLadderState({
				userPicked: "chopin-nocturne-op9-no2",
				confidentGuess: guess,
				dismissed: false,
			}),
		).toBe("user-picked");
	});

	it("shows the confirm chip when there is a guess and no pick", () => {
		expect(
			resolvePieceLadderState({
				userPicked: null,
				confidentGuess: guess,
				dismissed: false,
			}),
		).toBe("confirm-chip");
	});

	it("falls to pieceless once the guess is dismissed", () => {
		expect(
			resolvePieceLadderState({
				userPicked: null,
				confidentGuess: guess,
				dismissed: true,
			}),
		).toBe("pieceless");
	});

	it("is pieceless with neither a pick nor a guess", () => {
		expect(
			resolvePieceLadderState({
				userPicked: null,
				confidentGuess: null,
				dismissed: false,
			}),
		).toBe("pieceless");
	});
});
