import { describe, expect, it } from "vitest";
import { buildSynthesisFraming, UNIFIED_TEACHER_SYSTEM } from "./prompts";

describe("UNIFIED_TEACHER_SYSTEM", () => {
	it("includes search_catalog tool guidance", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain("search_catalog");
	});

	it("instructs teacher to search before asking student for piece_id", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain(
			"Never ask the student for a piece ID",
		);
	});

	it("instructs teacher to disambiguate multiple matches", () => {
		expect(UNIFIED_TEACHER_SYSTEM).toContain(
			"If multiple matches are returned",
		);
	});
});

describe("buildSynthesisFraming", () => {
	const pieceMetadata = {
		title: "Prelude No. 1",
		composer: "Bach",
		skill_level: 3,
	};
	const topMoments = [{ dimension: "articulation", score: 0.6 }];

	it("includes style_guidance block for Bach", () => {
		const out = buildSynthesisFraming(
			300_000,
			"continuous_play",
			topMoments,
			[],
			pieceMetadata,
			"",
			"Bach",
		);
		expect(out).toContain("<style_guidance");
		expect(out).toContain("Baroque");
		expect(out).toContain("articulation");
	});

	it("style_guidance appears between session_data and task", () => {
		const out = buildSynthesisFraming(
			300_000,
			"continuous_play",
			topMoments,
			[],
			pieceMetadata,
			"",
			"Bach",
		);
		const sessionIdx = out.indexOf("</session_data>");
		const styleIdx = out.indexOf("<style_guidance");
		const taskIdx = out.indexOf("<task>");
		expect(sessionIdx).toBeGreaterThan(-1);
		expect(styleIdx).toBeGreaterThan(sessionIdx);
		expect(taskIdx).toBeGreaterThan(styleIdx);
	});

	it("omits style_guidance for unknown composer", () => {
		const out = buildSynthesisFraming(
			300_000,
			"continuous_play",
			topMoments,
			[],
			{ ...pieceMetadata, composer: "Unknown" },
			"",
			"Unknown",
		);
		expect(out).not.toContain("<style_guidance");
	});

	it("includes student_memory when given", () => {
		const out = buildSynthesisFraming(
			300_000,
			"continuous_play",
			topMoments,
			[],
			pieceMetadata,
			"Student prefers slow practice.",
			"Bach",
		);
		expect(out).toContain("<student_memory>");
		expect(out).toContain("Student prefers slow practice.");
	});

	it("Chopin resolves to Romantic era guidance", () => {
		const out = buildSynthesisFraming(
			300_000,
			"continuous_play",
			topMoments,
			[],
			{ ...pieceMetadata, composer: "Chopin" },
			"",
			"Chopin",
		);
		expect(out).toContain("Romantic");
		expect(out).toContain("dynamics");
	});
});

describe("buildSynthesisFraming + teacher_voice", () => {
	const pieceMetadata = {
		title: "Prelude",
		composer: "Chopin",
		skill_level: 3,
	};

	it("includes <teacher_voice> block after <style_guidance>", () => {
		const out = buildSynthesisFraming(
			900_000,
			"continuous_play",
			[
				{
					dimension: "dynamics",
					score: 0.8,
					deviation_from_mean: 0.25,
					direction: "above_average",
				},
				{
					dimension: "timing",
					score: 0.3,
					deviation_from_mean: -0.18,
					direction: "below_average",
				},
			],
			[],
			pieceMetadata,
			"",
			"Chopin",
		);
		const styleIdx = out.indexOf("<style_guidance");
		const voiceIdx = out.indexOf("<teacher_voice");
		const taskIdx = out.indexOf("<task>");
		expect(styleIdx).toBeGreaterThan(-1);
		expect(voiceIdx).toBeGreaterThan(styleIdx);
		expect(taskIdx).toBeGreaterThan(voiceIdx);
	});

	it("includes <also_consider> block", () => {
		const out = buildSynthesisFraming(
			900_000,
			"continuous_play",
			[
				{
					dimension: "dynamics",
					score: 0.8,
					deviation_from_mean: 0.25,
					direction: "above_average",
				},
			],
			[],
			pieceMetadata,
			"",
			"Chopin",
		);
		expect(out).toContain("<also_consider");
	});
});
