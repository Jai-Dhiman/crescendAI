import { expect, test } from "@playwright/test";

// The preview build serves no API, but ScoreRenderer fetches score bytes via
// api.scores.getData() -> /api/scores/:pieceId/data. Fulfil that request from
// the statically-served .mxl instead, which is the pattern already established
// in src/scorehost/score-host.playwright.ts:186. Rendering a real engraving is
// the point of this test; running a real API is not.
const PIECE_ID = "chopin-nocturne-op9-no2";

test("a mark sits over its real measure on a real Verovio engraving", async ({
	page,
}) => {
	await page.route(`**/api/scores/${PIECE_ID}/data`, async (route) => {
		const bytes = await fetch(
			`http://localhost:4173/scores/${PIECE_ID}.mxl`,
		).then((r) => r.arrayBuffer());
		await route.fulfill({
			status: 200,
			contentType: "application/octet-stream",
			body: Buffer.from(bytes),
		});
	});

	await page.goto("/marks-preview");

	const realScore = page.locator("[data-testid='real-score']");
	// Verovio emits <g class="measure" id="..."> once the toolkit has rendered.
	const measures = realScore.locator("g.measure");
	await expect(measures.first()).toBeVisible({ timeout: 90000 });

	// A real engraving of this Nocturne puts many measures on page 1. Asserting
	// a plural count keeps a degenerate render — one measure, or a stand-in
	// injected by mistake — from satisfying this test.
	expect(await measures.count()).toBeGreaterThan(5);

	// The preview anchors its real-score mark to the FIRST bar the IR reports,
	// so the element it resolves to must exist and the glyph must be visible.
	const glyph = realScore.locator("button[aria-expanded]").first();
	await expect(glyph).toBeVisible();

	// The load-bearing assertion: the glyph's box overlaps the measure element
	// it claims to mark. A wrong-bar or invented position fails here.
	const markedId = await glyph.getAttribute("data-measure-on");
	expect(markedId).toBeTruthy();
	const target = realScore.locator(`g.measure[id="${markedId}"]`);
	await expect(target).toHaveCount(1);

	const glyphBox = await glyph.boundingBox();
	const targetBox = await target.boundingBox();
	expect(glyphBox).not.toBeNull();
	expect(targetBox).not.toBeNull();
	if (!glyphBox || !targetBox) throw new Error("unreachable");

	console.log(
		`measures=${await measures.count()} markedId=${markedId} glyph=${JSON.stringify(glyphBox)} target=${JSON.stringify(targetBox)}`,
	);

	// Horizontal overlap: the glyph starts within the measure's horizontal span.
	expect(glyphBox.x).toBeGreaterThanOrEqual(targetBox.x - 1);
	expect(glyphBox.x).toBeLessThanOrEqual(targetBox.x + targetBox.width);
	// Vertical: the glyph sits above the staff by GLYPH_OFFSET_PX (28), less the
	// container's 1px border. Asserting the ACTUAL gap, not merely "above":
	// a measure/inject ordering race put this at 7px while still satisfying
	// "above", and the glyph visibly overlapped the staff. A bare inequality
	// cannot see that; this can.
	const gap = targetBox.y - glyphBox.y;
	expect(gap).toBeGreaterThan(24);
	expect(gap).toBeLessThan(30);

	// And the degradation constraint holds on a real score too.
	await expect(realScore).not.toContainText("bars 21");
});
