import AxeBuilder from "@axe-core/playwright";
import { expect, type Page, test } from "@playwright/test";
import {
	FIXTURE_MARKS,
	PIECELESS_DURATION_SECONDS,
} from "../src/lib/practice-preview-fixtures";

// The dev server serves no API, but ScoreRenderer fetches score bytes via
// api.scores.getData() -> /api/scores/:pieceId/data. Fulfil that request from
// the statically-served .mxl instead, which is the pattern already established
// in src/scorehost/score-host.playwright.ts:186. Rendering a real engraving is
// the point of this test; running a real API is not.
const PIECE_ID = "chopin-nocturne-op9-no2";

// SessionTimelineStrip positions every mark with a `left` computed in a
// useLayoutEffect (measured off real DOM rects), not in the initial render --
// the initial render has an empty layout Map, so every glyph starts pinned to
// the strip's own left edge until that effect commits. `page.goto`'s "load"
// resolves once the SSR HTML has arrived, which is BEFORE that effect has
// necessarily run (this app hits a pre-existing, app-wide SSR/client
// hydration mismatch -- reproduces on /signin too, unrelated to marks -- that
// delays when hydration settles). Reading positions right after goto races
// that settle and sees the pre-measurement default. Wait for the real signal
// instead of a fixed sleep: the first glyph's left edge has moved off the
// strip's own left edge, meaning the layout effect has actually committed.
async function waitForTimelineMeasured(page: Page) {
	await page.waitForFunction(() => {
		const strip = document.querySelector("[data-testid='session-timeline']");
		const btn = strip?.querySelector("button[aria-expanded]");
		if (!strip || !btn) return false;
		return (
			btn.getBoundingClientRect().left !== strip.getBoundingClientRect().left
		);
	});
}

test("no timeline mark covers another, so every mark stays tappable", async ({
	page,
}) => {
	await page.goto("/practice-preview");
	await waitForTimelineMeasured(page);

	const boxes = await page.evaluate(() =>
		[...document.querySelectorAll("button[aria-expanded]")]
			.filter((b) => !b.hasAttribute("data-measure-on"))
			.map((b) => {
				const r = b.getBoundingClientRect();
				return {
					label: b.getAttribute("aria-label"),
					l: r.left,
					r: r.right,
					t: r.top,
					// Comparing vertical RANGES, not `top` equality. Glyph height
					// is not guaranteed uniform, so two marks in different lanes
					// can still overlap — an equality check is blind to exactly
					// the case lane packing does not already cover.
					b: r.bottom,
				};
			}),
	);
	expect(boxes.length).toBeGreaterThan(1);

	const collisions: string[] = [];
	for (let i = 0; i < boxes.length; i++) {
		for (let j = i + 1; j < boxes.length; j++) {
			const a = boxes[i];
			const b = boxes[j];
			if (a.t < b.b && b.t < a.b && a.l < b.r && b.l < a.r) {
				collisions.push(`${a.label} <-> ${b.label}`);
			}
		}
	}
	// Positioning marks purely by elapsed time put four pairs on top of each
	// other, and a covered mark cannot be clicked at all — the click lands on
	// whichever sibling is on top. jsdom cannot see this: fireEvent dispatches
	// straight at the node and never hit-tests.
	expect(collisions).toEqual([]);

	// The real proof: Playwright refuses to click an intercepted element, so
	// this line is what actually failed before lane packing existed.
	await page
		.locator("button[aria-expanded]:not([data-measure-on])")
		.first()
		.click();
});

// Collision is a RELATIONAL question ("does any mark overlap another?");
// containment is a different one ("does each mark fit inside its parent?").
// Pairwise geometry never references the container, so a collision test passes
// happily while every mark sits outside the box. Both are needed.
for (const width of [1280, 760, 640]) {
	test(`every timeline mark stays inside the strip at ${width}px`, async ({
		page,
	}) => {
		await page.setViewportSize({ width, height: 900 });
		await page.goto("/practice-preview");

		const result = await page.evaluate(() => {
			const strip = document.querySelector("[data-testid='session-timeline']");
			if (!strip) throw new Error("timeline strip not found");
			const s = strip.getBoundingClientRect();
			const escapees = [...strip.querySelectorAll("button[aria-expanded]")]
				.map((el) => {
					const r = el.getBoundingClientRect();
					return {
						label: el.getAttribute("aria-label"),
						overflowRight: Math.round(r.right - s.right),
						overflowLeft: Math.round(s.left - r.left),
					};
				})
				.filter((m) => m.overflowRight > 0 || m.overflowLeft > 0);
			return {
				escapees,
				scrollW: document.documentElement.scrollWidth,
				clientW: document.documentElement.clientWidth,
			};
		});

		// A mark anchored at 85% of the session ran 36px past the strip's right
		// edge at every viewport, because `left` is chosen from elapsed time
		// before the glyph's width is known.
		expect(result.escapees).toEqual([]);
		// And an overflowing mark widens the document, so the whole page scrolls
		// sideways on a narrow viewport.
		expect(result.scrollW).toBe(result.clientW);
	});
}

test("a mark sits at its share of the session duration", async ({ page }) => {
	await page.setViewportSize({ width: 1280, height: 900 });
	await page.goto("/practice-preview");
	await waitForTimelineMeasured(page);

	// Moved here from SessionTimelineStrip.test.tsx: position is derived from
	// the measured strip width, and jsdom reports every width as 0, so only a
	// real layout engine can produce this fact.
	const offset = await page.evaluate(() => {
		const strip = document.querySelector("[data-testid='session-timeline']");
		if (!strip) throw new Error("timeline strip not found");
		const s = strip.getBoundingClientRect();
		const glyph = [...strip.querySelectorAll("button[aria-expanded]")].find(
			(b) => b.getAttribute("aria-label")?.includes("Pedaling"),
		);
		if (!glyph) throw new Error("pedaling mark not found");
		const r = glyph.getBoundingClientRect();
		return { left: r.left - s.left, stripWidth: s.width };
	});

	const pedalingMark = FIXTURE_MARKS.find((m) => m.dimension === "pedaling");
	if (!pedalingMark) throw new Error("no pedaling fixture mark");

	// The pedaling fixture sits well clear of the right edge, so clamping
	// leaves it exactly where elapsed time put it.
	expect(offset.left).toBeCloseTo(
		(pedalingMark.anchor.atSeconds / PIECELESS_DURATION_SECONDS) *
			offset.stripWidth,
		0,
	);
});

test("the preview contributes no second main landmark", async ({ page }) => {
	await page.goto("/practice-preview");
	// The layout already provides <main>; a route that adds its own nests two,
	// which degrades landmark navigation. The axe gate runs color-contrast only
	// and is structurally unable to see this.
	expect(await page.locator("main").count()).toBe(1);
});

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

	await page.goto("/practice-preview");

	// .first(): practice-preview mounts two ScoreStand instances (the
	// standalone top-third one and PracticeMode's own, both against the same
	// piece), so this testid matches twice. Scoping to the first keeps every
	// downstream single-element assertion (toHaveCount(1), boundingBox) valid
	// without depending on Verovio giving the two renders distinct ids.
	const realScore = page.locator("[data-testid='score-stand-page']").first();
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

test("PracticeMode's Stop control never overlaps a sub-surface's own top control", async ({
	page,
}) => {
	await page.goto("/practice-preview");

	const scope = page.locator("[data-testid='practice-mode-preview']");
	const boxes = await scope.evaluate((root) => {
		const find = (label: RegExp) =>
			[...root.querySelectorAll("button")].find((b) =>
				label.test(b.getAttribute("aria-label") ?? b.textContent ?? ""),
			);
		const named: readonly [string, HTMLButtonElement | undefined][] = [
			["stop", find(/stop recording/i)],
			["dismiss", find(/dismiss/i)],
			["metronome", find(/metronome/i)],
		];
		return named
			.filter((entry): entry is [string, HTMLButtonElement] => entry[1] != null)
			.map(([name, el]) => {
				const r = el.getBoundingClientRect();
				return { name, l: r.left, r: r.right, t: r.top, b: r.bottom };
			});
	});

	expect(boxes.map((b) => b.name).sort()).toEqual([
		"dismiss",
		"metronome",
		"stop",
	]);

	const collisions: string[] = [];
	for (let i = 0; i < boxes.length; i++) {
		for (let j = i + 1; j < boxes.length; j++) {
			const a = boxes[i];
			const b = boxes[j];
			if (a.t < b.b && b.t < a.b && a.l < b.r && b.l < a.r) {
				collisions.push(`${a.name} <-> ${b.name}`);
			}
		}
	}
	expect(collisions).toEqual([]);
});

// #157 added these two color-contrast cases to tests/a11y.spec.ts against
// /marks-preview; #158 deletes that route. practice-preview.tsx is
// import.meta.env.DEV-gated and playwright.a11y.config.ts serves a
// production build (DEV === false there), so the cases cannot move with
// the route name alone -- they have to run under a config where DEV is
// true, which is exactly what this file's webServer now is. This is the
// only place mark contrast is verified -- never assert it from vitest.
for (const theme of ["light", "dark"] as const) {
	test(`practice-preview has no color-contrast violations (${theme})`, async ({
		page,
	}) => {
		await page.goto("/practice-preview");
		await page.evaluate((t) => {
			document.documentElement.dataset.theme = t;
		}, theme);

		const results = await new AxeBuilder({ page })
			.withRules(["color-contrast"])
			.exclude("[data-axe-exempt]")
			.analyze();

		for (const v of results.violations) {
			for (const node of v.nodes) {
				console.log(
					`[${theme} /practice-preview] ${v.id} :: ${node.target.join(" ")}`,
				);
			}
		}
		expect(results.violations).toEqual([]);
	});
}
